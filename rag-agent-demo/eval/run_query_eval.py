"""
Query Understanding / Contextual Rewrite Evaluation + Routing Mode A/B

    python eval/run_query_eval.py

eval/query_dataset.json：多轮指代、省略、实体切换、约束继承、ID 保留、数字 / 否定保留、无需改写、话题切换。

两种模式比较的是"第一次检索用的 Query 是否正确地被上下文补全"（RAG Query Quality）：
    query_understanding：standalone_query（QU 输出），以及第一次 search 实际使用的 Query
    v3_agent           ：Agent 写进 search_knowledge_base 参数的 Query（V3 中由 Agent Decision 同时完成改写）
指标：
    intent_accuracy        intent ∈ expected_intents（v3_agent 由 Tool 反推）
    query_must_contain     第一次检索 Query（DIRECT 样本用 standalone_query）包含 must_contain 且不含 must_not_contain
    id_preservation        expected_chunk_id / expected_document_id 出现在 Tool 参数中
    first_retrieval_recall@5  第一次检索（Rewrite 之前）的 Rerank Top-5 对 relevant_chunk_ids 的 Recall
"""
import argparse

from common import contains_all, load, norm, print_latency_table, run_agent_case, save_report, stats  # noqa: I001

from observability.versions import runtime_versions  # noqa: E402
from rag.pipeline import warmup  # noqa: E402

MODES = ("v3_agent", "query_understanding")


def first_search(run: dict) -> dict:
    return next((e for e in run["tool_log"] if e["name"] == "search_knowledge_base"), {})


def score(case: dict, run: dict) -> dict:
    search = first_search(run)
    direct_case = case["expected_intents"] == ["DIRECT"]
    if search:  # tool_log 中 search 的 args.query = 第一次检索实际使用的 Query（两种模式一致）
        query_text = (search.get("args") or {}).get("query") or ""
    elif direct_case:  # DIRECT：QU 模式检查 standalone_query；v3_agent 没有改写这一步，记为 n/a
        query_text = run["standalone_query"] or "" if run["mode"] == "query_understanding" else None
    else:  # 精确查找类：检查 Tool 参数（+ QU 的 standalone_query）里是否保留了实体 / ID
        query_text = " ".join([str(e.get("args")) for e in run["tool_log"]] + [run["standalone_query"] or ""])
    must_ok = None if query_text is None else (contains_all(query_text, case["must_contain"]) and not any(
        norm(x) in norm(query_text) for x in case["must_not_contain"]))
    query_text = query_text or ""
    args_text = " ".join(str(e.get("args")) for e in run["tool_log"])
    expected_id = case.get("expected_chunk_id") or case.get("expected_document_id")
    recall = None
    if case.get("relevant_chunk_ids") and search:
        top5 = (search.get("first_retrieval_ids") or [])[:5]
        recall = len(set(top5) & set(case["relevant_chunk_ids"])) / len(case["relevant_chunk_ids"])
    elif case.get("relevant_chunk_ids"):
        recall = 0.0
    return {
        "intent_ok": run["intent"] in case["expected_intents"],
        "query_text": query_text,
        "query_ok": must_ok,
        "id_ok": (expected_id in args_text) if expected_id else None,
        "first_retrieval_recall@5": recall,
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def run(modes=MODES) -> dict:
    dataset = load("query_dataset.json")
    report = {"versions": runtime_versions(), "modes": {}}
    for mode in modes:
        print(f"\n[QueryEval] mode={mode} n={len(dataset)}")
        rows = []
        for i, case in enumerate(dataset, 1):
            r = run_agent_case(case["query"], case.get("history"), mode)
            s = score(case, r)
            rows.append({"case": case, "score": s, "run": {k: v for k, v in r.items() if k != "log_tail"}})
            print(f"  [{i:>2}/{len(dataset)}] {'OK ' if s['intent_ok'] and s['query_ok'] is not False else 'ERR'} "
                  f"intent={r['intent']!s:<17} query={s['query_text'][:50]!r}  [{case['category']}] {case['query']}")
        scores = [x["score"] for x in rows]
        report["modes"][mode] = {
            "metrics": {
                "n": len(dataset),
                "intent_accuracy": _mean(s["intent_ok"] for s in scores),
                "query_must_contain_rate": _mean(s["query_ok"] for s in scores),
                "id_preservation_rate": _mean(s["id_ok"] for s in scores),
                "first_retrieval_recall@5": _mean(s["first_retrieval_recall@5"] for s in scores),
                "qu_contract_violations": sum(bool(x["run"].get("qu_violations")) for x in rows),
                "qu_fallbacks": sum(x["run"].get("qu_status") == "fallback" for x in rows),
            },
            "latency": {
                "e2e_ttft": stats(x["run"]["latency"]["e2e_ttft"] for x in rows),
                "query_understanding": stats(x["run"]["latency"]["query_understanding"] for x in rows),
                "agent_decision_first": stats(x["run"]["latency"]["first_agent_decision"] for x in rows),
                "tool_selection": stats(x["run"]["latency"]["tool_selection"] for x in rows),
            },
            "rows": rows,
        }
    return report


def print_report(report: dict) -> None:
    modes = list(report["modes"])
    print("\n========== Query Understanding / Contextual Rewrite ==========")
    print(f"{'metric':<28}" + "".join(f"{m:>22}" for m in modes))
    for k in report["modes"][modes[0]]["metrics"]:
        print(f"{k:<28}" + "".join(f"{str(report['modes'][m]['metrics'][k]):>22}" for m in modes))
    for m in modes:
        lat = report["modes"][m]["latency"]
        print_latency_table(f"Latency [{m}]", {})
        for k, v in lat.items():
            if v["n"]:
                print(f"{k:<26}{v['avg']:>10.1f}{v['p50']:>10.1f}{v['p95']:>10.1f}{v['max']:>10.1f}{v['n']:>6}")
    for m in modes:
        print(f"\n[{m}] 第一次检索 Query / standalone_query：")
        for x in report["modes"][m]["rows"]:
            s = x["score"]
            print(f"  {'OK ' if s['intent_ok'] and s['query_ok'] is not False else 'ERR'} {x['case']['query']!r:<34} → {s['query_text']!r}"
                  f"  intent={x['run']['intent']} recall@5={s['first_retrieval_recall@5']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()
    warmup()
    report = run(args.modes)
    print_report(report)
    path = save_report("query_eval.json", report)
    print(f"\n[Report] {path}")


if __name__ == "__main__":
    main()
