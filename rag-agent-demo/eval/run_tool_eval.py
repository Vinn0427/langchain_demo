"""
Tool Calling Evaluation + Routing Mode A/B（v3_agent vs query_understanding）

    python eval/run_tool_eval.py                       # 两种模式都跑
    python eval/run_tool_eval.py --modes query_understanding

每条样本走完整 Agent Graph（真实 LLM / 真实 Tool / 真实 Qdrant），从 State.tool_log 和 Trace 中统计：

    Intent Accuracy             预测 intent ∈ {expected_intent} ∪ also_acceptable_intents
                                （v3_agent 没有显式 intent：由实际调用的 Tool 反推，见 common.infer_intent）
    Candidate Tool Recall       |expected_tools ∩ candidate_tools| / |expected_tools|（v3_agent 绑定全部 Tool，恒为 1）
    Tool Selection Accuracy     实际调用的 Tool 集合 == expected_tools（或 also_acceptable），且调用次数满足 expected_call_count
    First Tool Accuracy         第一个调用的 Tool == first_tool（默认 expected_tools[0]）；DIRECT 样本 = 没有调用任何 Tool
    Tool Argument Valid Rate    通过 Pydantic Args Schema 校验的调用 / 全部调用（不含 Loop Safety 拦截）
    Tool Execution Success Rate 参数合法的调用中，没有 TIMEOUT / DEPENDENCY_ERROR / INTERNAL_ERROR 的比例
                                （NOT_FOUND 是正常业务结果，计为执行成功）
    Unnecessary Tool Call Rate  expected ⊆ called 但多调了别的 Tool，或 DIRECT 样本调用了 Tool
    Wrong Tool Rate             expected 非空，但 expected 没有被全部调用
    No-Tool / Direct Accuracy   DIRECT 样本中没有调用任何 Tool 的比例
    Multi-Tool Task Success     multi_tool 样本：Tool Selection 正确 + 回答包含 answer_must_contain + 无系统错误
    Expected Args Accuracy      给出 expected_args 的样本中，参数与期望一致的比例
    No-Answer Correct           expect_no_answer 样本中最终回答为固定 No-Answer 文案的比例
    Source Integrity            Sources 中的 ID 全部来自 ToolResult；回答正文中没有 Tool 输出里不存在的 chunk_id
"""
import argparse

from common import (contains_all, load, print_latency_table, print_stability, rate, run_agent_case,  # noqa: I001
                    save_report, stability_metrics, stats)

import config  # noqa: E402
from observability.versions import runtime_versions  # noqa: E402
from rag.pipeline import warmup  # noqa: E402

MODES = ("v3_agent", "query_understanding")
SYSTEM_ERRORS = {"TIMEOUT", "DEPENDENCY_ERROR", "INTERNAL_ERROR"}
LOOP_BLOCKS = {"DUPLICATE_TOOL_CALL", "TOOL_BUDGET_EXCEEDED"}


def _args_match(case: dict, tool_log: list[dict]) -> bool:
    for tool, expected in case.get("expected_args", {}).items():
        calls = [e for e in tool_log if e["name"] == tool and e["args_valid"]]
        if not calls:
            return False
        args = calls[0]["args"] or {}
        for key, value in expected.items():
            if key == "query_contains":
                if not contains_all(args.get("query", ""), value):
                    return False
            elif args.get(key) != value:
                return False
    return True


def score_case(case: dict, run: dict) -> dict:
    expected = set(case["expected_tools"])
    requested = run["tools_requested"]
    called = set(requested)
    acceptable = [expected] + [set(a) for a in case.get("also_acceptable", [])]
    counts_ok = all(requested.count(t) >= n for t, n in case.get("expected_call_count", {}).items())
    selection_ok = called in acceptable and counts_ok
    first_expected = case.get("first_tool") or (case["expected_tools"][0] if case["expected_tools"] else None)
    if not expected:
        first_ok = not requested
    else:
        first_ok = bool(requested) and (requested[0] == first_expected or
                                        any(requested[0] in a for a in acceptable if len(a) == 1))
    intents = [case["expected_intent"]] + case.get("also_acceptable_intents", [])
    candidates = run["candidate_tools"] if run["mode"] == "query_understanding" else list(called | expected)
    if run["mode"] == "query_understanding" and run.get("qu_status") == "fallback":
        candidates = run["candidate_tools"]
    cand_recall = (len(expected & set(candidates or [])) / len(expected)) if expected else None
    system_error = any(e["error_code"] in SYSTEM_ERRORS for e in run["tool_log"]) or run["final_answer_kind"] == "llm_error"
    must = case.get("answer_must_contain")
    answer_ok = contains_all(run["answer_body"], must) if must else None
    any_alt = any(called == a for a in acceptable)
    return {
        "intent_ok": run["intent"] in intents,
        "candidate_recall": cand_recall,
        "selection_ok": selection_ok,
        "first_ok": first_ok,
        "unnecessary": (not expected and bool(called)) or (bool(expected) and expected <= called and not any_alt),
        "wrong": bool(expected) and not expected <= called and not any_alt,
        "args_ok": _args_match(case, run["tool_log"]) if case.get("expected_args") else None,
        "answer_ok": answer_ok,
        "system_error": system_error,
        "no_answer_ok": (run["final_answer_kind"] == "no_answer") if case.get("expect_no_answer") else None,
        "not_found_ok": (any(e["error_code"] == "NOT_FOUND" for e in run["tool_log"])) if case.get("expect_not_found") else None,
        "multi_ok": (selection_ok and (answer_ok is not False) and not system_error) if case["category"] == "multi_tool" else None,
        "source_integrity_ok": not run["sources_not_from_tools"] and not run["hallucinated_ids_in_answer"],
    }


def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def aggregate(dataset: list[dict], runs: list[dict], scores: list[dict]) -> dict:
    calls = [e for r in runs for e in r["tool_log"] if e["error_code"] not in LOOP_BLOCKS]
    valid_calls = [e for e in calls if e["args_valid"]]
    direct = [s for c, s in zip(dataset, scores) if not c["expected_tools"]]
    metrics = {
        "n": len(dataset),
        "intent_accuracy": _mean(s["intent_ok"] for s in scores),
        "candidate_tool_recall": _mean(s["candidate_recall"] for s in scores),
        "tool_selection_accuracy": _mean(s["selection_ok"] for s in scores),
        "first_tool_accuracy": _mean(s["first_ok"] for s in scores),
        "tool_argument_valid_rate": rate(len(valid_calls), len(calls)),
        "tool_execution_success_rate": rate(sum(e["error_code"] not in SYSTEM_ERRORS for e in valid_calls), len(valid_calls)),
        "tool_business_success_rate": rate(sum(e["success"] for e in valid_calls), len(valid_calls)),
        "unnecessary_tool_call_rate": _mean(s["unnecessary"] for s in scores),
        "wrong_tool_rate": _mean(s["wrong"] for s in scores),
        "direct_accuracy": _mean(not r["tools_requested"] for c, r in zip(dataset, runs) if not c["expected_tools"]) if direct else None,
        "multi_tool_task_success_rate": _mean(s["multi_ok"] for s in scores),
        "expected_args_accuracy": _mean(s["args_ok"] for s in scores),
        "answer_contains_rate": _mean(s["answer_ok"] for s in scores),
        "no_answer_correct_rate": _mean(s["no_answer_ok"] for s in scores),
        "not_found_handled_rate": _mean(s["not_found_ok"] for s in scores),
        "source_integrity_rate": _mean(s["source_integrity_ok"] for s in scores),
        "total_tool_calls": len(calls),
    }
    by_cat: dict[str, dict] = {}
    for case, s in zip(dataset, scores):
        d = by_cat.setdefault(case["category"], {"n": 0, "intent_ok": 0, "selection_ok": 0})
        d["n"] += 1
        d["intent_ok"] += s["intent_ok"]
        d["selection_ok"] += s["selection_ok"]
    metrics["by_category"] = by_cat
    latency = {
        "e2e_ttft_direct": [r["latency"]["e2e_ttft"] for c, r in zip(dataset, runs) if c["category"] == "direct"],
        "e2e_ttft_rag": [r["latency"]["e2e_ttft"] for c, r in zip(dataset, runs)
                         if c["category"] in ("knowledge_search", "confusable") and r["tools_requested"] == ["search_knowledge_base"]],
        "e2e_ttft_exact_lookup": [r["latency"]["e2e_ttft"] for c, r in zip(dataset, runs)
                                  if c["category"] in ("document_lookup", "chunk_lookup", "document_list", "index_status")],
        "total_all": [r["latency"]["total"] for r in runs],
        "query_understanding": [r["latency"]["query_understanding"] for r in runs],
        "agent_decision_first": [r["latency"]["first_agent_decision"] for r in runs],
        "tool_selection_latency": [r["latency"]["tool_selection"] for r in runs],
        "tool_execution_latency_rag": [e["latency_ms"] for r in runs for e in r["tool_log"] if e["name"] == "search_knowledge_base"],
        "tool_execution_latency_exact": [e["latency_ms"] for r in runs for e in r["tool_log"]
                                         if e["name"] != "search_knowledge_base" and e["args_valid"]],
    }
    return {"metrics": metrics, "latency_raw": latency, "latency": {k: stats(v) for k, v in latency.items()}}


def run(modes=MODES, dataset_name: str = "tool_dataset.json") -> dict:
    dataset = load(dataset_name)
    report = {"versions": runtime_versions(), "dataset": dataset_name, "modes": {}}
    for mode in modes:
        print(f"\n[ToolEval] mode={mode}  n={len(dataset)}")
        runs, scores = [], []
        for i, case in enumerate(dataset, 1):
            r = run_agent_case(case["query"], case.get("history"), mode)
            s = score_case(case, r)
            runs.append(r)
            scores.append(s)
            flag = "OK " if s["selection_ok"] and s["intent_ok"] else "ERR"
            print(f"  [{i:>2}/{len(dataset)}] {flag} intent={r['intent']!s:<17} tools={r['tools_requested']}  "
                  f"ttft={r['latency']['e2e_ttft'] or 0:>6.0f}ms  [{case['category']}] {case['query'][:40]}")
        agg = aggregate(dataset, runs, scores)
        agg["stability"] = stability_metrics(runs)
        agg["cases"] = [{"case": c, "score": s, "run": {k: v for k, v in r.items() if k != "log_tail"}}
                        for c, s, r in zip(dataset, scores, runs)]
        report["modes"][mode] = agg
    return report


def print_report(report: dict) -> None:
    modes = list(report["modes"])
    keys = ["intent_accuracy", "candidate_tool_recall", "tool_selection_accuracy", "first_tool_accuracy",
            "tool_argument_valid_rate", "tool_execution_success_rate", "tool_business_success_rate",
            "unnecessary_tool_call_rate", "wrong_tool_rate", "direct_accuracy", "multi_tool_task_success_rate",
            "expected_args_accuracy", "answer_contains_rate", "no_answer_correct_rate", "not_found_handled_rate",
            "source_integrity_rate", "total_tool_calls"]
    print("\n========== Tool Calling Metrics ==========")
    print(f"{'metric':<32}" + "".join(f"{m:>22}" for m in modes))
    for k in keys:
        print(f"{k:<32}" + "".join(f"{str(report['modes'][m]['metrics'][k]):>22}" for m in modes))
    print("\n---------- by category (intent_ok / selection_ok / n) ----------")
    cats = report["modes"][modes[0]]["metrics"]["by_category"]
    for c in cats:
        cells = []
        for m in modes:
            d = report["modes"][m]["metrics"]["by_category"][c]
            cells.append(f"{d['intent_ok']}/{d['selection_ok']}/{d['n']}")
        print(f"{c:<22}" + "".join(f"{x:>22}" for x in cells))
    for m in modes:
        print_latency_table(f"Latency [{m}]", report["modes"][m]["latency_raw"])
        print_stability(f"Stability [{m}]", report["modes"][m]["stability"])
    for m in modes:
        bad = [x for x in report["modes"][m]["cases"] if not (x["score"]["selection_ok"] and x["score"]["intent_ok"])]
        print(f"\n[{m}] 未通过（intent 或 tool selection）{len(bad)} 条：")
        for x in bad:
            print(f"  {x['case']['query']}  expected={x['case']['expected_intent']}/{x['case']['expected_tools']}  "
                  f"got={x['run']['intent']}/{x['run']['tools_requested']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()
    info = warmup()
    print(f"[ToolEval] collection={info['collection']} chat_model={config.CHAT_MODEL}")
    report = run(args.modes)
    print_report(report)
    for m in report["modes"].values():
        m.pop("latency_raw", None)
    path = save_report("tool_eval.json", report)
    print(f"\n[Report] {path}")


if __name__ == "__main__":
    main()
