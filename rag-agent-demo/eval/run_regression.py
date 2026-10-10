"""
Regression Suite：统一入口

    python eval/run_regression.py                 # 全部（Tool / Query Eval 只跑默认 ROUTING_MODE）
    python eval/run_regression.py --ab            # Tool / Query Eval 同时跑 v3_agent 与 query_understanding

依次运行：
    1. Retrieval Eval            serving alias 上 dense / sparse / hybrid / rerank 指标
    2. Chunk Strategy Eval       A / B / C 测试 collection（增量：未变化文件直接 skip）
    3. Incremental Verification  新增 / 不变 / 修改 / 删除 / 等价性 / fingerprint（独立测试 collection）
    4. Query Understanding Eval
    5. Tool Calling Eval
    6. No-Answer Eval            来自 Tool Eval 的 no_answer 样本 + Retrieval Eval 中无答案 query 的 rerank 分数
    7. Latency / Stability       来自 Tool Eval 的全部端到端请求
最后输出 summary（带版本信息）并按阈值给出 PASS / FAIL。阈值是本 Demo 的回归基线，不具备普适性。
"""
import argparse
import subprocess
import sys

from common import ROOT, save_report  # noqa: I001

import config  # noqa: E402
from observability.versions import runtime_versions  # noqa: E402

THRESHOLDS = {
    "retrieval.hybrid_rerank.recall@5": 0.90,
    "retrieval.hybrid_rerank.mrr": 0.90,
    "incremental.passed": True,
    "query.intent_accuracy": 0.90,
    "query.query_must_contain_rate": 0.85,
    "tool.intent_accuracy": 0.90,
    "tool.tool_selection_accuracy": 0.90,
    "tool.tool_argument_valid_rate": 0.95,
    "tool.tool_execution_success_rate": 0.98,
    "tool.multi_tool_task_success_rate": 0.75,
    "tool.no_answer_correct_rate": 1.0,
    "tool.source_integrity_rate": 1.0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab", action="store_true")
    args = parser.parse_args()
    modes = ["v3_agent", "query_understanding"] if args.ab else [config.ROUTING_MODE]

    import run_query_eval
    import run_retrieval_eval
    import run_tool_eval

    summary = {"versions": runtime_versions(), "modes": modes}
    print("\n[Regression] 1/7 Retrieval Eval")
    r = run_retrieval_eval.run()
    summary["retrieval"] = {"summary": r["summary"], "misses": r["misses"], "latency": r["latency"],
                            "rerank_top1": {k: v for k, v in r["rerank_top1"].items() if not k.endswith("_raw")}}

    for step, script in (("2/7 Chunk Strategy Eval", "run_chunk_eval.py"),
                         ("3/7 Incremental Verification", "../scripts/verify_incremental.py")):
        print(f"\n[Regression] {step}")
        proc = subprocess.run([sys.executable, str(ROOT / "eval" / script)], cwd=ROOT, capture_output=True, text=True,
                              env={**__import__("os").environ, "TRACE_PRINT": "false"})
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        print(tail)
        summary[script] = {"exit_code": proc.returncode}
    import json
    chunk = json.loads((ROOT / "eval/reports/chunk_strategy.json").read_text(encoding="utf-8"))
    summary["chunk_strategy"] = {"selected": chunk["selected"], "rule": chunk["selection_rule"],
                                 "hybrid_rerank": {s: v["eval"]["summary"]["hybrid_rerank"] for s, v in chunk["strategies"].items()},
                                 "analysis_raw": chunk["strategies"]["A"]["analysis"]["raw_text"]}
    inc = json.loads((ROOT / "eval/reports/incremental_verification.json").read_text(encoding="utf-8"))
    summary["incremental"] = {"passed": inc["passed"], "checks": f"{sum(c['ok'] for c in inc['checks'])}/{len(inc['checks'])}"}

    print("\n[Regression] 4/7 Query Understanding Eval")
    q = run_query_eval.run(modes)
    run_query_eval.print_report(q)
    summary["query"] = {m: v["metrics"] for m, v in q["modes"].items()}
    save_report("query_eval.json", q)

    print("\n[Regression] 5/7 Tool Calling Eval")
    t = run_tool_eval.run(modes)
    run_tool_eval.print_report(t)
    summary["tool"] = {m: {k: v for k, v in x["metrics"].items() if k != "by_category"} for m, x in t["modes"].items()}
    print("\n[Regression] 6/7 No-Answer Eval / 7/7 Latency & Stability")
    summary["no_answer"] = {m: {"no_answer_correct_rate": x["metrics"]["no_answer_correct_rate"],
                                "retrieval_no_answer_rerank_top1": summary["retrieval"]["rerank_top1"]["no_answer"]}
                            for m, x in t["modes"].items()}
    summary["latency"] = {m: x["latency"] for m, x in t["modes"].items()}
    summary["stability"] = {m: x["stability"] for m, x in t["modes"].items()}
    for x in t["modes"].values():
        x.pop("latency_raw", None)
    save_report("tool_eval.json", t)

    main_mode = config.ROUTING_MODE if config.ROUTING_MODE in modes else modes[-1]
    values = {
        "retrieval.hybrid_rerank.recall@5": r["summary"]["hybrid_rerank"]["recall@5"],
        "retrieval.hybrid_rerank.mrr": r["summary"]["hybrid_rerank"]["mrr"],
        "incremental.passed": inc["passed"],
        "query.intent_accuracy": summary["query"][main_mode]["intent_accuracy"],
        "query.query_must_contain_rate": summary["query"][main_mode]["query_must_contain_rate"],
        **{f"tool.{k}": summary["tool"][main_mode][k] for k in
           ("intent_accuracy", "tool_selection_accuracy", "tool_argument_valid_rate", "tool_execution_success_rate",
            "multi_tool_task_success_rate", "no_answer_correct_rate", "source_integrity_rate")},
    }
    checks = {k: {"value": values[k], "threshold": th,
                  "ok": (values[k] is th) if isinstance(th, bool) else (values[k] is not None and values[k] >= th)}
              for k, th in THRESHOLDS.items()}
    summary["checks"] = checks
    summary["passed"] = all(c["ok"] for c in checks.values())

    print(f"\n==================== Regression Summary (mode={main_mode}) ====================")
    for k, c in checks.items():
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {k:<40} {c['value']}  (threshold {c['threshold']})")
    s = summary["stability"][main_mode]
    print(f"  stability: timeout_rate={s['timeout_rate']} retry_rate={s['retry_rate']} "
          f"parse_failure_rate={s['structured_parse_failure_rate']} rewrite_rate={s['query_rewrite_rate']} "
          f"grader_rate={s['grader_trigger_rate']} tool_error_rate={s['tool_error_rate']} "
          f"dup_rate={s['duplicate_tool_call_rate']} no_answer_rate={s['no_answer_rate']}")
    print(f"  versions: chat={config.CHAT_MODEL} embedding={config.EMBEDDING_MODEL} reranker={config.RERANKER_MODEL} "
          f"chunk={config.CHUNK_STRATEGY} collection={summary['versions']['collection']}")
    print(f"  REGRESSION {'PASSED' if summary['passed'] else 'FAILED'}")
    print(f"[Report] {save_report('regression_summary.json', summary)}")
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
