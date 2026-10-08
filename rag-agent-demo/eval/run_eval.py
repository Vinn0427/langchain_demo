"""
Offline Retrieval Evaluation + Latency Benchmark

    python eval/run_eval.py            # 检索评测 + 端到端（含 LLM）评测
    python eval/run_eval.py --no-e2e   # 只跑检索评测（不调用 Chat 模型）

Part 1 Retrieval：对每条 query 直接调用 rag.pipeline.retrieve()，分别评估四个阶段的排序结果：
    dense / sparse(BM25) / hybrid(RRF) / hybrid + rerank
    Recall@K = |relevant ∩ topK| / |relevant|
    MRR      = mean(1 / rank of first relevant)，Top-N 内没有命中记为 0
Part 2 End-to-End：对每条 query 走完整 Agent Graph（stream），统计 e2e latency / TTFT / Gate 分布。

所有数字都来自本次实际运行；样本量很小，p95 仅用于 Demo，不具备生产统计意义。
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from observability.metrics import start_request  # noqa: E402
from rag.pipeline import retrieve, warmup  # noqa: E402

DATASET = Path(__file__).parent / "dataset.json"
KS = (1, 3, 5)


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def latency_line(name: str, values: list) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return f"{name:<22}{'n/a':>10}"
    arr = np.array(values)
    return (
        f"{name:<22}{arr.mean():>10.1f}{np.percentile(arr, 50):>10.1f}"
        f"{np.percentile(arr, 95):>10.1f}{arr.max():>10.1f}{len(arr):>6}"
    )


def latency_table(title: str, rows: dict[str, list]) -> None:
    print(f"\n---------- {title} (ms) ----------")
    print(f"{'stage':<22}{'avg':>10}{'p50':>10}{'p95':>10}{'max':>10}{'n':>6}")
    for name, values in rows.items():
        print(latency_line(name, values))


def run_retrieval_eval(dataset: list[dict]) -> None:
    stages = ("dense", "sparse", "hybrid_rrf", "hybrid_rerank")
    scores = {s: {**{f"recall@{k}": [] for k in KS}, "mrr": []} for s in stages}
    latency = {"query_embedding": [], "dense_retrieval": [], "sparse_retrieval": [],
               "rrf_fusion": [], "hybrid_retrieval": [], "rerank": [], "retrieval_total": []}
    misses = []

    for item in dataset:
        relevant = set(item["relevant_chunk_ids"])
        metrics = start_request()
        result = retrieve(item["query"])
        metrics.finish()

        ranked = {
            "dense": [d.metadata["chunk_id"] for d in result.dense],
            "sparse": [d.metadata["chunk_id"] for d in result.sparse],
            "hybrid_rrf": [d.metadata["chunk_id"] for d in result.fused],
            "hybrid_rerank": [d.metadata["chunk_id"] for d in result.reranked],
        }
        for stage, ids in ranked.items():
            for k in KS:
                scores[stage][f"recall@{k}"].append(recall_at_k(ids, relevant, k))
            scores[stage]["mrr"].append(reciprocal_rank(ids, relevant))
        if not set(ranked["hybrid_rerank"][:1]) & relevant:
            misses.append((item["query"], sorted(relevant), ranked["hybrid_rerank"][:3]))

        for stage in latency:
            if stage != "retrieval_total":
                latency[stage].append(metrics.stage_ms(stage))
        latency["retrieval_total"].append(metrics.total_ms)

    print(f"\n========== Retrieval Quality (n={len(dataset)}) ==========")
    print(f"candidates: dense top {config.DENSE_TOP_K}, sparse top {config.SPARSE_TOP_K}, "
          f"rrf top {config.RRF_TOP_K} (k={config.RRF_K}), rerank top {config.RERANK_TOP_K}")
    header = "".join(f"{f'Recall@{k}':>11}" for k in KS) + f"{'MRR':>9}"
    print(f"{'stage':<16}{header}")
    for stage in stages:
        row = "".join(f"{np.mean(scores[stage][f'recall@{k}']):>11.3f}" for k in KS)
        print(f"{stage:<16}{row}{np.mean(scores[stage]['mrr']):>9.3f}")
    if misses:
        print("\nhybrid_rerank Top-1 未命中：")
        for query, relevant, top3 in misses:
            print(f"  {query}  expected={relevant}  got={top3}")

    latency_table("Retrieval Latency", latency)


def run_e2e_eval(dataset: list[dict]) -> None:
    from main import ask  # 只在端到端评测时加载 Agent / Chat 模型

    rows = {"total(e2e latency)": [], "e2e_ttft": [], "model_ttft": [], "generation": [],
            "agent_decision": [], "hybrid_retrieval": [], "rerank": [], "llm_grader": [], "query_rewrite": []}
    gates, grader_calls, rewrites, tool_called = [], 0, 0, 0

    for i, item in enumerate(dataset, start=1):
        with contextlib.redirect_stdout(io.StringIO()):  # 隐藏节点日志和流式 token
            state, metrics = ask(item["query"], show=False)
        summary = metrics.summary()
        rows["total(e2e latency)"].append(summary["total"])
        for name in rows:
            if name != "total(e2e latency)":
                rows[name].append(summary.get(name))
        if state.get("pending_tool_call_id"):
            tool_called += 1
            gates.append(state["gate_level"])
        grader_calls += "llm_grader" in metrics.stages
        rewrites += len(metrics.stages.get("query_rewrite", []))
        print(f"  [{i:>2}/{len(dataset)}] e2e_ttft={summary['e2e_ttft'] or 0:>7.0f} ms  "
              f"total={summary['total']:>7.0f} ms  gate={state.get('gate_level') or '-':<6}  {item['query']}")

    print(f"\n========== End-to-End (n={len(dataset)}) ==========")
    print(f"tool called: {tool_called}/{len(dataset)}   final gate: "
          + ", ".join(f"{g}={gates.count(g)}" for g in ("high", "medium", "low")))
    print(f"requests that ran LLM Grader: {grader_calls}/{len(dataset)}   query rewrites: {rewrites}")
    print("（latency 统计中 skipped 的阶段不计入样本，n 列为实际执行该阶段的请求数）")
    latency_table("End-to-End Latency", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-e2e", action="store_true", help="只跑检索评测")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    info = warmup()
    print(f"[Eval] dataset={DATASET.name} n={len(dataset)}  Qdrant points={info['points']}  "
          f"reranker={config.RERANKER_MODEL}  embedding={config.EMBEDDING_MODEL}")

    run_retrieval_eval(dataset)
    if not args.no_e2e:
        print(f"\n[Eval] running full Agent Graph on {len(dataset)} queries (chat model: {config.CHAT_MODEL}) ...")
        run_e2e_eval(dataset)

    print(f"\n注意：样本量只有 {len(dataset)} 条，p95 仅用于 Demo，不具备生产统计意义；"
          "延迟包含公网调用 Embedding / Chat API 的网络波动。")


if __name__ == "__main__":
    main()
