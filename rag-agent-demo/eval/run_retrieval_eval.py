"""
Retrieval Eval（替代 V3 的 eval/run_eval.py 检索部分；数据集升级为 eval/retrieval_dataset.json，含 V3 原 24 条）

    python eval/run_retrieval_eval.py      # 在 serving alias 上评估 dense / sparse / hybrid_rrf / hybrid_rerank
"""
from common import evaluate_retrieval, load, print_latency_table, print_retrieval_summary, save_report  # noqa: I001

from observability.versions import runtime_versions  # noqa: E402
from rag.pipeline import warmup  # noqa: E402


def run() -> dict:
    info = warmup()
    dataset = load("retrieval_dataset.json")
    result = evaluate_retrieval(dataset)
    result["versions"] = runtime_versions(info["collection"])
    return result


def main() -> None:
    result = run()
    print_retrieval_summary(f"========== Retrieval Quality (n_labeled={result['n_labeled']}, "
                            f"collection={result['versions']['collection']}) ==========", result["summary"])
    for m in result["misses"]:
        print(f"  miss@1: {m['query']} expected={m['expected']} got={m['got_top3']}")
    print(f"rerank top1: positive={result['rerank_top1']['positive']} no_answer={result['rerank_top1']['no_answer']}")
    print("\n---------- Retrieval Latency (ms) ----------")
    for k, v in result["latency"].items():
        print(f"{k:<20} avg={v['avg']} p50={v['p50']} p95={v['p95']} max={v['max']}")
    print(f"[Report] {save_report('retrieval_eval.json', result)}")


if __name__ == "__main__":
    main()
