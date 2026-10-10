"""
Chunk Strategy A/B/C Evaluation

    python eval/run_chunk_eval.py

对 A（raw）/ B（文档标题 + 小节标题 + raw）/ C（文档标题 + 完整标题路径 + raw）分别构建独立的测试 collection
（rag_eval_chunk_<s>_<fingerprint>，走同一套增量 Indexing：第二次运行时未变化的文件直接 skip），
用同一份 eval/retrieval_dataset.json 比较 Dense / Sparse / Hybrid / Rerank 的 Recall@1/3/5 与 MRR，
并输出 Chunk Analysis（chunk_count / token 长度分布 / too_short / too_long）。

切分边界在三种策略下完全相同（chunk_id 一致），只有 retrieval_text 不同，所以差异只来自标题信息。
"""
import contextlib
import io

from common import STAGES, evaluate_retrieval, load, print_retrieval_summary, save_report  # 先 import：把项目根目录加入 sys.path

import config  # noqa: E402
from observability.versions import runtime_versions  # noqa: E402
from rag.chunking import STRATEGIES, analyze_chunks, chunk_document  # noqa: E402
from rag.indexing import collection_name, full_build, incremental_update, pipeline_fingerprint, scan_documents  # noqa: E402
from rag.manifest import Manifest  # noqa: E402
from rag.reranker import get_reranker  # noqa: E402
from rag.store import get_client  # noqa: E402


def ensure_collection(strategy: str, manifest: Manifest) -> tuple[str, dict]:
    fp, _ = pipeline_fingerprint(strategy)
    name = collection_name(strategy, fp, prefix="rag_eval_chunk")
    meta = manifest.collection(name)
    with contextlib.redirect_stdout(io.StringIO()):
        if meta and meta["status"] != "building" and get_client().collection_exists(name):
            stats = incremental_update(name, strategy, manifest=manifest)
        else:
            stats = full_build(name, strategy, manifest=manifest)
    return name, stats.as_dict()


def main() -> None:
    dataset = load("retrieval_dataset.json")
    get_reranker()
    manifest = Manifest()
    report = {"versions": runtime_versions(), "dataset_n": len(dataset), "strategies": {}}

    for strategy in STRATEGIES:
        name, stats = ensure_collection(strategy, manifest)
        chunks = [c for p in scan_documents(config.DATA_DIR).values() for c in chunk_document(p, strategy)]
        analysis = {"raw_text": analyze_chunks(chunks, "raw_text"), "retrieval_text": analyze_chunks(chunks, "retrieval_text")}
        print(f"\n[Chunk {strategy}] collection={name} index_mode={stats['mode']} "
              f"embedded={stats['chunks_embedded']} skipped={len(stats['unchanged'])} docs")
        result = evaluate_retrieval(dataset, collection=name)
        report["strategies"][strategy] = {"collection": name, "index_stats": stats, "analysis": analysis, "eval": result}
        print_retrieval_summary(f"========== Strategy {strategy} (n_labeled={result['n_labeled']}) ==========",
                                result["summary"])

    # ---- Chunk Analysis ----
    a = report["strategies"]["A"]["analysis"]["raw_text"]
    print("\n========== Chunk Analysis (raw_text, jieba tokens) ==========")
    print(f"chunk_count={a['chunk_count']}  avg={a['token_avg']}  p50={a['token_p50']}  p95={a['token_p95']}  "
          f"min={a['token_min']}  max={a['token_max']}")
    print(f"too_short(<{a['too_short_threshold']}) ratio={a['too_short_ratio']} {a['too_short_chunks']}")
    print(f"too_long(>{a['too_long_threshold']}) ratio={a['too_long_ratio']} {a['too_long_chunks']}")
    for s in STRATEGIES:
        r = report["strategies"][s]["analysis"]["retrieval_text"]
        print(f"retrieval_text {s}: avg={r['token_avg']} p50={r['token_p50']} p95={r['token_p95']} "
              f"too_short_ratio={r['too_short_ratio']}")

    # ---- 对比表 ----
    print("\n========== Chunk Strategy Comparison ==========")
    print(f"{'stage':<16}{'metric':<11}" + "".join(f"{s:>9}" for s in STRATEGIES))
    for stage in STAGES:
        for metric in ("recall@1", "recall@3", "recall@5", "mrr"):
            values = [report["strategies"][s]["eval"]["summary"][stage][metric] for s in STRATEGIES]
            print(f"{stage:<16}{metric:<11}" + "".join(f"{v:>9.3f}" for v in values))

    print("\n---------- hybrid_rerank MRR by category ----------")
    cats = report["strategies"]["A"]["eval"]["by_category"].keys()
    print(f"{'category':<18}{'n':>4}" + "".join(f"{s:>9}" for s in STRATEGIES))
    for c in cats:
        n = report["strategies"]["A"]["eval"]["by_category"][c]["n"]
        values = [report["strategies"][s]["eval"]["by_category"][c]["hybrid_rerank"]["mrr"] for s in STRATEGIES]
        print(f"{c:<18}{n:>4}" + "".join(f"{v:>9.3f}" for v in values))

    print("\n---------- rerank top1 score (Gate 校准参考) ----------")
    for s in STRATEGIES:
        t = report["strategies"][s]["eval"]["rerank_top1"]
        print(f"{s}: positive {t['positive']}  no_answer {t['no_answer']}")

    for s in STRATEGIES:
        misses = report["strategies"][s]["eval"]["misses"]
        print(f"\n[{s}] hybrid_rerank Top-1 未命中 {len(misses)} 条：")
        for m in misses:
            print(f"  {m['query']}  expected={m['expected']}  got={m['got_top3']}")

    # 决策规则（事先固定，不看结果再改）：hybrid_rerank MRR 最高者；平手看 Recall@1，再平手看 hybrid_rrf MRR
    def key(s):
        e = report["strategies"][s]["eval"]["summary"]
        return (e["hybrid_rerank"]["mrr"], e["hybrid_rerank"]["recall@1"], e["hybrid_rrf"]["mrr"])

    best = max(STRATEGIES, key=key)
    report["selected"] = best
    report["selection_rule"] = "max(hybrid_rerank MRR, hybrid_rerank Recall@1, hybrid_rrf MRR)"
    print(f"\n[Decision] rule={report['selection_rule']} → selected strategy {best}  "
          f"(current config CHUNK_STRATEGY={config.CHUNK_STRATEGY})")
    path = save_report("chunk_strategy.json", report)
    print(f"[Report] {path}")


if __name__ == "__main__":
    main()
