"""
Offline Indexing 入口（V4：默认增量）。

    docker compose up -d
    python scripts/build_index.py                  # 增量：未变化的文件直接 skip（0 chunk / 0 embedding / 0 写入）
    python scripts/build_index.py --strategy B     # 换 Chunk 策略 → fingerprint 变化 → Blue-Green 新 collection
    python scripts/build_index.py --rebuild        # 强制 Blue-Green 全量重建（例如 BM25 avgdl drift 过大）
    python scripts/build_index.py --no-switch      # 只构建 / 更新 green collection，不切换 alias
    python scripts/build_index.py --no-eval        # 切换前跳过 Retrieval Eval Gate（不建议）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.indexing import build  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["A", "B", "C"], default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-switch", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--json", action="store_true", help="最后输出完整 JSON 报告")
    args = parser.parse_args()

    print(f"[Index] Qdrant: {config.QDRANT_URL}  alias: {config.QDRANT_ALIAS}  data: {config.DATA_DIR}")
    report = build(strategy=args.strategy, force_rebuild=args.rebuild, switch=not args.no_switch,
                   eval_gate=not args.no_eval)
    s = report["stats"]
    print("\n========== Indexing ==========")
    print(f"mode                 {s['mode']}  ({s['collection']})")
    print(f"documents scanned    {s['documents_scanned']}")
    print(f"added / modified     {s['added']} / {s['modified']}")
    print(f"unchanged (skipped)  {s['unchanged']}")
    print(f"deleted              {s['deleted']}")
    print(f"chunks chunked       {s['chunks_chunked']}")
    print(f"chunks embedded      {s['chunks_embedded']}   (embedding requests: {s['embedding_requests']})")
    print(f"dense vectors reused {s['chunks_vector_reused']}")
    print(f"points upserted      {s['points_upserted']}   points deleted: {s['points_deleted']}")
    print(f"points in collection {s['points_in_collection']}")
    print(f"BM25 avgdl           frozen={s['bm25_avgdl_frozen']} actual={s['bm25_avgdl_actual']} drift={s['bm25_avgdl_drift']}")
    for stage, seconds in s["timings"].items():
        print(f"{stage:<21}{seconds * 1000:>9.1f} ms")
    print(f"alias switched       {report['switched']}")
    print("==============================")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
