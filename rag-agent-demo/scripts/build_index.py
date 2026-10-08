"""
Offline Indexing 入口：构建完成后退出。

    docker compose up -d
    python scripts/build_index.py

文档有变更时重新运行即可（全量重建 collection）。
"""
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag.indexing import build_index  # noqa: E402


def main() -> None:
    print(f"[Index] Qdrant: {config.QDRANT_URL}  collection: {config.QDRANT_COLLECTION}")
    start = perf_counter()
    result = build_index()
    total = perf_counter() - start

    print("\n========== Indexing ==========")
    for stage, seconds in result["timings"].items():
        print(f"{stage:<20}{seconds * 1000:>10.1f} ms")
    print(f"{'total':<20}{total * 1000:>10.1f} ms")
    print("==============================")
    print(f"chunks: {', '.join(result['chunks'])}")


if __name__ == "__main__":
    main()
