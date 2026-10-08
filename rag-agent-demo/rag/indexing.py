"""
Offline Indexing Pipeline：documents → load → chunk → dense embedding → BM25 sparse vector → Qdrant

只由 scripts/build_index.py 调用。在线 Query Pipeline 不 import 本模块，也不会重新做文档 Embedding。
"""
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from qdrant_client import models

import config
from rag.dense_retriever import embed_texts
from rag.sparse_retriever import bm25_document_vector, tokenize
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, get_client, point_id


@dataclass
class Chunk:
    chunk_id: str      # 例如 redis_001
    document_id: str   # 文件名（不含扩展名），例如 redis
    source: str        # 例如 data/redis.md
    title: str         # 文档一级标题
    topic: str         # chunk 所在的二级标题
    text: str          # chunk 正文（包含 "## topic" 标题行）


def load_documents(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.md"))


def chunk_document(path: Path) -> list[Chunk]:
    """按 Markdown 二级标题切分：每个 "## " 小节成为一个 chunk。"""
    markdown = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", markdown, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    sections = re.split(r"^(?=## )", markdown, flags=re.M)[1:]  # 第 0 段是一级标题，丢弃

    chunks = []
    for i, section in enumerate(sections, start=1):
        topic = section.splitlines()[0].removeprefix("## ").strip()
        chunks.append(
            Chunk(
                chunk_id=f"{path.stem}_{i:03d}",
                document_id=path.stem,
                source=str(path.relative_to(config.PROJECT_ROOT)),
                title=title,
                topic=topic,
                text=section.strip(),
            )
        )
    return chunks


def build_index() -> dict:
    timings = {}

    t = perf_counter()
    paths = load_documents(config.DATA_DIR)
    chunks = [chunk for path in paths for chunk in chunk_document(path)]
    timings["load_and_chunk"] = perf_counter() - t
    print(f"[Index] loaded {len(paths)} documents → {len(chunks)} chunks")

    t = perf_counter()
    dense_vectors = embed_texts([c.text for c in chunks])
    timings["dense_embedding"] = perf_counter() - t
    dim = len(dense_vectors[0])
    print(f"[Index] dense embedding: {len(dense_vectors)} vectors, dim={dim} ({config.EMBEDDING_MODEL})")

    t = perf_counter()
    tokenized = [tokenize(c.text) for c in chunks]
    avgdl = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    sparse_vectors = [bm25_document_vector(tokens, avgdl) for tokens in tokenized]
    timings["bm25_encoding"] = perf_counter() - t
    print(f"[Index] BM25 sparse vectors: avgdl={avgdl:.1f} tokens, k1={config.BM25_K1}, b={config.BM25_B}")

    t = perf_counter()
    client = get_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)  # 全量重建：避免旧 chunk 残留
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    points = [
        models.PointStruct(
            id=point_id(chunk.chunk_id),
            vector={DENSE_VECTOR: dense, SPARSE_VECTOR: sparse},
            payload={
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "title": chunk.title,
                "topic": chunk.topic,
                "text": chunk.text,
            },
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=True)
    timings["qdrant_upsert"] = perf_counter() - t
    count = client.count(config.QDRANT_COLLECTION, exact=True).count
    print(f"[Index] upserted {count} points into Qdrant collection '{config.QDRANT_COLLECTION}'")

    return {"documents": len(paths), "chunks": [c.chunk_id for c in chunks], "dim": dim, "timings": timings}
