"""
Online Query Pipeline（RAG 模块内部的检索流程）：

    query ─┬─ embed_query → dense_search  (Top DENSE_TOP_K) ─┐
           └─ tokenize    → sparse_search (Top SPARSE_TOP_K) ┴→ RRF (Top RRF_TOP_K) → Rerank (Top RERANK_TOP_K)

只读取 Qdrant 中已有的索引，不对文档做任何 Embedding。
LangGraph 只调用 retrieve()，不关心内部有几路召回 —— 编排归 agent，检索 pipeline 归 rag。

V4：
- retrieve(query, collection=None)：collection 默认是 serving alias；Chunk Strategy Eval 传入测试 collection
- dedup_documents()：Context Dedup（chunk_id 去重 + 规范化正文 hash 去重）
- format_documents()：交给 LLM 的是 raw_text，标题路径只作为一行元数据，不混进正文
"""
import hashlib
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from langchain_core.documents import Document

import config
from observability.metrics import timer
from rag.dense_retriever import dense_search, embed_query
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import get_reranker, rerank
from rag.sparse_retriever import sparse_search, tokenize
from rag.store import get_client, resolve_alias


@dataclass
class RetrievalResult:
    query: str
    dense: list[Document]     # Dense 召回（语义）
    sparse: list[Document]    # Sparse / BM25 召回（关键词）
    fused: list[Document]     # RRF 融合后的候选
    reranked: list[Document]  # Reranker 精排后的结果，metadata["rerank_score"] ∈ [0, 1]


def retrieve(query: str, collection: Optional[str] = None) -> RetrievalResult:
    with timer("hybrid_retrieval"):
        dense = dense_search(embed_query(query), config.DENSE_TOP_K, collection)
        sparse = sparse_search(query, config.SPARSE_TOP_K, collection)
        with timer("rrf_fusion"):
            fused = reciprocal_rank_fusion({"dense": dense, "sparse": sparse})

    reranked = []
    for result in rerank(query, fused, config.RERANK_TOP_K):
        result.document.metadata["rerank_score"] = result.rerank_score
        reranked.append(result.document)
    return RetrievalResult(query=query, dense=dense, sparse=sparse, fused=fused, reranked=reranked)


_WS = re.compile(r"\s+")


def text_fingerprint(text: str) -> str:
    return hashlib.md5(_WS.sub("", text).lower().encode("utf-8")).hexdigest()


def dedup_documents(docs: list[Document], seen_chunk_ids: Optional[set] = None) -> tuple[list[Document], list[str]]:
    """
    Context Dedup：同一 chunk_id 只保留一次；正文规范化（去空白、小写）后 hash 相同的也只保留分数最高的那个。
    seen_chunk_ids：本次请求中之前的 ToolMessage 已经给过 LLM 的 chunk，不再重复放进 Prompt。
    → (保留的文档, 被去掉的 chunk_id)
    """
    seen_ids, seen_hashes, kept, removed = set(seen_chunk_ids or ()), set(), [], []
    for doc in docs:
        chunk_id = doc.metadata["chunk_id"]
        h = text_fingerprint(doc.page_content)
        if chunk_id in seen_ids or h in seen_hashes:
            removed.append(chunk_id)
            continue
        seen_ids.add(chunk_id)
        seen_hashes.add(h)
        kept.append(doc)
    return kept, removed


def section_label(meta: dict) -> str:
    path = meta.get("section_path") or ([meta["topic"]] if meta.get("topic") else [])
    title = meta.get("document_title") or meta.get("title") or meta.get("document_id")
    return " > ".join([title] + list(path))


def format_documents(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata['chunk_id']}] ({doc.metadata['source']} | {section_label(doc.metadata)})\n{doc.page_content}"
        for doc in docs
    )


def warmup() -> dict:
    """在线服务启动时调用：检查索引是否存在、预加载 Reranker 模型和 jieba 词典（不做任何文档 Embedding）。"""
    timings = {}

    t = perf_counter()
    client = get_client()
    collection = resolve_alias(config.QDRANT_ALIAS)
    if collection is None:
        raise SystemExit(
            f"Qdrant alias '{config.QDRANT_ALIAS}' 不存在。请先运行：\n"
            "    docker compose up -d\n    python scripts/build_index.py"
        )
    points = client.count(config.QDRANT_ALIAS, exact=True).count
    timings["qdrant_check"] = perf_counter() - t

    t = perf_counter()
    get_reranker()
    timings["reranker_load"] = perf_counter() - t

    t = perf_counter()
    tokenize("预热")
    timings["jieba_load"] = perf_counter() - t
    return {"points": points, "collection": collection, "timings": timings}
