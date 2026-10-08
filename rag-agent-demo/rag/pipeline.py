"""
Online Query Pipeline（RAG 模块内部的检索流程）：

    query ─┬─ embed_query → dense_search  (Top DENSE_TOP_K) ─┐
           └─ tokenize    → sparse_search (Top SPARSE_TOP_K) ┴→ RRF (Top RRF_TOP_K) → Rerank (Top RERANK_TOP_K)

只读取 Qdrant 中已有的索引，不对文档做任何 Embedding。
LangGraph 只调用 retrieve()，不关心内部有几路召回 —— 编排归 agent，检索 pipeline 归 rag。
"""
from dataclasses import dataclass
from time import perf_counter

from langchain_core.documents import Document

import config
from observability.metrics import timer
from rag.dense_retriever import dense_search, embed_query
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import get_reranker, rerank
from rag.sparse_retriever import sparse_search, tokenize
from rag.store import get_client


@dataclass
class RetrievalResult:
    query: str
    dense: list[Document]     # Dense 召回（语义）
    sparse: list[Document]    # Sparse / BM25 召回（关键词）
    fused: list[Document]     # RRF 融合后的候选
    reranked: list[Document]  # Reranker 精排后的结果，metadata["rerank_score"] ∈ [0, 1]


def retrieve(query: str) -> RetrievalResult:
    with timer("hybrid_retrieval"):
        dense = dense_search(embed_query(query), config.DENSE_TOP_K)
        sparse = sparse_search(query, config.SPARSE_TOP_K)
        with timer("rrf_fusion"):
            fused = reciprocal_rank_fusion({"dense": dense, "sparse": sparse})

    reranked = []
    for result in rerank(query, fused, config.RERANK_TOP_K):
        result.document.metadata["rerank_score"] = result.rerank_score
        reranked.append(result.document)
    return RetrievalResult(query=query, dense=dense, sparse=sparse, fused=fused, reranked=reranked)


def format_documents(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{doc.metadata['chunk_id']}] ({doc.metadata['source']})\n{doc.page_content}" for doc in docs)


def warmup() -> dict:
    """在线服务启动时调用：检查索引是否存在、预加载 Reranker 模型和 jieba 词典（不做任何文档 Embedding）。"""
    timings = {}

    t = perf_counter()
    client = get_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        raise SystemExit(
            f"Qdrant collection '{config.QDRANT_COLLECTION}' 不存在。请先运行：\n"
            "    docker compose up -d\n    python scripts/build_index.py"
        )
    points = client.count(config.QDRANT_COLLECTION, exact=True).count
    timings["qdrant_check"] = perf_counter() - t

    t = perf_counter()
    get_reranker()
    timings["reranker_load"] = perf_counter() - t

    t = perf_counter()
    tokenize("预热")
    timings["jieba_load"] = perf_counter() - t
    return {"points": points, "timings": timings}
