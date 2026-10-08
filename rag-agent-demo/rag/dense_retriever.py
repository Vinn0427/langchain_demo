"""
Dense Retrieval（语义召回）：query → Embedding → Qdrant dense 向量检索 → Top K

Embedding 模型与 Version 2 相同（OpenAI-compatible Embedding API），
离线 Indexing 用 embed_texts() 批量向量化 chunk，在线 Query 只用 embed_query() 向量化问题本身。
"""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config
from observability.metrics import timer
from rag.store import DENSE_VECTOR, get_client, point_to_document


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL,
        chunk_size=config.EMBEDDING_BATCH_SIZE,
        check_embedding_ctx_length=False,  # 直接发送原始文本，兼容非 OpenAI 的兼容服务商
        timeout=config.API_TIMEOUT,        # 不设置时为 openai SDK 默认的 600 秒
        max_retries=config.API_MAX_RETRIES,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """离线 Indexing 使用：批量向量化 chunk。"""
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    with timer("query_embedding"):
        return get_embeddings().embed_query(query)


def dense_search(query_vector: list[float], top_k: int = config.DENSE_TOP_K) -> list[Document]:
    with timer("dense_retrieval"):
        response = get_client().query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            using=DENSE_VECTOR,
            limit=top_k,
            with_payload=True,
        )
    documents = []
    for rank, point in enumerate(response.points, start=1):
        doc = point_to_document(point)
        doc.metadata.update(dense_score=point.score, dense_rank=rank)
        documents.append(doc)
    return documents
