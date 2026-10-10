"""
Dense Retrieval（语义召回）：query → Embedding → Qdrant dense 向量检索 → Top K

离线 Indexing 用 llm.client.embed_documents() 批量向量化 chunk 的 retrieval_text，
在线 Query 只用 embed_query() 向量化问题本身。V4：Embedding 调用也走统一的 retry + trace。
"""
from typing import Optional

from langchain_core.documents import Document

import config
from llm import client as llm_client
from observability.metrics import timer
from rag.store import DENSE_VECTOR, get_client, point_to_document, serving_collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """离线 Indexing 使用：按 EMBEDDING_BATCH_SIZE 分批向量化。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        vectors.extend(llm_client.embed_documents(texts[i:i + config.EMBEDDING_BATCH_SIZE]))
    return vectors


def embed_query(query: str) -> list[float]:
    with timer("query_embedding"):
        return llm_client.embed_query(query)


def dense_search(query_vector: list[float], top_k: int = config.DENSE_TOP_K, collection: Optional[str] = None) -> list[Document]:
    with timer("dense_retrieval"):
        response = get_client().query_points(
            collection_name=serving_collection(collection),
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
