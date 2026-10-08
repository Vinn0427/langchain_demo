"""
Qdrant 连接与数据模型。Indexing 和 Query 两条 Pipeline 共用这里的定义。

一个 chunk = 一个 Point：
    id       : uuid5(chunk_id)，同一个 chunk 每次重建索引得到相同 id
    vector   : {"dense": [1024 floats]}          Named dense vector，Cosine
    sparse   : {"bm25":  {indices, values}}      Named sparse vector，modifier=IDF（Qdrant 端计算 IDF）
    payload  : document_id / chunk_id / source / title / topic / text
"""
import uuid
from functools import lru_cache

from langchain_core.documents import Document
from qdrant_client import QdrantClient

import config

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def point_to_document(point) -> Document:
    payload = dict(point.payload)
    text = payload.pop("text")
    return Document(id=payload["chunk_id"], page_content=text, metadata=payload)
