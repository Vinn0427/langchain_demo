"""
Qdrant 连接与数据模型。Indexing 和 Query 两条 Pipeline 共用这里的定义。

一个 chunk = 一个 Point：
    id       : uuid5(chunk_id)，同一个 chunk 每次重建索引得到相同 id
    vector   : {"dense": [1024 floats]}          Named dense vector，Cosine
    sparse   : {"bm25":  {indices, values}}      Named sparse vector，modifier=IDF（Qdrant 端计算 IDF）
    payload  : V4 —— raw_text / retrieval_text 分开存：
               document_id / chunk_id / chunk_index / source / document_title / section_title / section_path /
               raw_text（交给 LLM 的 Context）/ retrieval_text（Embedding、BM25、Reranker 用）/ content_hash /
               chunk_strategy

V4 Blue-Green：在线服务只访问 alias（config.QDRANT_ALIAS），alias 指向某个物理 collection
（rag_demo_v4_<strategy>_<fingerprint前8位>）。切换 = 一次 update_collection_aliases 请求（Qdrant 原子执行）。
"""
import uuid
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

import config

# Qdrant 计算 BM25 是用"词表 + 稀疏向量"；如果用 ES，就是建倒排索引
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def serving_collection(name: Optional[str] = None) -> str:
    """在线查询的目标：显式传入的 collection（Eval 用），否则是 alias。"""
    return name or config.QDRANT_ALIAS


def resolve_alias(alias: str = config.QDRANT_ALIAS) -> Optional[str]:
    for item in get_client().get_aliases().aliases:
        if item.alias_name == alias:
            return item.collection_name
    return None


def switch_alias(collection: str, alias: str = config.QDRANT_ALIAS) -> Optional[str]:
    """原子切换：delete + create 放在同一个 update_collection_aliases 请求里。返回旧 collection。"""
    old = resolve_alias(alias)
    ops = []
    if old is not None:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)))
    get_client().update_collection_aliases(change_aliases_operations=ops)
    return old


def create_collection(name: str, dim: int, vacuum_on_delete: bool = True) -> None:
    client = get_client()
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        # 见 rag/sparse_retriever.py：Qdrant 的 IDF 统计包含"已删除 / 被覆盖但尚未 vacuum"的旧版本 point。
        # 小 collection 默认永远达不到 vacuum 条件，所以这里让 optimizer 在有删除时就清理。
        # vacuum_on_delete=False 只用于 scripts/verify_incremental.py 的对照组。
        optimizers_config=(
            models.OptimizersConfigDiff(deleted_threshold=0.0001, vacuum_min_vector_number=1)
            if vacuum_on_delete else None
        ),
    )
    for field in ("document_id", "chunk_id"):
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def document_filter(document_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])


def point_to_document(point) -> Document:
    payload = dict(point.payload)
    text = payload.pop("raw_text", None)
    if text is None:  # V3 payload 兼容
        text = payload.pop("text", "")
        payload.setdefault("retrieval_text", text)
    return Document(id=payload["chunk_id"], page_content=text, metadata=payload)
