"""
System Tool：get_index_status —— 返回当前 serving 索引的版本与规模信息。

数据来源：Qdrant（alias → collection、point 数、collection 状态）+ Index Manifest（策略、fingerprint、更新时间）。
只返回白名单字段；不读取 .env，不返回 API Key / Base URL 等任何敏感配置。
"""
import config
from rag.manifest import Manifest
from rag.store import get_client, resolve_alias
from tools.schemas import GetIndexStatusArgs, ToolResult


def get_index_status(_: GetIndexStatusArgs) -> ToolResult:
    collection = resolve_alias(config.QDRANT_ALIAS)
    if collection is None:
        return ToolResult.error("NOT_FOUND", f"alias {config.QDRANT_ALIAS} 当前没有指向任何 collection（索引尚未构建）")
    client = get_client()
    info = client.get_collection(collection)
    manifest = Manifest()
    meta = manifest.collection(collection) or {}
    docs = manifest.documents(collection)
    detail = meta.get("fingerprint_detail") or {}
    last_indexed = max((d["indexed_at"] for d in docs.values()), default=None)
    data = {
        "alias": config.QDRANT_ALIAS,
        "collection": collection,
        "index_version": f"{collection}@rev{meta.get('revision', 0)}",
        "pipeline_fingerprint": meta.get("pipeline_fingerprint"),
        "collection_status": str(info.status.value if hasattr(info.status, "value") else info.status),
        "document_count": len(docs),
        "chunk_count": client.count(collection, exact=True).count,
        "embedding_model": detail.get("embedding_model"),
        "embedding_dimension": detail.get("embedding_dimension"),
        "chunk_strategy": meta.get("strategy"),
        "chunk_strategy_version": detail.get("chunk_strategy_version"),
        "retrieval_text_version": detail.get("retrieval_text_version"),
        "sparse": detail.get("bm25", {}).get("version"),
        "bm25_avgdl_frozen": round(meta["bm25_avgdl"], 3) if meta.get("bm25_avgdl") else None,
        "reranker_model": config.RERANKER_MODEL,
        "last_indexed_at": last_indexed,
        "collection_created_at": meta.get("created_at"),
    }
    return ToolResult(success=True, data=data)
