"""
Exact Lookup Tools：get_document / get_chunk / list_documents。

全部是对 Qdrant payload 的精确查询（按 document_id 过滤 / 按 uuid5(chunk_id) 取 point / 扫描元数据），
不做 Embedding、不做向量检索。执行器只返回 ToolResult，异常交给 tools/registry.py 统一转换。
"""
from typing import Optional

import config
from rag.pipeline import section_label
from rag.store import document_filter, get_client, point_id
from tools.schemas import GetChunkArgs, GetDocumentArgs, ListDocumentsArgs, ToolResult

_META_FIELDS = ["document_id", "document_title", "source", "chunk_id", "chunk_index", "section_title", "section_path"]


def _catalog(collection: str = config.QDRANT_ALIAS) -> dict[str, dict]:
    """document_id → {title, source, chunks: [...]}（只读元数据字段，不读正文）"""
    docs: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = get_client().scroll(collection, limit=256, offset=offset, with_payload=_META_FIELDS)
        for p in points:
            meta = p.payload
            doc = docs.setdefault(meta["document_id"], {
                "document_id": meta["document_id"], "title": meta.get("document_title"), "source": meta.get("source"),
                "chunks": [],
            })
            doc["chunks"].append({"chunk_id": meta["chunk_id"], "chunk_index": meta.get("chunk_index", 0),
                                  "section": " > ".join(meta.get("section_path") or [])})
        if offset is None:
            break
    for doc in docs.values():
        doc["chunks"].sort(key=lambda c: c["chunk_index"])
    return dict(sorted(docs.items()))


def document_catalog() -> list[dict]:
    """给 Query Understanding Prompt 用的轻量目录：[{document_id, title}]"""
    return [{"document_id": d["document_id"], "title": d["title"]} for d in _catalog().values()]


def get_document(args: GetDocumentArgs) -> ToolResult:
    points, _ = get_client().scroll(
        config.QDRANT_ALIAS, scroll_filter=document_filter(args.document_id), limit=500, with_payload=True,
    )
    if not points:
        available = list(_catalog())
        return ToolResult.error(
            "NOT_FOUND", f"知识库中没有 document_id={args.document_id} 的文档。可用的 document_id：{', '.join(available)}",
            data={"available_document_ids": available},
        )
    points.sort(key=lambda p: p.payload.get("chunk_index", 0))
    first = points[0].payload
    sections, used, truncated = [], 0, False
    for p in points:
        meta = p.payload
        item = {"chunk_id": meta["chunk_id"], "section": " > ".join(meta.get("section_path") or [])}
        if args.view == "full":
            text = meta["raw_text"]
            if used + len(text) > config.MAX_DOCUMENT_CHARS:
                truncated = True
                break
            used += len(text)
            item["text"] = text
        sections.append(item)
    chunk_ids = [s["chunk_id"] for s in sections]
    return ToolResult(
        success=True,
        data={"document_id": args.document_id, "title": first.get("document_title"), "source": first.get("source"),
              "view": args.view, "chunk_count": len(points), "sections": sections, "truncated": truncated},
        message=f"文档正文超过 {config.MAX_DOCUMENT_CHARS} 字，已截断" if truncated else None,
        source_ids=[args.document_id], chunk_ids=chunk_ids, document_ids=[args.document_id],
    )


def get_chunk(args: GetChunkArgs) -> ToolResult:
    records = get_client().retrieve(config.QDRANT_ALIAS, ids=[point_id(args.chunk_id)], with_payload=True)
    if not records:
        document_id = args.chunk_id.rsplit("_", 1)[0]
        catalog = _catalog()
        hint = (f"文档 {document_id} 现有的 chunk_id：{', '.join(c['chunk_id'] for c in catalog[document_id]['chunks'])}"
                if document_id in catalog else f"知识库中也没有文档 {document_id}")
        return ToolResult.error("NOT_FOUND", f"知识库中没有 chunk_id={args.chunk_id}。{hint}")
    meta = records[0].payload
    return ToolResult(
        success=True,
        data={"chunk_id": meta["chunk_id"], "document_id": meta["document_id"], "source": meta["source"],
              "section": section_label(meta), "text": meta["raw_text"]},
        source_ids=[meta["chunk_id"]], chunk_ids=[meta["chunk_id"]], document_ids=[meta["document_id"]],
    )


def _matches(doc: dict, topic: Optional[str]) -> bool:
    if not topic:
        return True
    needle = topic.lower().strip()
    haystack = [doc["document_id"], doc["title"] or ""] + [c["section"] for c in doc["chunks"]]
    return any(needle in h.lower() for h in haystack)


def list_documents(args: ListDocumentsArgs) -> ToolResult:
    docs = [
        {"document_id": d["document_id"], "title": d["title"], "source": d["source"], "chunk_count": len(d["chunks"]),
         "sections": sorted({c["section"].split(" > ")[0] for c in d["chunks"] if c["section"]})}
        for d in _catalog().values() if _matches(d, args.topic)
    ]
    return ToolResult(
        success=True, data={"topic": args.topic, "documents": docs},
        message=None if docs else f"没有与 topic={args.topic!r} 匹配的文档",
        document_ids=[d["document_id"] for d in docs],
    )
