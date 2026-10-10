"""
Offline Indexing Pipeline（V4：文件级 Incremental Indexing + Pipeline Fingerprint + Blue-Green）

                     扫描 data/*.md → file_hash
                                │
             manifest 中 (file_hash, pipeline_fingerprint) 是否都没变？
                 ┌──────────────┴──────────────┐
                是                             否（新增 / 修改）
                 ↓                              ↓
              SKIP                 chunk → retrieval_text → dense（只对 content_hash 变化的 chunk）
     （不 chunk / 不 embedding /           + BM25（冻结 avgdl）→ upsert → 删除该文档多余的旧 point
        不写 Qdrant）                       → manifest
     manifest 里有、data/ 里没有的文档 → 按 document_id 删除 Qdrant point → manifest

Pipeline Fingerprint 变化（Embedding 模型 / 维度 / Chunk 策略 / retrieval_text / Sparse 表示 …）
意味着整个索引的语义变了 —— 不在当前 collection 上覆盖，而是 Blue-Green：
    新建 collection（名字带 fingerprint）→ 全量构建 → Retrieval Eval Gate → alias 原子切换 → 旧 collection 标记 retired

只由 scripts/build_index.py、eval/、scripts/verify_incremental.py 调用；在线 Query Pipeline 不 import 本模块。
"""
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Optional

from qdrant_client import models

import config
from rag.chunking import Chunk, chunk_document, file_hash, splitter_config
from rag.dense_retriever import embed_texts
from rag.manifest import Manifest
from rag.sparse_retriever import bm25_document_vector, tokenize, tokenizer_signature
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, create_collection, document_filter, get_client, point_id, resolve_alias, switch_alias


# ---------------------------------------------------------------------
# Pipeline Fingerprint
# ---------------------------------------------------------------------
def fingerprint_detail(strategy: str) -> dict:
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "chunk_strategy_version": config.CHUNK_STRATEGY_VERSION,
        "chunk_enrichment_mode": strategy,
        "splitter_config": splitter_config(),
        "retrieval_text_version": config.RETRIEVAL_TEXT_VERSION,
        "bm25": {"version": config.BM25_VERSION, "k1": config.BM25_K1, "b": config.BM25_B, **tokenizer_signature()},
        "sparse_representation": "bm25_tf(k1,b,frozen_avgdl)+qdrant_idf_modifier",
    }


def pipeline_fingerprint(strategy: str) -> tuple[str, dict]:
    detail = fingerprint_detail(strategy)
    digest = hashlib.sha256(json.dumps(detail, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return digest, detail


def collection_name(strategy: str, fingerprint: str, prefix: str = config.COLLECTION_PREFIX) -> str:
    return f"{prefix}_{strategy.lower()}_{fingerprint[:8]}"


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------
@dataclass
class IndexStats:
    collection: str
    mode: str                      # full / incremental
    documents_scanned: int = 0
    added: list = field(default_factory=list)
    modified: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)
    deleted: list = field(default_factory=list)
    chunks_chunked: int = 0        # 实际执行了 chunk 的 chunk 数（未变化文件为 0）
    chunks_embedded: int = 0       # 实际送去 Embedding 的 chunk 数
    chunks_vector_reused: int = 0  # 修改的文档中 content_hash 没变、直接复用旧 dense 向量的 chunk 数
    embedding_requests: int = 0    # Embedding API 请求批次数
    points_upserted: int = 0
    points_deleted: int = 0
    bm25_avgdl_frozen: Optional[float] = None
    bm25_avgdl_actual: Optional[float] = None
    bm25_avgdl_drift: Optional[float] = None
    points_in_collection: Optional[int] = None
    timings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def scan_documents(data_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(Path(data_dir).glob("*.md"))}


def _wait_green(collection: str, timeout_s: float = 15.0) -> None:
    """删除 / 覆盖写之后等 optimizer vacuum 完成，IDF 统计才会排除旧版本 point（见 sparse_retriever.py）。"""
    client = get_client()
    deadline = time.time() + timeout_s
    time.sleep(0.2)
    while time.time() < deadline:
        if client.get_collection(collection).status == models.CollectionStatus.GREEN:
            return
        time.sleep(0.2)


def _old_dense_vectors(collection: str, chunk_ids: list[str]) -> dict[str, list[float]]:
    """content_hash → dense 向量：修改过的文档里，内容没变的 chunk 不需要重新 Embedding。"""
    if not chunk_ids:
        return {}
    records = get_client().retrieve(
        collection, ids=[point_id(c) for c in chunk_ids], with_payload=["content_hash"], with_vectors=[DENSE_VECTOR],
    )
    out = {}
    for r in records:
        h = (r.payload or {}).get("content_hash")
        vec = r.vector.get(DENSE_VECTOR) if isinstance(r.vector, dict) else None
        if h and vec:
            out[h] = vec
    return out


def _index_document(collection: str, strategy: str, fingerprint: str, avgdl: float, path: Path,
                    chunks: list[Chunk], old_chunk_ids: list[str], manifest: Manifest, stats: IndexStats) -> None:
    client = get_client()
    reuse = _old_dense_vectors(collection, old_chunk_ids)
    to_embed = [c for c in chunks if c.content_hash not in reuse]

    t = perf_counter()
    vectors = embed_texts([c.retrieval_text for c in to_embed]) if to_embed else []
    stats.timings["dense_embedding"] = stats.timings.get("dense_embedding", 0.0) + perf_counter() - t
    stats.embedding_requests += -(-len(to_embed) // config.EMBEDDING_BATCH_SIZE)
    stats.chunks_embedded += len(to_embed)
    stats.chunks_vector_reused += len(chunks) - len(to_embed)
    for vec in vectors:
        if len(vec) != config.EMBEDDING_DIMENSION:
            raise SystemExit(f"Embedding 维度 {len(vec)} 与 EMBEDDING_DIMENSION={config.EMBEDDING_DIMENSION} 不一致")
    dense = dict(reuse)
    dense.update({c.content_hash: v for c, v in zip(to_embed, vectors)})

    t = perf_counter()
    token_lists = [tokenize(c.retrieval_text) for c in chunks]
    points = [
        models.PointStruct(
            id=point_id(c.chunk_id),
            vector={DENSE_VECTOR: dense[c.content_hash], SPARSE_VECTOR: bm25_document_vector(tokens, avgdl)},
            payload={**c.payload(strategy), "pipeline_fingerprint": fingerprint},
        )
        for c, tokens in zip(chunks, token_lists)
    ]
    stats.timings["bm25_encoding"] = stats.timings.get("bm25_encoding", 0.0) + perf_counter() - t

    t = perf_counter()
    # 先 upsert 新版本，再删除多余的旧 point：任何时刻该文档都至少有一个完整版本可查
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)
    stale = sorted(set(old_chunk_ids) - {c.chunk_id for c in chunks})
    if stale:
        client.delete(collection, points_selector=models.PointIdsList(points=[point_id(c) for c in stale]), wait=True)
    stats.timings["qdrant_write"] = stats.timings.get("qdrant_write", 0.0) + perf_counter() - t
    stats.points_upserted += len(points)
    stats.points_deleted += len(stale)

    manifest.upsert_document(collection, path.stem, chunks[0].source if chunks else str(path), file_hash(path),
                             fingerprint, [c.chunk_id for c in chunks], sum(len(t) for t in token_lists))


def _delete_document(collection: str, document_id: str, manifest: Manifest, stats: IndexStats) -> None:
    client = get_client()
    before = client.count(collection, count_filter=document_filter(document_id), exact=True).count
    client.delete(collection, points_selector=models.FilterSelector(filter=document_filter(document_id)), wait=True)
    manifest.delete_document(collection, document_id)
    stats.points_deleted += before


def _avgdl_report(collection: str, manifest: Manifest, stats: IndexStats) -> None:
    docs = manifest.documents(collection)
    chunks = sum(len(d["chunk_ids"]) for d in docs.values())
    tokens = sum(d["token_count"] for d in docs.values())
    frozen = manifest.collection(collection)["bm25_avgdl"]
    stats.bm25_avgdl_frozen = round(frozen, 3)
    if chunks:
        actual = tokens / chunks
        stats.bm25_avgdl_actual = round(actual, 3)
        stats.bm25_avgdl_drift = round(abs(actual - frozen) / frozen, 4)
        if stats.bm25_avgdl_drift > config.BM25_AVGDL_DRIFT_THRESHOLD:
            print(f"[Index] WARNING: BM25 avgdl drift {stats.bm25_avgdl_drift:.1%} > "
                  f"{config.BM25_AVGDL_DRIFT_THRESHOLD:.0%}，建议 --rebuild（Blue-Green 全量重建 Sparse Index）")


# ---------------------------------------------------------------------
# Full build（新 collection）与 Incremental update（已有 collection）
# ---------------------------------------------------------------------
def full_build(collection: str, strategy: str, data_dir: Path = config.DATA_DIR,
               manifest: Optional[Manifest] = None, log: bool = True,
               avgdl_override: Optional[float] = None) -> IndexStats:
    """avgdl_override：只用于正确性验证 —— 用"增量 collection 的冻结 avgdl"做一次全量重建来逐位对比。"""
    manifest = manifest or Manifest()
    fingerprint, detail = pipeline_fingerprint(strategy)
    stats = IndexStats(collection=collection, mode="full")
    t0 = perf_counter()
    files = scan_documents(data_dir)
    stats.documents_scanned = len(files)

    t = perf_counter()
    chunked = {doc_id: chunk_document(path, strategy) for doc_id, path in files.items()}
    all_chunks = [c for chunks in chunked.values() for c in chunks]
    stats.chunks_chunked = len(all_chunks)
    lengths = [len(tokenize(c.retrieval_text)) for c in all_chunks]
    avgdl = avgdl_override or sum(lengths) / max(len(lengths), 1)
    stats.timings["load_and_chunk"] = perf_counter() - t

    create_collection(collection, config.EMBEDDING_DIMENSION)
    manifest.create_collection(collection, strategy, fingerprint, detail, avgdl)
    if log:
        print(f"[Index] full build → {collection}: {len(files)} documents → {len(all_chunks)} chunks, "
              f"frozen BM25 avgdl={avgdl:.2f}")
    for doc_id, path in files.items():
        _index_document(collection, strategy, fingerprint, avgdl, path, chunked[doc_id], [], manifest, stats)
        stats.added.append(doc_id)
        if log:
            print(f"[Index] {path.name} added → {len(chunked[doc_id])} chunks")
    _wait_green(collection)
    manifest.set_status(collection, "ready")
    _avgdl_report(collection, manifest, stats)
    stats.points_in_collection = get_client().count(collection, exact=True).count
    stats.timings["total"] = perf_counter() - t0
    return stats


def incremental_update(collection: str, strategy: str, data_dir: Path = config.DATA_DIR,
                       manifest: Optional[Manifest] = None, log: bool = True) -> IndexStats:
    manifest = manifest or Manifest()
    fingerprint, _ = pipeline_fingerprint(strategy)
    meta = manifest.collection(collection)
    if meta is None:
        raise RuntimeError(f"manifest 中没有 collection {collection}，请先 full build")
    avgdl = meta["bm25_avgdl"]
    stats = IndexStats(collection=collection, mode="incremental")
    t0 = perf_counter()
    files = scan_documents(data_dir)
    stats.documents_scanned = len(files)
    indexed = manifest.documents(collection)

    for doc_id, path in files.items():
        record = indexed.get(doc_id)
        h = file_hash(path)
        if record and record["file_hash"] == h and record["pipeline_fingerprint"] == fingerprint:
            stats.unchanged.append(doc_id)
            if log:
                print(f"[Index] {path.name} unchanged → skipped")
            continue
        t = perf_counter()
        chunks = chunk_document(path, strategy)
        stats.timings["load_and_chunk"] = stats.timings.get("load_and_chunk", 0.0) + perf_counter() - t
        stats.chunks_chunked += len(chunks)
        old_ids = record["chunk_ids"] if record else []
        embedded_before, reused_before = stats.chunks_embedded, stats.chunks_vector_reused
        _index_document(collection, strategy, fingerprint, avgdl, path, chunks, old_ids, manifest, stats)
        (stats.modified if record else stats.added).append(doc_id)
        if log:
            action = "modified" if record else "added"
            print(f"[Index] {path.name} {action} → {len(chunks)} chunks re-chunked, "
                  f"{stats.chunks_embedded - embedded_before} embedded, "
                  f"{stats.chunks_vector_reused - reused_before} dense vectors reused (content_hash unchanged)")

    for doc_id in sorted(set(indexed) - set(files)):
        before = stats.points_deleted
        _delete_document(collection, doc_id, manifest, stats)
        stats.deleted.append(doc_id)
        if log:
            print(f"[Index] {doc_id} deleted from data/ → removed {stats.points_deleted - before} points from Qdrant")

    if stats.added or stats.modified or stats.deleted:
        _wait_green(collection)
        manifest.bump_revision(collection)
    _avgdl_report(collection, manifest, stats)
    stats.points_in_collection = get_client().count(collection, exact=True).count
    stats.timings["total"] = perf_counter() - t0
    return stats


# ---------------------------------------------------------------------
# Blue-Green Eval Gate
# ---------------------------------------------------------------------
def retrieval_gate_eval(collection: str, dataset_path: Path = config.EVAL_GATE_DATASET) -> dict:
    """在指定 collection 上跑 hybrid_rerank 的 Recall@5 / MRR（只用有标注 relevant_chunk_ids 的样本）。"""
    from rag.pipeline import retrieve

    raw = dataset_path.read_bytes()
    dataset = [d for d in json.loads(raw) if d.get("relevant_chunk_ids")]
    recalls, rrs = [], []
    for item in dataset:
        relevant = set(item["relevant_chunk_ids"])
        ids = [d.metadata["chunk_id"] for d in retrieve(item["query"], collection=collection).reranked]
        recalls.append(len(set(ids[:5]) & relevant) / len(relevant))
        rrs.append(next((1.0 / r for r, c in enumerate(ids, 1) if c in relevant), 0.0))
    return {
        "dataset_sha": hashlib.sha256(raw).hexdigest()[:12],
        "n": len(dataset),
        "hybrid_rerank_recall@5": round(sum(recalls) / max(len(recalls), 1), 4),
        "hybrid_rerank_mrr": round(sum(rrs) / max(len(rrs), 1), 4),
    }


def build(strategy: Optional[str] = None, force_rebuild: bool = False, switch: bool = True,
          eval_gate: bool = True, data_dir: Path = config.DATA_DIR, alias: str = config.QDRANT_ALIAS) -> dict:
    strategy = strategy or config.CHUNK_STRATEGY
    manifest = Manifest()
    fingerprint, detail = pipeline_fingerprint(strategy)
    target = collection_name(strategy, fingerprint)
    current = resolve_alias(alias)
    if force_rebuild and target == current:
        # 同一 fingerprint 强制重建（例如 avgdl drift 过大）：也不能覆盖正在服务的 collection
        target = f"{target}_r{time.strftime('%m%d%H%M%S')}"
    meta = manifest.collection(target)
    exists = get_client().collection_exists(target)
    print(f"[Index] alias '{alias}' → {current or '(none)'}")
    print(f"[Index] pipeline_fingerprint={fingerprint} → target collection {target}")

    report = {"alias": alias, "previous": current, "target": target, "fingerprint": fingerprint,
              "fingerprint_detail": detail, "switched": False}

    if not force_rebuild and exists and meta and meta["status"] in ("ready", "serving", "retired"):
        stats = incremental_update(target, strategy, data_dir, manifest)
    else:
        if current and current != target:
            print(f"[Index] fingerprint changed → Blue-Green: build {target} while {current} keeps serving")
        stats = full_build(target, strategy, data_dir, manifest)
    report["stats"] = stats.as_dict()

    if current == target:
        manifest.set_status(target, "serving")
        report["switched"] = None   # 本来就在服务：增量更新是 in-place 的
        return report

    if eval_gate:
        green = retrieval_gate_eval(target)
        manifest.set_eval(target, green)
        blue = None
        if current:
            blue_meta = manifest.collection(current)
            blue = (blue_meta or {}).get("eval_summary")
            if not blue or blue.get("dataset_sha") != green["dataset_sha"]:
                blue = retrieval_gate_eval(current)
                if blue_meta:
                    manifest.set_eval(current, blue)
        passed = green["hybrid_rerank_recall@5"] >= config.BLUE_GREEN_MIN_RECALL5 and (
            blue is None or green["hybrid_rerank_mrr"] >= blue["hybrid_rerank_mrr"] - config.BLUE_GREEN_MAX_MRR_DROP
        )
        report.update(eval_green=green, eval_blue=blue, eval_passed=passed)
        print(f"[BlueGreen] green {target}: {green}")
        if blue:
            print(f"[BlueGreen] blue  {current}: {blue}")
        if not passed:
            print("[BlueGreen] eval gate FAILED → alias 不切换，继续由当前 collection 服务")
            return report

    if switch:
        old = switch_alias(target, alias)
        manifest.set_status(target, "serving")
        if old and manifest.collection(old):
            manifest.set_status(old, "retired")
        report["switched"] = True
        print(f"[BlueGreen] alias '{alias}': {old or '(none)'} → {target}（原子切换；旧 collection 保留，可回滚）")
    return report


def build_index() -> dict:
    """V3 兼容入口。"""
    return build()
