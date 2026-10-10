"""
Incremental Indexing 正确性验证（在 data/ 的临时副本 + 独立测试 collection 上运行，不影响在线索引）

    python scripts/verify_incremental.py           # 跑完删除测试 collection
    python scripts/verify_incremental.py --keep    # 保留测试 collection 便于在 Dashboard 查看

场景：
  1. 第一次：全量构建
  2. 文件不变：0 chunk / 0 embedding / 0 写入
  3. 修改一个文件：只重新处理该文件；内容未变的 chunk 复用旧 dense 向量
  4. 删除一个文件：Qdrant 中该 document 的 point 全部删除，manifest 同步
  5. 新增一个文件
  6. 等价性：增量得到的 collection 与"同一冻结 avgdl 的全量重建"逐 point 比较 payload / dense / sparse，
     并逐 query 比较 Qdrant 返回的 BM25 分数（验证 IDF 统计没有残留已删除 / 被覆盖的旧版本）
  7. pipeline_fingerprint 变化：file_hash 不变也不能 skip
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag import indexing  # noqa: E402
from rag.indexing import full_build, incremental_update, pipeline_fingerprint  # noqa: E402
from rag.manifest import Manifest  # noqa: E402
from rag.sparse_retriever import sparse_search  # noqa: E402
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, document_filter, get_client  # noqa: E402

STRATEGY = config.CHUNK_STRATEGY
INCR, FULL = "rag_verify_incr", "rag_verify_full"
WORK = config.PROJECT_ROOT / "index_state" / "verify_data"
CHECKS: list[dict] = []


def check(name: str, ok: bool, detail) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def all_points(collection: str) -> dict:
    points, offset = {}, None
    while True:
        batch, offset = get_client().scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=True)
        for p in batch:
            points[p.payload["chunk_id"]] = p
        if offset is None:
            return points


CTRL = "rag_verify_ctrl"


def _ensure_ctrl(points: dict) -> None:
    from qdrant_client import models

    from rag.store import create_collection

    import uuid

    create_collection(CTRL, config.EMBEDDING_DIMENSION, vacuum_on_delete=False)
    structs = [models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points.values()]
    get_client().upsert(CTRL, structs, wait=True)
    # 模拟"文件修改导致 chunk 重新编号 / 文件删除"：先写入一批临时 point，再把它们全部删除。
    # 删除之后，CTRL 的逻辑内容与 FULL 完全相同。
    ghosts = [models.PointStruct(id=str(uuid.uuid4()), vector=p.vector, payload=p.payload) for p in points.values()]
    get_client().upsert(CTRL, ghosts, wait=True)
    get_client().delete(CTRL, points_selector=models.PointIdsList(points=[g.id for g in ghosts]), wait=True)
    indexing._wait_green(CTRL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    manifest = Manifest()
    client = get_client()
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.copytree(config.DATA_DIR, WORK)
    report = {"strategy": STRATEGY, "fingerprint": pipeline_fingerprint(STRATEGY)[0], "steps": {}}

    print("\n== Step 1: 第一次全量构建")
    s1 = full_build(INCR, STRATEGY, data_dir=WORK, manifest=manifest)
    report["steps"]["1_full"] = s1.as_dict()
    check("全量构建 embedding chunk 数 = chunk 总数", s1.chunks_embedded == s1.points_in_collection,
          f"embedded={s1.chunks_embedded} points={s1.points_in_collection}")

    print("\n== Step 2: 文件不变，再跑一次")
    s2 = incremental_update(INCR, STRATEGY, data_dir=WORK, manifest=manifest)
    report["steps"]["2_unchanged"] = s2.as_dict()
    check("无修改：0 chunking / 0 embedding / 0 写入",
          s2.chunks_chunked == 0 and s2.chunks_embedded == 0 and s2.embedding_requests == 0
          and s2.points_upserted == 0 and s2.points_deleted == 0,
          f"chunked={s2.chunks_chunked} embedded={s2.chunks_embedded} embedding_requests={s2.embedding_requests} "
          f"upserted={s2.points_upserted} deleted={s2.points_deleted} skipped={len(s2.unchanged)}")

    print("\n== Step 3: 修改 redis_ops.md（改 1 个 chunk 的一句话 + 新增 1 个小节）")
    path = WORK / "redis_ops.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace("单次迁移不超过 1000 个 slot", "单次迁移不超过 500 个 slot")
    text = text.replace("## 高可用", "### 缩容\n\n缩容前必须先把待下线分片的 slot 全部迁出，并保留节点 7 天再回收。\n\n## 高可用")
    path.write_text(text, encoding="utf-8")
    before_total = client.count(INCR, exact=True).count
    before_vectors = {c: p.vector[DENSE_VECTOR] for c, p in all_points(INCR).items() if c.startswith("redis_ops_")}
    before_hash = {c: p.payload["content_hash"] for c, p in all_points(INCR).items() if c.startswith("redis_ops_")}
    s3 = incremental_update(INCR, STRATEGY, data_dir=WORK, manifest=manifest)
    report["steps"]["3_modified"] = s3.as_dict()
    redis_ops_chunks = len(manifest.documents(INCR)["redis_ops"]["chunk_ids"])
    check("只处理被修改的文件", s3.modified == ["redis_ops"] and not s3.added and not s3.deleted
          and len(s3.unchanged) == s3.documents_scanned - 1, f"modified={s3.modified} unchanged={len(s3.unchanged)}")
    check("只重新 chunk 该文件", s3.chunks_chunked == redis_ops_chunks, f"chunked={s3.chunks_chunked} (redis_ops 共 {redis_ops_chunks})")
    check("只 embedding 内容变化的 chunk", s3.chunks_embedded < s3.chunks_chunked,
          f"embedded={s3.chunks_embedded} reused={s3.chunks_vector_reused}")
    p = client.scroll(INCR, scroll_filter=document_filter("redis_ops"), limit=50, with_payload=True)[0]
    new_text = [x.payload["raw_text"] for x in p if "500 个 slot" in x.payload["raw_text"]]
    check("Qdrant 中是新内容", bool(new_text) and not any("1000 个 slot" in x.payload["raw_text"] for x in p),
          f"points(redis_ops)={len(p)}")
    check("point 总数变化 = 新增小节数", client.count(INCR, exact=True).count == before_total + 1,
          f"{before_total} → {client.count(INCR, exact=True).count}")
    after = {c: p for c, p in all_points(INCR).items() if c.startswith("redis_ops_")}
    old_by_hash = {before_hash[c]: before_vectors[c] for c in before_vectors}
    reused = [c for c, p in after.items() if p.payload["content_hash"] in old_by_hash]
    check("复用的 dense 向量与修改前逐位相同", reused and all(
        after[c].vector[DENSE_VECTOR] == old_by_hash[after[c].payload["content_hash"]] for c in reused),
        f"{len(reused)} reused vectors identical")

    print("\n== Step 4: 删除 api.md")
    (WORK / "api.md").unlink()
    api_before = client.count(INCR, count_filter=document_filter("api"), exact=True).count
    s4 = incremental_update(INCR, STRATEGY, data_dir=WORK, manifest=manifest)
    report["steps"]["4_deleted"] = s4.as_dict()
    api_after = client.count(INCR, count_filter=document_filter("api"), exact=True).count
    check("Qdrant 中 api 的 point 全部删除", s4.deleted == ["api"] and api_before > 0 and api_after == 0
          and s4.points_deleted == api_before, f"api points {api_before} → {api_after}")
    check("manifest 同步删除", "api" not in manifest.documents(INCR), "api row removed")
    check("其他文件不处理", s4.chunks_chunked == 0 and s4.chunks_embedded == 0, f"chunked={s4.chunks_chunked}")

    print("\n== Step 5: 新增 faq.md")
    (WORK / "faq.md").write_text("# 常见问题\n\n## 工位网络\n\n工位有线网络故障请在 IT 服务台提交工单，紧急情况拨打分机 8000。\n",
                                 encoding="utf-8")
    s5 = incremental_update(INCR, STRATEGY, data_dir=WORK, manifest=manifest)
    report["steps"]["5_added"] = s5.as_dict()
    check("新增文件：chunk → embedding → upsert → manifest", s5.added == ["faq"] and s5.chunks_embedded == 1
          and "faq" in manifest.documents(INCR), f"added={s5.added} embedded={s5.chunks_embedded}")

    print("\n== Step 6: 增量结果 vs 同一冻结 avgdl 的全量重建")
    frozen = manifest.collection(INCR)["bm25_avgdl"]
    full_build(FULL, STRATEGY, data_dir=WORK, manifest=manifest, log=False, avgdl_override=frozen)
    a, b = all_points(INCR), all_points(FULL)
    check("point 集合一致", set(a) == set(b), f"incr={len(a)} full={len(b)}")
    payload_diff = [c for c in a if c in b and
                    {k: v for k, v in a[c].payload.items()} != {k: v for k, v in b[c].payload.items()}]
    sparse_diff = [c for c in a if c in b and (
        a[c].vector[SPARSE_VECTOR].indices != b[c].vector[SPARSE_VECTOR].indices
        or any(abs(x - y) > 1e-6 for x, y in zip(a[c].vector[SPARSE_VECTOR].values, b[c].vector[SPARSE_VECTOR].values)))]

    def cos(u, v):
        return sum(x * y for x, y in zip(u, v)) / math.sqrt(sum(x * x for x in u) * sum(y * y for y in v))

    dense_min_cos = min(cos(a[c].vector[DENSE_VECTOR], b[c].vector[DENSE_VECTOR]) for c in a if c in b)
    check("payload 逐 point 一致", not payload_diff, f"diff={payload_diff}")
    check("BM25 sparse vector 逐 point 一致", not sparse_diff, f"diff={sparse_diff}")
    # Embedding Provider 本身不是逐位确定的：实测同一个 batch 重复请求余弦约 0.998（单条请求为 1.0），
    # 所以"全量重新 embedding"与增量结果只能在 Provider 噪声范围内一致。
    check("dense 向量与全量重建一致（Provider 噪声范围内，cos>0.995）", dense_min_cos > 0.995,
          f"min cosine={dense_min_cos:.6f}")
    queries = [d["query"] for d in json.loads(config.EVAL_GATE_DATASET.read_text(encoding="utf-8"))]
    score_diff = []
    for q in queries:
        ra = {d.metadata["chunk_id"]: round(d.metadata["sparse_score"], 4) for d in sparse_search(q, 32, INCR)}
        rb = {d.metadata["chunk_id"]: round(d.metadata["sparse_score"], 4) for d in sparse_search(q, 32, FULL)}
        if ra != rb:  # 取全部命中（limit ≥ point 数）后按 chunk_id → score 比较：同分并列的先后顺序不影响结论
            score_diff.append({"query": q, "only_incr": {k: v for k, v in ra.items() if rb.get(k) != v},
                               "only_full": {k: v for k, v in rb.items() if ra.get(k) != v}})
    check("Qdrant BM25 检索分数逐 query 一致（IDF 无残留旧版本）", not score_diff,
          f"{len(queries) - len(score_diff)}/{len(queries)} queries identical")
    report["score_diff"] = score_diff

    # 对照组：Qdrant 默认 optimizer。写入 FULL 的 point，再写入并删除一批临时 point（逻辑内容与 FULL 相同）
    _ensure_ctrl(b)
    ctrl_diff = 0
    for q in queries:
        ra = {d.metadata["chunk_id"]: round(d.metadata["sparse_score"], 4) for d in sparse_search(q, 32, CTRL)}
        rb = {d.metadata["chunk_id"]: round(d.metadata["sparse_score"], 4) for d in sparse_search(q, 32, FULL)}
        ctrl_diff += ra != rb
    report["control_default_optimizer_score_diff_queries"] = ctrl_diff
    print(f"  [INFO] 对照组（Qdrant 默认 optimizer，写入并删除临时 point 后）：{ctrl_diff}/{len(queries)} queries 的 "
          "BM25 分数与全量重建不同 —— 已删除的 point 仍计入 IDF 统计")

    print("\n== Step 7: pipeline_fingerprint 变化时不能 skip")
    original = config.RETRIEVAL_TEXT_VERSION
    config.RETRIEVAL_TEXT_VERSION = "rt-v1-test-change"
    try:
        new_fp = pipeline_fingerprint(STRATEGY)[0]
        check("fingerprint 变化 → 目标 collection 名变化（build() 会走 Blue-Green）",
              indexing.collection_name(STRATEGY, new_fp) != indexing.collection_name(STRATEGY, report["fingerprint"]),
              f"{report['fingerprint']} → {new_fp}")
        s7 = incremental_update(INCR, STRATEGY, data_dir=WORK, manifest=manifest, log=False)
        check("file_hash 不变但 fingerprint 变化：全部文件重新处理", not s7.unchanged
              and len(s7.modified) == s7.documents_scanned, f"modified={len(s7.modified)} unchanged={len(s7.unchanged)}")
        report["steps"]["7_fingerprint_changed"] = s7.as_dict()
    finally:
        config.RETRIEVAL_TEXT_VERSION = original

    report["checks"] = CHECKS
    report["passed"] = all(c["ok"] for c in CHECKS)
    out = config.PROJECT_ROOT / "eval" / "reports" / "incremental_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[Verify] {sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} checks passed → {out}")

    if not args.keep:
        for name in (INCR, FULL, CTRL):
            client.delete_collection(name)
            manifest.drop_collection(name)
        shutil.rmtree(WORK, ignore_errors=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
