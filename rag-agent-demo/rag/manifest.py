"""
Index Manifest（SQLite）：记录每个物理 collection 里"哪个文件、以什么 hash、用什么 pipeline 建的索引"。

documents  ：(collection, document_id) → source_path / file_hash / pipeline_fingerprint / chunk_ids / indexed_at
collections：collection → strategy / pipeline_fingerprint / fingerprint_detail / bm25_avgdl（冻结值）/
             status（building / ready / serving / retired）/ revision（增量更新次数）/ eval_summary / 时间戳

在线 get_index_status Tool 只读这里和 Qdrant，不读 .env，不返回任何密钥。
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    collection            TEXT NOT NULL,
    document_id           TEXT NOT NULL,
    source_path           TEXT NOT NULL,
    file_hash             TEXT NOT NULL,
    pipeline_fingerprint  TEXT NOT NULL,
    chunk_ids             TEXT NOT NULL,
    token_count           INTEGER NOT NULL,
    indexed_at            TEXT NOT NULL,
    PRIMARY KEY (collection, document_id)
);
CREATE TABLE IF NOT EXISTS collections (
    collection            TEXT PRIMARY KEY,
    strategy              TEXT NOT NULL,
    pipeline_fingerprint  TEXT NOT NULL,
    fingerprint_detail    TEXT NOT NULL,
    bm25_avgdl            REAL NOT NULL,
    status                TEXT NOT NULL,
    revision              INTEGER NOT NULL DEFAULT 0,
    eval_summary          TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path = config.MANIFEST_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------------- documents ----------------
    def documents(self, collection: str) -> dict[str, dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents WHERE collection = ?", (collection,)).fetchall()
        return {r["document_id"]: {**dict(r), "chunk_ids": json.loads(r["chunk_ids"])} for r in rows}

    def upsert_document(self, collection: str, document_id: str, source_path: str, file_hash: str,
                        fingerprint: str, chunk_ids: list[str], token_count: int) -> None:
        """token_count：该文档所有 chunk 的 BM25 token 总数，用来在不重新 chunk 的情况下计算语料实际 avgdl。"""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (collection, document_id, source_path, file_hash, fingerprint, json.dumps(chunk_ids),
                 token_count, now_iso()),
            )

    def delete_document(self, collection: str, document_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE collection = ? AND document_id = ?", (collection, document_id))

    # ---------------- collections ----------------
    def collection(self, collection: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM collections WHERE collection = ?", (collection,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["fingerprint_detail"] = json.loads(data["fingerprint_detail"])
        data["eval_summary"] = json.loads(data["eval_summary"]) if data["eval_summary"] else None
        return data

    def create_collection(self, collection: str, strategy: str, fingerprint: str, detail: dict, avgdl: float) -> None:
        ts = now_iso()
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE collection = ?", (collection,))
            conn.execute(
                "INSERT OR REPLACE INTO collections VALUES (?, ?, ?, ?, ?, 'building', 0, NULL, ?, ?)",
                (collection, strategy, fingerprint, json.dumps(detail, ensure_ascii=False), avgdl, ts, ts),
            )

    def set_status(self, collection: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE collections SET status = ?, updated_at = ? WHERE collection = ?",
                         (status, now_iso(), collection))

    def bump_revision(self, collection: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE collections SET revision = revision + 1, updated_at = ? WHERE collection = ?",
                         (now_iso(), collection))

    def set_eval(self, collection: str, summary: dict) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE collections SET eval_summary = ?, updated_at = ? WHERE collection = ?",
                         (json.dumps(summary, ensure_ascii=False), now_iso(), collection))

    def drop_collection(self, collection: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE collection = ?", (collection,))
            conn.execute("DELETE FROM collections WHERE collection = ?", (collection,))
