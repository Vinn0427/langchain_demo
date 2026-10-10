"""
V4 Chunking：按 Markdown 标题层级切分，并记录完整标题路径；raw_text 与 retrieval_text 分离。

切分规则（CHUNK_STRATEGY_VERSION = heading-v1）：
    - "# " 是文档标题（document_title），不单独成为 chunk
    - 每个 level ≤ CHUNK_SPLIT_LEVEL（默认 3，即 ## 和 ###）的标题开启一个新小节
    - 正文为空的小节（例如只有 ### 子标题的 "## 部署"）不产生 chunk，但它的标题会进入子小节的 section_path
    - 超过 CHUNK_MAX_CHARS 的小节按句子切分，相邻片段保留 CHUNK_OVERLAP_CHARS 重叠
    - chunk_id = {document_id}_{序号:03d}，在同一文档内按出现顺序编号

raw_text      ：只有正文（不含标题行）—— 最终交给 LLM 的 Context 用它
retrieval_text：按 enrichment 策略拼接标题信息 —— Dense Embedding / BM25 / Reranker 用它
    A：raw_text
    B：文档标题 + 当前小节标题 + raw_text
    C：文档标题 + 完整标题路径 + raw_text
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config

STRATEGIES = ("A", "B", "C")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|\n")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    source: str
    document_title: str
    section_path: list[str]           # 不含文档标题，例如 ["高可用", "脑裂处理"]
    raw_text: str
    retrieval_text: str = ""
    content_hash: str = ""            # sha256(retrieval_text)：增量更新时用来复用已有 dense 向量
    extra: dict = field(default_factory=dict)

    @property
    def section_title(self) -> str:
        return self.section_path[-1] if self.section_path else self.document_title

    def payload(self, strategy: str) -> dict:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "document_title": self.document_title,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "raw_text": self.raw_text,
            "retrieval_text": self.retrieval_text,
            "content_hash": self.content_hash,
            "chunk_strategy": strategy,
        }


def build_retrieval_text(document_title: str, section_path: list[str], raw_text: str, strategy: str) -> str:
    if strategy == "A":
        return raw_text
    if strategy == "B":
        section = section_path[-1] if section_path else ""
        header = f"{document_title}\n> {section}" if section else document_title
        return f"{header}\n\n{raw_text}"
    if strategy == "C":
        header = "\n".join([document_title] + [f"> {s}" for s in section_path])
        return f"{header}\n\n{raw_text}"
    raise ValueError(f"unknown chunk strategy: {strategy}")


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [s for s in _SENTENCE_END.split(text) if s and s.strip()]
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > max_chars:
            pieces.append(current.strip())
            current = current[-overlap:] if overlap else ""
        current += sentence
        while len(current) > max_chars:   # 单句本身超长：硬切
            pieces.append(current[:max_chars].strip())
            current = current[max_chars - overlap:]
    if current.strip():
        pieces.append(current.strip())
    return pieces


def parse_sections(markdown: str, fallback_title: str, split_level: int) -> tuple[str, list[tuple[list[str], str]]]:
    """→ (document_title, [(section_path, body)])"""
    document_title: Optional[str] = None
    path: list[tuple[int, str]] = []   # [(level, title)]
    sections: list[tuple[list[str], list[str]]] = []
    body: list[str] = []
    current_path: list[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append((list(current_path), text))

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 1 and document_title is None:
                flush()
                body, document_title = [], title
                continue
            if level <= split_level:
                flush()
                body = []
                path = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
                current_path = [t for _, t in path]
                continue
        body.append(line)
    flush()
    return document_title or fallback_title, sections


def chunk_document(path: Path, strategy: str = config.CHUNK_STRATEGY) -> list[Chunk]:
    markdown = path.read_text(encoding="utf-8")
    title, sections = parse_sections(markdown, path.stem, config.CHUNK_SPLIT_LEVEL)
    try:
        source = str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        source = str(path)
    chunks, index = [], 0
    for section_path, text in sections:
        for piece in _split_long(text, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS):
            index += 1
            retrieval_text = build_retrieval_text(title, section_path, piece, strategy)
            chunks.append(Chunk(
                chunk_id=f"{path.stem}_{index:03d}", document_id=path.stem, chunk_index=index, source=source,
                document_title=title, section_path=section_path, raw_text=piece, retrieval_text=retrieval_text,
                content_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
            ))
    return chunks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def splitter_config() -> dict:
    return {
        "split_level": config.CHUNK_SPLIT_LEVEL,
        "max_chars": config.CHUNK_MAX_CHARS,
        "overlap_chars": config.CHUNK_OVERLAP_CHARS,
    }


# ---------------------------------------------------------------------
# Chunk Analysis
# ---------------------------------------------------------------------
def token_length(text: str) -> int:
    import jieba

    return sum(1 for t in jieba.lcut(text) if t.strip())


def analyze_chunks(chunks: list[Chunk], field_name: str = "raw_text") -> dict:
    import numpy as np

    lengths = np.array([token_length(getattr(c, field_name)) for c in chunks]) if chunks else np.array([0])
    chars = np.array([len(getattr(c, field_name)) for c in chunks]) if chunks else np.array([0])
    short = [c.chunk_id for c, n in zip(chunks, lengths) if n < config.CHUNK_TOO_SHORT_TOKENS]
    long = [c.chunk_id for c, n in zip(chunks, lengths) if n > config.CHUNK_TOO_LONG_TOKENS]
    return {
        "field": field_name,
        "chunk_count": len(chunks),
        "token_avg": round(float(lengths.mean()), 1),
        "token_p50": round(float(np.percentile(lengths, 50)), 1),
        "token_p95": round(float(np.percentile(lengths, 95)), 1),
        "token_min": int(lengths.min()),
        "token_max": int(lengths.max()),
        "char_avg": round(float(chars.mean()), 1),
        "too_short_threshold": config.CHUNK_TOO_SHORT_TOKENS,
        "too_long_threshold": config.CHUNK_TOO_LONG_TOKENS,
        "too_short_ratio": round(len(short) / max(len(chunks), 1), 3),
        "too_long_ratio": round(len(long) / max(len(chunks), 1), 3),
        "too_short_chunks": short,
        "too_long_chunks": long,
    }
