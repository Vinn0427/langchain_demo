"""
Query 层的结构化 Contract。

QueryUnderstandingResult ：Query Understanding 一次 LLM 调用的输出（Contextual Rewrite + Intent + Entity + Candidate Tool）
RetrievalRewriteResult   ：Retrieval Rewrite 的输出（只在检索失败后使用，只负责提高检索质量）
GradeResult              ：Retrieval Grader 的输出
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from tools.schemas import CHUNK_ID_PATTERN, DOCUMENT_ID_PATTERN

Intent = Literal[
    "DIRECT",            # 与团队知识无关的通用问题，不需要 Tool
    "KNOWLEDGE_SEARCH",  # 团队系统 / 规范 / 流程 / 运维等知识问题 → search_knowledge_base
    "DOCUMENT_LOOKUP",   # 明确要某篇文档全文 / 大纲 → get_document
    "CHUNK_LOOKUP",      # 明确引用 chunk_id → get_chunk
    "DOCUMENT_LIST",     # 知识库有哪些文档 → list_documents
    "INDEX_STATUS",      # 索引本身的状态 → get_index_status
    "MULTI_TOOL",        # 需要两类及以上 Tool 才能完成
]


class QueryUnderstandingResult(BaseModel):
    standalone_query: str = Field(min_length=1, max_length=300, description="不依赖上下文、可单独理解的问题")
    intent: Intent
    entities: list[str] = Field(default_factory=list, description="关键实体：系统、组件、代号、ID")
    constraints: list[str] = Field(default_factory=list, description="数字 / 时间 / 版本 / 否定条件的原文")
    document_id: Optional[str] = Field(default=None, description="用户明确指定的 document_id")
    chunk_id: Optional[str] = Field(default=None, description="用户明确引用的 chunk_id")
    topic: Optional[str] = Field(default=None, description="DOCUMENT_LIST 的主题过滤词")
    candidate_tools: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @field_validator("document_id", "chunk_id", "topic", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            return None
        return v

    @field_validator("document_id")
    @classmethod
    def _doc_format(cls, v):
        import re

        return v if v is None or re.fullmatch(DOCUMENT_ID_PATTERN, v) else None

    @field_validator("chunk_id")
    @classmethod
    def _chunk_format(cls, v):
        import re

        return v if v is None or re.fullmatch(CHUNK_ID_PATTERN, v) else None


class RetrievalRewriteResult(BaseModel):
    query: str = Field(min_length=2, max_length=200, description="改写后、更适合在知识库中检索的 Query")


class GradeResult(BaseModel):
    relevant: bool = Field(description="文档是否包含能帮助回答用户问题的信息")
