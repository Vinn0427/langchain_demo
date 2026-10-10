"""
V4 Tool Contract：所有 Tool 的输入（Pydantic Args Schema）和输出（ToolResult）都在这里定义。

Tool 不允许"有时返回字符串、有时 None、有时抛异常、有时 dict"：执行器统一返回 ToolResult，
异常由 tools/registry.py 转换成结构化的 error_code，不会把 Python traceback 交给 LLM。
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# document_id = data/ 下的文件名（不含 .md）；每一段以字母开头，因此不会和 chunk_id（以 _三位数字 结尾）混淆
DOCUMENT_ID_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*$"
CHUNK_ID_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*_[0-9]{3}$"

ErrorCode = Literal[
    "INVALID_ARGUMENT",       # 参数不符合 Args Schema / 未知 Tool / 参数 JSON 无法解析
    "NOT_FOUND",              # 参数合法，但目标不存在（业务结果，不是系统故障）
    "TIMEOUT",                # 执行超过 TOOL_TIMEOUT / 依赖的模型调用超时
    "DEPENDENCY_ERROR",       # Qdrant / Embedding 等依赖不可用
    "INTERNAL_ERROR",         # 其他未预期的异常
    "DUPLICATE_TOOL_CALL",    # Loop Safety：同一 Tool + 完全相同参数重复调用
    "TOOL_BUDGET_EXCEEDED",   # Loop Safety：超过 MAX_TOOL_CALLS_PER_REQUEST
]


class ToolResult(BaseModel):
    success: bool
    data: Optional[Any] = None
    error_code: Optional[ErrorCode] = None
    message: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)    # 可以作为回答证据展示的 ID（chunk_id 或 document_id）
    chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    retrieval_status: Optional[Literal["FOUND", "NOT_FOUND", "FAILED"]] = None  # 只有 search_knowledge_base 有

    @classmethod
    def error(cls, code: str, message: str, **kwargs) -> "ToolResult":
        return cls(success=False, error_code=code, message=message, **kwargs)


# ---------------------------------------------------------------------
# Args Schemas
# ---------------------------------------------------------------------
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchKnowledgeBaseArgs(_Args):
    query: str = Field(
        min_length=2, max_length=200,
        description="自包含的检索 Query：补全指代，保留原问题中的实体、数字、时间、版本和否定条件",
    )


class GetDocumentArgs(_Args):
    document_id: str = Field(
        pattern=DOCUMENT_ID_PATTERN, max_length=64,
        description="文档 ID，即 data/ 下的文件名（不含 .md），例如 redis、redis_ops、mysql_ops",
    )
    view: Literal["full", "outline"] = Field(
        default="full", description="full = 返回全文；outline = 只返回章节结构和 chunk_id",
    )


class GetChunkArgs(_Args):
    chunk_id: str = Field(
        pattern=CHUNK_ID_PATTERN, max_length=72,
        description="chunk ID，格式为 <document_id>_<3 位序号>，例如 redis_003、redis_ops_006",
    )


class ListDocumentsArgs(_Args):
    topic: Optional[str] = Field(
        default=None, min_length=1, max_length=50,
        description="可选：主题关键词，按文档 ID / 标题 / 章节标题做不区分大小写的包含匹配，例如 MySQL；不传则返回全部文档",
    )


class GetIndexStatusArgs(_Args):
    pass
