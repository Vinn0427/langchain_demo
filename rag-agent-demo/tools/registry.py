"""
Tool Registry：5 个 Tool 的名称、边界说明（Use when / Do NOT use when）、Args Schema、执行器。

    search_knowledge_base  kind=rag     由 Graph 的 retrieve → gate → grade → rewrite → build_tool_message 执行
    get_document           kind=exact   Qdrant 按 document_id 精确过滤
    get_chunk              kind=exact   Qdrant 按 uuid5(chunk_id) 精确取 point
    list_documents         kind=exact   扫描 payload 元数据
    get_index_status       kind=system  alias / manifest / Qdrant count

- langchain_tools(names)：动态 Tool Binding —— 只把候选 Tool 的 JSON Schema 交给 LLM
- validate()：执行前用 Pydantic 校验参数；无效参数返回 INVALID_ARGUMENT，不会让 Agent 崩溃
- execute()：TOOL_TIMEOUT 超时控制 + 异常 → 结构化 error_code（不把 traceback 交给 LLM）
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

import config
from tools.document import get_chunk, get_document, list_documents
from tools.index_status import get_index_status
from tools.schemas import (
    GetChunkArgs,
    GetDocumentArgs,
    GetIndexStatusArgs,
    ListDocumentsArgs,
    SearchKnowledgeBaseArgs,
    ToolResult,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: Type[BaseModel]
    kind: str                                            # rag / exact / system
    executor: Optional[Callable[[BaseModel], ToolResult]] = None


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec for spec in [
        ToolSpec(
            "search_knowledge_base",
            "语义 + 关键词检索团队内部知识库（Hybrid Retrieval + Rerank），返回与问题最相关的若干段落。"
            "知识库覆盖 Redis / MySQL 的使用与运维、RAG 与向量数据库、Agent 开发、发布流程、值班与故障处理、API 规范。\n"
            "Use when：用户提出关于团队系统、组件、规范、流程、运维、故障处理的知识问题，且没有给出明确的 document_id / chunk_id；"
            "包括知识库里可能不存在的技术组件（例如 Kafka），由检索结果决定有没有答案。\n"
            "Do NOT use when：用户已经明确给出 chunk_id（用 get_chunk）；用户明确要某篇文档的完整内容（用 get_document）；"
            "用户问知识库有哪些文档（用 list_documents）；用户问索引本身的状态（用 get_index_status）；"
            "与团队知识无关的通用常识问题（直接回答）。",
            SearchKnowledgeBaseArgs, "rag",
        ),
        ToolSpec(
            "get_document",
            "按 document_id 精确读取一篇文档的完整内容（或章节大纲），不做向量检索。\n"
            "Use when：用户明确指定了 document_id，或明确要求某篇文档的'完整内容 / 全文 / 整篇 / 大纲'"
            "（例如'把 redis 文档完整内容给我'→ document_id=redis）。\n"
            "Do NOT use when：用户只是提出一个知识问题（即使问题里提到 Redis、MySQL 等主题词），这时应使用 search_knowledge_base；"
            "用户引用的是 chunk_id（例如 redis_003，用 get_chunk）。",
            GetDocumentArgs, "exact", get_document,
        ),
        ToolSpec(
            "get_chunk",
            "按 chunk_id 精确读取一个 chunk 的原文。\n"
            "Use when：用户明确引用了某个 chunk_id（格式 <document_id>_<3 位数字>，例如 redis_003、mysql_ops_002）。\n"
            "Do NOT use when：用户没有给出 chunk_id；不要猜测或编造 chunk_id 去调用。",
            GetChunkArgs, "exact", get_chunk,
        ),
        ToolSpec(
            "list_documents",
            "列出知识库中可用的文档及基本元数据（document_id、标题、来源、chunk 数、章节），可按主题过滤。\n"
            "Use when：用户问'知识库里有哪些文档'、'有哪些 MySQL 相关文档'、需要先知道有哪些 document_id。\n"
            "Do NOT use when：用户问的是具体知识内容（用 search_knowledge_base）；用户问 chunk 数量 / 索引版本 / "
            "embedding 模型等索引状态（用 get_index_status）。",
            ListDocumentsArgs, "exact", list_documents,
        ),
        ToolSpec(
            "get_index_status",
            "返回当前在线索引的状态：index version、collection、文档数、chunk 数、embedding 模型、chunk 策略、最近索引时间。\n"
            "Use when：用户询问索引 / 知识库本身的规模、版本、配置或更新时间（例如'当前一共有多少 chunk'）。\n"
            "Do NOT use when：用户问具体知识内容，或想看文档列表（用 list_documents）。",
            GetIndexStatusArgs, "system", get_index_status,
        ),
    ]
}

ALL_TOOLS = list(TOOLS)

# Candidate Tool Gating：Intent → 默认候选 Tool（Query Understanding 输出的 candidate_tools 会与它取并集）
INTENT_TOOLS: dict[str, list[str]] = {
    "DIRECT": [],
    "KNOWLEDGE_SEARCH": ["search_knowledge_base", "get_document", "get_chunk"],
    "DOCUMENT_LOOKUP": ["get_document", "list_documents"],
    "CHUNK_LOOKUP": ["get_chunk", "get_document"],
    "DOCUMENT_LIST": ["list_documents"],
    "INDEX_STATUS": ["get_index_status"],
    "MULTI_TOOL": ALL_TOOLS,
}


def _never_called(**_kwargs):  # Graph 自己执行 Tool；StructuredTool 只用来生成 JSON Schema
    raise RuntimeError("tools are executed by the graph, not by LangChain")


_LC_TOOLS = {
    name: StructuredTool.from_function(func=_never_called, name=name, description=spec.description,
                                       args_schema=spec.args_schema)
    for name, spec in TOOLS.items()
}


def langchain_tools(names: list[str]) -> list[StructuredTool]:
    return [_LC_TOOLS[n] for n in names if n in _LC_TOOLS]


def validate(name: str, args) -> tuple[Optional[BaseModel], Optional[ToolResult]]:
    spec = TOOLS.get(name)
    if spec is None:
        return None, ToolResult.error("INVALID_ARGUMENT", f"未知的 Tool：{name}。可用 Tool：{', '.join(ALL_TOOLS)}")
    if not isinstance(args, dict):
        return None, ToolResult.error("INVALID_ARGUMENT", f"{name} 的参数不是合法的 JSON 对象")
    try:
        return spec.args_schema.model_validate(args), None
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '(root)'}: {e['msg']}" for e in exc.errors()[:3])
        return None, ToolResult.error("INVALID_ARGUMENT", f"{name} 参数校验失败：{problems}")


_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


def execute(name: str, args: BaseModel, timeout: float = config.TOOL_TIMEOUT) -> ToolResult:
    spec = TOOLS[name]
    ctx = contextvars.copy_context()
    future = _executor.submit(ctx.run, spec.executor, args)
    try:
        result = future.result(timeout=timeout)
    except FutureTimeout:
        return ToolResult.error("TIMEOUT", f"{name} 执行超过 {timeout}s")
    except Exception as exc:  # noqa: BLE001
        module = type(exc).__module__
        if module.startswith(("qdrant_client", "httpx", "httpcore", "grpc")) or isinstance(exc, (ConnectionError, OSError)):
            return ToolResult.error("DEPENDENCY_ERROR", f"{name} 依赖服务不可用：{type(exc).__name__}")
        return ToolResult.error("INTERNAL_ERROR", f"{name} 执行异常：{type(exc).__name__}")
    if not isinstance(result, ToolResult):
        return ToolResult.error("INTERNAL_ERROR", f"{name} 返回了非 ToolResult 类型：{type(result).__name__}")
    return result
