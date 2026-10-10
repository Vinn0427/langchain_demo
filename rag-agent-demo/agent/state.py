"""
AgentState（Version 4）。

保留 V3 的全部字段（retrieval_query / documents / relevant_documents / gate_level / top_rerank_score /
retry_count / pending_tool_call_id），新增四组：

1. Query Understanding：routing_mode / standalone_query / intent / entities / ... / candidate_tools
2. Multi-Tool Loop：tool_queue（当前 AIMessage 中还没执行的 tool_call）/ tool_calls_count / tool_call_signatures /
   tools_disabled / tool_log（每次 Tool 执行的结构化摘要，reducer = 追加）
3. RAG 子流程：rag_original_query（本次 search 第一次检索用的 Query）/ retrieval_status / evidence_summary / rewrite_failed
4. Evidence：used_chunk_ids（已经放进 ToolMessage 交给 LLM 的 chunk）/ sources（来自真实 ToolResult 的来源）

这些中间数据只在 Workflow 内部流转，不写进 messages，因此 LLM 看不到它们。
（Python 3.9 上继承 MessagesState 会报 NameError，所以显式写出 messages 字段。）
"""
import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # 对话历史（历史轮次的 Human / AI 最终回答 + 本轮全部消息），reducer = add_messages

    # ---------------- V4：Routing / Query Understanding ----------------
    routing_mode: str                 # v3_agent / query_understanding
    standalone_query: Optional[str]
    intent: Optional[str]             # None = 未做 Query Understanding（v3_agent）或 QU fallback
    entities: list[str]
    constraints: list[str]
    document_id: Optional[str]
    chunk_id: Optional[str]
    topic: Optional[str]
    candidate_tools: list[str]        # Candidate Tool Gating：Agent 本轮只能看到这些 Tool
    qu_confidence: Optional[float]
    qu_status: Optional[str]          # ok / recovered / fallback
    qu_violations: list[str]

    # ---------------- V4：Multi-Tool Loop ----------------
    tool_queue: list[dict]            # 最近一条 AIMessage 中尚未执行的 tool_call（顺序执行）
    tool_calls_count: int             # 本次请求已执行（含校验失败）的 Tool 调用次数
    tool_call_signatures: list[str]   # name + 规范化参数，用于 duplicate tool call detection
    tools_disabled: bool              # Loop Safety / Deadline 触发后，Agent 不再绑定任何 Tool
    tool_log: Annotated[list[dict], operator.add]
    dispatch_next: Optional[str]      # tool_dispatch 之后去哪：retrieve / agent

    # ---------------- V3：Corrective RAG（search_knowledge_base 内部）----------------
    retrieval_query: Optional[str]    # 当前真正交给 Hybrid Retrieval 的 Query（Rewrite 之后会变）
    rag_original_query: Optional[str]  # V4：本次 search 第一次检索用的 Query（standalone_query 或 Tool 参数）
    query_source: Optional[str]       # V4：standalone_query / tool_args
    documents: list[Document]         # Gate = MEDIUM 时交给 LLM Grader 的候选
    relevant_documents: list[Document]  # 最终进入 ToolMessage 的文档
    gate_level: Optional[str]         # high / medium / low / error
    top_rerank_score: Optional[float]
    retry_count: int
    pending_tool_call_id: Optional[str]  # 当前正在执行的 search_knowledge_base 的 tool_call_id
    evidence_summary: Optional[str]   # V4：给 Retrieval Rewrite 的检索证据摘要
    first_retrieval_ids: list[str]    # V4：第一次检索（Rewrite 前）的 rerank Top-K，Eval 用
    rewrite_failed: bool
    retrieval_error: Optional[str]    # V4：检索依赖失败（TIMEOUT / DEPENDENCY_ERROR）
    rag_started_at: Optional[float]

    # ---------------- V4：No-Answer / Evidence ----------------
    retrieval_status: Optional[str]   # 最近一次 search：FOUND / NOT_FOUND / FAILED
    used_chunk_ids: list[str]         # 已经交给 LLM 的 chunk（Context Dedup 跨 Tool 调用去重也用它）
    sources: list[dict]               # [{id, source, label}]：只来自真实 ToolResult，最终回答后由代码附加
    final_answer_kind: Optional[str]  # llm / no_answer / retrieval_failed / llm_error
