"""
Version 4 编排：Query Understanding + Multi-Tool Agent + V3 Corrective Hybrid RAG

START ─(ROUTING_MODE=query_understanding)→ query_understanding ─(单 Tool 且参数确定)→ tool_dispatch
  │                                              └────────(其他)──────────→ agent
  └──(ROUTING_MODE=v3_agent)──────────────────────────────────────────────→ agent

agent ─(no tool_call)→ END
  └─(tool_calls：可以有多个)→ tool_dispatch ─(当前 tool_call 是 search_knowledge_base)→ retrieve
                                │   ↑                                             (Hybrid → RRF → Rerank → Gate)
                                │   │                    ┌─(HIGH / error)──────────────────┐
                                │   │       retrieve ────┼─(MEDIUM)→ grade_documents ──────┤
                                │   │          ↑         └─(LOW)──┐        │(0 relevant)    │
                                │   │          │                  ↓        ↓                ↓
                                │   │          └──────────── rewrite_query ──(failed)→ build_tool_message
                                │   └────────────────────────────────────────────────────────┘
                                └─(队列中所有 tool_call 都有了 ToolMessage)→ agent

Node 的数量由职责决定：Hybrid / RRF / Rerank / Gate 仍在 retrieve 一个 Node 内（rag 模块内部 pipeline）；
get_document / get_chunk / list_documents / get_index_status 都在 tool_dispatch 内执行（确定性的精确查询）。
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    agent_node,
    build_tool_message_node,
    grade_documents_node,
    query_understanding_node,
    retrieve_node,
    rewrite_query_node,
    route_after_agent,
    route_after_dispatch,
    route_after_gate,
    route_after_grading,
    route_after_qu,
    route_after_rewrite,
    route_start,
    tool_dispatch_node,
)
from agent.state import AgentState

builder = StateGraph(AgentState)

builder.add_node("query_understanding", query_understanding_node)
builder.add_node("agent", agent_node)
builder.add_node("tool_dispatch", tool_dispatch_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("build_tool_message", build_tool_message_node)

builder.add_conditional_edges(START, route_start, ["query_understanding", "agent"])
builder.add_conditional_edges("query_understanding", route_after_qu, ["tool_dispatch", "agent"])
builder.add_conditional_edges("agent", route_after_agent, ["tool_dispatch", END])
builder.add_conditional_edges("tool_dispatch", route_after_dispatch, ["retrieve", "agent"])

builder.add_conditional_edges("retrieve", route_after_gate, ["build_tool_message", "grade_documents", "rewrite_query"])
builder.add_conditional_edges("grade_documents", route_after_grading, ["build_tool_message", "rewrite_query"])
builder.add_conditional_edges("rewrite_query", route_after_rewrite, ["retrieve", "build_tool_message"])
builder.add_edge("build_tool_message", "tool_dispatch")  # 继续执行同一条 AIMessage 中剩余的 tool_call

graph = builder.compile()

RECURSION_LIMIT = 60  # MAX_TOOL_CALLS_PER_REQUEST × RAG 子流程步数 + Agent 轮次，远大于默认 25
