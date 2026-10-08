"""
Version 3 编排：Agent + Hybrid RAG + Retrieval Gate

START → agent ─(no tool_call)→ END
          │
          └─(tool_call)→ retrieve ──(HIGH)──────────────────────────→ build_tool_message → agent
                         (Hybrid → RRF → Rerank → Gate)                   ↑
                            ↑      ├─(MEDIUM)→ grade_documents ─(relevant / max retry)┘
                            │      │                 │
                            │      └─(LOW)──┐  (not relevant & can retry)
                            │               ↓        ↓
                            └──────────── rewrite_query
                                       (LOW & max retry → build_tool_message)

与 Version 2 相比只改了一处编排：retrieve 之后不再固定进入 grade_documents，
而是由 route_after_gate 根据 rerank_score 分三路。Hybrid / RRF / Rerank 是 rag 模块内部的 pipeline，不拆成 Node。
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    agent_node,
    build_tool_message_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    route_after_agent,
    route_after_gate,
    route_after_grading,
)
from agent.state import AgentState

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_documents", grade_documents_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("build_tool_message", build_tool_message_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["retrieve", END])

builder.add_conditional_edges("retrieve", route_after_gate, ["build_tool_message", "grade_documents", "rewrite_query"])
builder.add_conditional_edges("grade_documents", route_after_grading, ["build_tool_message", "rewrite_query"])
builder.add_edge("rewrite_query", "retrieve")  # Rewrite Loop：rewrite → retrieve → (gate / grade) → rewrite ...

builder.add_edge("build_tool_message", "agent")  # Agent Loop：ToolMessage 交还给 Agent

graph = builder.compile()
