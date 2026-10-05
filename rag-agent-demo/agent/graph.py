"""
Version 2 编排：Agent + Corrective RAG Workflow

START → agent ─(no tool_call)→ END
          │
          └─(tool_call)→ retrieve → grade_documents ─(relevant / max retry)→ build_tool_message → agent
                            ↑               │
                            │       (not relevant & can retry)
                            │               ↓
                            └──────── rewrite_query
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    agent_node,
    build_tool_message_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    route_after_agent,
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

builder.add_edge("retrieve", "grade_documents")
builder.add_conditional_edges("grade_documents", route_after_grading, ["build_tool_message", "rewrite_query"])
builder.add_edge("rewrite_query", "retrieve")  # Rewrite Loop：rewrite → retrieve → grade → rewrite ...

builder.add_edge("build_tool_message", "agent")  # Agent Loop：ToolMessage 交还给 Agent

graph = builder.compile()
