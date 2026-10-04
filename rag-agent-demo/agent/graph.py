"""
Agent 编排：START → agent →(tool_call?)→ tools → agent → ... → END
"""
from langgraph.graph import END, START, MessagesState, StateGraph

from agent.nodes import agent_node, route_after_agent, tool_node

builder = StateGraph(MessagesState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
builder.add_edge("tools", "agent")  # Tool 执行完必定回到 Agent，与上一行共同构成循环

graph = builder.compile()
