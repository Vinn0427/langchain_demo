"""
Agent Graph 的节点与路由：agent_node（Reason）、tool_node（Act + Observe）、route_after_agent。

State 直接使用 LangGraph 内置的 MessagesState，它等价于：
    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
add_messages 是 reducer：节点返回 {"messages": [新消息]} 时，新消息会被"追加"而不是"覆盖"。
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState

from agent.tools import tools

load_dotenv()

tools_by_name = {t.name: t for t in tools}

# LLM：bind_tools 会把工具的 JSON Schema 附加到每一次 LLM 请求上
llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    "你是一个乐于助人的助手。如果问题涉及团队内部知识库（Redis 用途与规范、RAG 流程、Agent 定义），"
    "请先调用 search_knowledge_base 检索，再基于检索结果回答；其他常识问题请直接回答，不要调用工具。"
)


# Agent Node（Reason）：调用 LLM，由 LLM 决定"直接回答"还是"发起 tool_call"
def agent_node(state: MessagesState) -> dict:
    is_first_call = not any(isinstance(m, AIMessage) for m in state["messages"])
    print("[Agent] LLM called" if is_first_call else "[Agent] LLM called again")

    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])

    for tool_call in response.tool_calls:
        print(f"[Agent] tool_call: {tool_call['name']} args={tool_call['args']}")
    if not response.tool_calls:
        print(f"[Agent] final answer: {response.content}")
    return {"messages": [response]}  # 交给 add_messages 追加到 state["messages"]


# Tool Node（Act + Observe）：执行上一条 AIMessage 中的每个 tool_call，结果包装成 ToolMessage
# 这段逻辑等价于 LangGraph 内置的 ToolNode(tools)，这里手写出来便于理解
def tool_node(state: MessagesState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        output = selected_tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    print(f"[Tool] {len(tool_messages)} ToolMessage written back to messages")
    return {"messages": tool_messages}  # 写回 messages，下一轮 LLM 就能"看到"检索结果


# 路由函数：Conditional Edge 的判断依据 —— 最后一条消息里有没有 tool_call？
def route_after_agent(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
