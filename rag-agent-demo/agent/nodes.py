"""
Version 2 Graph 的节点与路由。

    agent_node               Reason：LLM 决定直接回答还是发起 search_knowledge_base
    route_after_agent        有 tool_call → retrieve；否则 → END
    retrieve_node            读取 tool_call / retrieval_query，调用 Retriever，文档写入 State
    grade_documents_node     Retrieval Grader：逐个文档判断 relevant / irrelevant
    route_after_grading      有相关文档 → build_tool_message；没有且可重试 → rewrite_query；否则 → build_tool_message
    rewrite_query_node       Query Rewrite：生成更适合检索的新 Query，retry_count += 1
    build_tool_message_node  把过滤后的 Context 包装成 ToolMessage（tool_call_id = pending_tool_call_id）

注意：从 agent 发出 tool_call 到 build_tool_message 写回 ToolMessage 之间，
retrieve / grade / rewrite 都不往 messages 里写东西，所以这期间 state["messages"][-1]
始终是 Agent 发起 tool_call 的那条 AIMessage。
"""
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel, Field

from agent.state import AgentState
from agent.tools import tools
from rag.retriever import format_documents, retrieve_documents

load_dotenv()

# Stop Condition：最多 Rewrite 几次。第一版只允许重写一次
MAX_RETRY = 1

NO_RELEVANT_CONTEXT = "知识库中没有找到与当前问题足够相关的信息。"

llm = ChatOpenAI(
    model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

# Agent 用的 LLM：bind_tools 把 search_knowledge_base 的 JSON Schema 附加到每一次请求上。
# 只有一个 pending_tool_call_id，所以要求 LLM 每次只发起一个 tool_call
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

SYSTEM_PROMPT = SystemMessage(
    "你是一个乐于助人的助手。如果问题涉及团队内部知识库（Redis 用途与规范、RAG 流程、Agent 定义），"
    "请先调用 search_knowledge_base 检索，再基于检索结果回答；其他常识问题请直接回答，不要调用工具。"
)


# ---------------------------------------------------------------------
# Structured Output：让 LLM 返回固定结构，而不是自由文本
# ---------------------------------------------------------------------
class GradeResult(BaseModel):
    relevant: bool = Field(description="文档是否包含能帮助回答用户问题的信息")


class RewriteResult(BaseModel):
    query: str = Field(description="改写后、更适合在知识库中做向量检索的 Query")


grader_llm = llm.with_structured_output(GradeResult, method="function_calling")
rewriter_llm = llm.with_structured_output(RewriteResult, method="function_calling")

GRADER_PROMPT = SystemMessage(
    "你是检索结果评估员。判断给定文档是否包含能够帮助回答用户问题的信息。"
    "文档中有与问题直接相关的内容则 relevant=true；只是主题沾边、但无法帮助回答该问题，则 relevant=false。"
)

REWRITE_PROMPT = SystemMessage(
    "你是检索 Query 改写助手。上一次用当前 Query 检索到的文档都与用户问题无关。"
    "请结合用户原始问题，改写出一个更具体、关键词更明确、更适合在团队内部知识库中做向量检索的 Query。"
    "只输出新的检索 Query，不要回答问题。"
)


def _user_question(state: AgentState) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


# ---------------------------------------------------------------------
# Agent Node（Reason）：与 Version 1 相同
# ---------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    is_first_call = not any(isinstance(m, AIMessage) for m in state["messages"])
    print("[Agent] LLM called" if is_first_call else "[Agent] LLM called again")

    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])

    for tool_call in response.tool_calls:
        print(f"[Agent] tool_call: {tool_call['name']} args={tool_call['args']}")
    if not response.tool_calls:
        print(f"[Agent] final answer: {response.content}")
    return {"messages": [response]}


def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "retrieve"
    return END


# ---------------------------------------------------------------------
# Retrieve Node：tool_call → query → retrieve_documents → documents 写入 State
# ---------------------------------------------------------------------
def retrieve_node(state: AgentState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]

    if tool_call["id"] != state.get("pending_tool_call_id"):
        # 第一次进入（Agent 刚发起一个新的 tool_call）：
        # 使用 original query —— 即 Agent 写在 tool_call.args 里的 query，
        # 同时记下 tool_call_id，并重置本轮 Workflow 的状态
        original_query = tool_call["args"]["query"]
        print(f"[Retrieve] query: {original_query}  (original query from tool_call)")
        documents = retrieve_documents(original_query)
        print(f"[Retrieve] retrieved {len(documents)} documents")
        return {
            "pending_tool_call_id": tool_call["id"],
            "retrieval_query": original_query,
            "retry_count": 0,
            "documents": documents,
            "relevant_documents": [],
        }

    # Rewrite 之后再次进入：使用 rewritten retrieval query（state["retrieval_query"]），
    # 而不是重新读取 tool_call 里的 original query
    rewritten_query = state["retrieval_query"]
    print(f"[Retrieve] query: {rewritten_query}  (rewritten retrieval query)")
    documents = retrieve_documents(rewritten_query)
    print(f"[Retrieve] retrieved {len(documents)} documents")
    return {"documents": documents}


# ---------------------------------------------------------------------
# Retrieval Grader：Question + Document → LLM → relevant = true / false
# ---------------------------------------------------------------------
def grade_documents_node(state: AgentState) -> dict:
    question = _user_question(state)
    relevant_documents = []
    for i, doc in enumerate(state["documents"], start=1):
        result = grader_llm.invoke(
            [GRADER_PROMPT, HumanMessage(f"用户问题：{question}\n\n文档：\n{doc.page_content}")]
        )
        print(f"[Grader] document {i}: {'relevant' if result.relevant else 'irrelevant'}")
        if result.relevant:
            relevant_documents.append(doc)
    print(f"[Grader] {len(relevant_documents)} relevant documents")
    return {"relevant_documents": relevant_documents}


def route_after_grading(state: AgentState) -> str:
    if state["relevant_documents"]:
        return "build_tool_message"
    if state["retry_count"] < MAX_RETRY:
        return "rewrite_query"
    print(f"[Retry] max retry reached ({state['retry_count']} / {MAX_RETRY}), stop rewriting")
    return "build_tool_message"


# ---------------------------------------------------------------------
# Query Rewrite Node：用户原始问题 + 当前 retrieval_query → 新的 retrieval_query
# ---------------------------------------------------------------------
def rewrite_query_node(state: AgentState) -> dict:
    question = _user_question(state)
    current_query = state["retrieval_query"]
    result = rewriter_llm.invoke(
        [REWRITE_PROMPT, HumanMessage(f"用户原始问题：{question}\n当前检索 Query：{current_query}")]
    )
    retry_count = state["retry_count"] + 1
    print(f"[Rewrite] original query: {current_query}")
    print(f"[Rewrite] rewritten query: {result.query}")
    print(f"[Retry] {retry_count} / {MAX_RETRY}")
    return {"retrieval_query": result.query, "retry_count": retry_count}


# ---------------------------------------------------------------------
# Build ToolMessage Node：relevant_documents → Context → ToolMessage
# ---------------------------------------------------------------------
def build_tool_message_node(state: AgentState) -> dict:
    relevant_documents = state["relevant_documents"]
    if relevant_documents:
        content = format_documents(relevant_documents)
        print(f"[Tool] filtered context ({len(relevant_documents)} documents) written as ToolMessage")
    else:
        content = NO_RELEVANT_CONTEXT
        print(f"[Tool] no relevant context, ToolMessage: {NO_RELEVANT_CONTEXT}")
    # tool_call_id 必须等于 Agent 最初那条 AIMessage 里 tool_call 的 id
    tool_message = ToolMessage(content=content, tool_call_id=state["pending_tool_call_id"])
    return {"messages": [tool_message]}
