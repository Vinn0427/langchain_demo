"""
Version 3 Graph 的节点与路由（在 Version 2 基础上修改）。

    agent_node               Reason：LLM 决定直接回答还是发起 search_knowledge_base；最终回答以流式输出
    route_after_agent        有 tool_call → retrieve；否则 → END
    retrieve_node            V3：Hybrid Retrieval（Dense + BM25 → RRF）→ Rerank → Retrieval Gate
    route_after_gate         V3 新增：HIGH → build_tool_message；MEDIUM → grade_documents；LOW → rewrite_query
    grade_documents_node     Retrieval Grader：V3 中只判断 MEDIUM 置信度的文档
    route_after_grading      有相关文档 → build_tool_message；没有且可重试 → rewrite_query；否则 → build_tool_message
    rewrite_query_node       Query Rewrite：生成更适合检索的新 Query，retry_count += 1
    build_tool_message_node  把最终 Context 包装成 ToolMessage（tool_call_id = pending_tool_call_id）

注意：从 agent 发出 tool_call 到 build_tool_message 写回 ToolMessage 之间，
retrieve / grade / rewrite 都不往 messages 里写东西，所以这期间 state["messages"][-1]
始终是 Agent 发起 tool_call 的那条 AIMessage。
"""
from time import perf_counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph import END
from pydantic import BaseModel, Field

import config
from agent.state import AgentState
from agent.tools import tools
from observability.metrics import current, mark, timer
from rag.gate import HIGH, MEDIUM, retrieval_gate
from rag.pipeline import format_documents, retrieve

MAX_RETRY = config.MAX_RETRY  # Stop Condition：最多 Rewrite 几次

NO_RELEVANT_CONTEXT = "知识库中没有找到与当前问题足够相关的信息。"

llm = ChatOpenAI(
    model=config.CHAT_MODEL,
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    temperature=0,
    timeout=60,                                # 最多等 60 秒，超时报错，不再无限等待
    max_retries=2,                             # 超时后自动重试
    extra_body={"enable_thinking": False},     # 关闭 Qwen3 的深度思考
)

# Agent 用的 LLM：bind_tools 把 search_knowledge_base 的 JSON Schema 附加到每一次请求上。
# 只有一个 pending_tool_call_id，所以要求 LLM 每次只发起一个 tool_call
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

SYSTEM_PROMPT = SystemMessage(
    "你是一个乐于助人的助手。如果问题涉及团队内部知识库（Redis、RAG 与向量数据库、Agent、发布流程、值班与故障、"
    "MySQL、API 规范等团队内部规范），请直接调用 search_knowledge_base 检索（调用前不要输出任何文字），"
    "再基于检索结果回答；其他常识问题请直接回答，不要调用工具。"
)


# ---------------------------------------------------------------------
# Structured Output：让 LLM 返回固定结构，而不是自由文本
# ---------------------------------------------------------------------
class GradeResult(BaseModel):
    relevant: bool = Field(description="文档是否包含能帮助回答用户问题的信息")


class RewriteResult(BaseModel):
    query: str = Field(description="改写后、更适合在知识库中做检索的 Query")


grader_llm = llm.with_structured_output(GradeResult, method="function_calling")
rewriter_llm = llm.with_structured_output(RewriteResult, method="function_calling")

GRADER_PROMPT = SystemMessage(
    "你是检索结果评估员。判断给定文档是否包含能够帮助回答用户问题的信息。"
    "文档中有与问题直接相关的内容则 relevant=true；只是主题沾边、但无法帮助回答该问题，则 relevant=false。"
)

REWRITE_PROMPT = SystemMessage(
    "你是检索 Query 改写助手。上一次用当前 Query 检索到的文档都与用户问题无关或相关度很低。"
    "请结合用户原始问题，改写出一个更具体、关键词更明确、更适合在团队内部知识库中检索的 Query。"
    "只输出新的检索 Query，不要回答问题。"
)


def _user_question(state: AgentState) -> str:
    return next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))


# ---------------------------------------------------------------------
# Agent Node（Reason）：与 V2 相同的决策逻辑，V3 改为流式调用
# ---------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """
    一次 LLM 调用要么产出 tool_call（内部决策），要么产出最终回答（用户可见），但只有看到输出才知道是哪一种。
    所以对每次调用都先记下 call_start，再根据第一个有意义的 chunk 分类：
      - 第一个 chunk 带 tool_call_chunks → 内部决策：不向用户输出任何内容，耗时记为 agent_decision
      - 第一个 chunk 带文本 content     → 最终回答：call_start 记为 final_llm_start，此刻记为 final_first_token，
                                          文本通过 stream writer 推给 main.py 打印（custom stream）
    tool_call 的 chunk 永远不会进入 writer，因此不会混入用户 Answer Stream，也不会被计入 TTFT。
    """
    is_first_call = not any(isinstance(m, AIMessage) for m in state["messages"])
    print("[Agent] LLM called" if is_first_call else "[Agent] LLM called again")

    writer = get_stream_writer()
    call_start = perf_counter()
    kind = None  # "tool_call" / "answer"
    response = None
    for chunk in llm_with_tools.stream([SYSTEM_PROMPT] + state["messages"]):
        response = chunk if response is None else response + chunk
        if kind is None:
            if chunk.tool_call_chunks:
                kind = "tool_call"
            elif chunk.content:
                kind = "answer"
                mark("final_llm_start", call_start)
                mark("final_first_token")
        if kind == "answer" and chunk.content:
            writer({"answer_token": chunk.content})
    call_end = perf_counter()

    metrics = current()
    if response.tool_calls:
        if kind == "answer":
            # 模型先输出了一段文字、随后又发起 tool_call：这段文字不是最终回答，撤回 TTFT 标记
            writer({"retract": True})
            for event in ("final_llm_start", "final_first_token", "first_visible_token"):
                metrics and metrics.marks.pop(event, None)
            metrics and metrics.notes.setdefault("preamble", "模型在 tool_call 前输出了前导文本，已撤回，不计入 TTFT")
        metrics and metrics.add("agent_decision", call_end - call_start)
        for tool_call in response.tool_calls:
            print(f"[Agent] tool_call: {tool_call['name']} args={tool_call['args']}")
    else:
        metrics and metrics.add("final_llm", call_end - call_start)
        mark("final_llm_end", call_end)
        if kind is None:  # 空回答
            mark("final_llm_start", call_start)
            mark("final_first_token", call_end)
        if metrics is not None:
            metrics.final_message_id = response.id

    return {"messages": [AIMessage(content=response.content, tool_calls=response.tool_calls, id=response.id)]}


def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "retrieve"
    return END


# ---------------------------------------------------------------------
# Retrieve Node：tool_call → query → Hybrid Retrieval → Rerank → Retrieval Gate
# ---------------------------------------------------------------------
def _short(docs, score_key: str, n: int = 5) -> str:
    return ", ".join(f"{d.metadata['chunk_id']}({d.metadata[score_key]:.3f})" for d in docs[:n]) or "-"


def retrieve_node(state: AgentState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]
    is_new_tool_call = tool_call["id"] != state.get("pending_tool_call_id")

    if is_new_tool_call:
        # 第一次进入（Agent 刚发起一个新的 tool_call）：使用 original query，记下 tool_call_id，重置本轮 Workflow 状态
        query = tool_call["args"]["query"]
        print(f"[Retrieve] query: {query}  (original query from tool_call)")
    else:
        # Rewrite 之后再次进入：使用 rewritten retrieval query，而不是重新读取 tool_call 里的 original query
        query = state["retrieval_query"]
        print(f"[Retrieve] query: {query}  (rewritten retrieval query)")

    result = retrieve(query)
    print(f"[Retrieve] dense  top: {_short(result.dense, 'dense_score')}")
    print(f"[Retrieve] sparse top: {_short(result.sparse, 'sparse_score')}")
    print(f"[Retrieve] rrf    top: {_short(result.fused, 'rrf_score')}")
    print(f"[Rerank]   top: {_short(result.reranked, 'rerank_score')}")

    decision = retrieval_gate(result.reranked)
    print(
        f"[Gate] {decision.level.upper()} (top rerank_score={decision.top_score:.3f}, "
        f"high>={config.HIGH_CONFIDENCE_THRESHOLD}, low<{config.LOW_CONFIDENCE_THRESHOLD})"
        + (f" → pass {len(decision.passed)} documents" if decision.level == HIGH else "")
        + (f" → {len(decision.to_grade)} documents to LLM Grader" if decision.level == MEDIUM else "")
    )

    update = {
        "retrieval_query": query,
        "gate_level": decision.level,
        "top_rerank_score": decision.top_score,
        "documents": decision.to_grade,
        "relevant_documents": decision.passed,
    }
    if is_new_tool_call:
        update.update(pending_tool_call_id=tool_call["id"], retry_count=0)
    return update


def _rewrite_or_stop(state: AgentState) -> str:
    if state["retry_count"] < MAX_RETRY:
        return "rewrite_query"
    print(f"[Retry] max retry reached ({state['retry_count']} / {MAX_RETRY}), stop rewriting")
    return "build_tool_message"


def route_after_gate(state: AgentState) -> str:
    if state["gate_level"] == HIGH:
        return "build_tool_message"   # 高置信度：跳过 LLM Grader
    if state["gate_level"] == MEDIUM:
        return "grade_documents"      # 边界情况：交给 LLM 做语义判断
    return _rewrite_or_stop(state)    # 低置信度：没有值得判断的文档，直接 Rewrite


# ---------------------------------------------------------------------
# Retrieval Grader：Question + Document → LLM → relevant = true / false（V3 只处理 MEDIUM）
# ---------------------------------------------------------------------
def grade_documents_node(state: AgentState) -> dict:
    question = _user_question(state)
    relevant_documents = []
    with timer("llm_grader"):
        for doc in state["documents"]:
            result = grader_llm.invoke(
                [GRADER_PROMPT, HumanMessage(f"用户问题：{question}\n\n文档：\n{doc.page_content}")]
            )
            relevant = bool(result and result.relevant)  # structured output 返回 None 时按不相关处理
            print(
                f"[Grader] {doc.metadata['chunk_id']} (rerank_score={doc.metadata['rerank_score']:.3f}): "
                f"{'relevant' if relevant else 'irrelevant'}{'' if result else ' (structured output None)'}"
            )
            if relevant:
                relevant_documents.append(doc)
    print(f"[Grader] {len(relevant_documents)} relevant documents")
    return {"relevant_documents": relevant_documents}


def route_after_grading(state: AgentState) -> str:
    if state["relevant_documents"]:
        return "build_tool_message"
    return _rewrite_or_stop(state)


# ---------------------------------------------------------------------
# Query Rewrite Node：用户原始问题 + 当前 retrieval_query → 新的 retrieval_query
# ---------------------------------------------------------------------
def rewrite_query_node(state: AgentState) -> dict:
    question = _user_question(state)
    current_query = state["retrieval_query"]
    messages = [REWRITE_PROMPT, HumanMessage(f"用户原始问题：{question}\n当前检索 Query：{current_query}")]
    with timer("query_rewrite"):
        result = rewriter_llm.invoke(messages)
        if result is not None:
            new_query = result.query
        else:
            # 个别模型偶尔不按要求调用 function（structured output 返回 None）：退化为普通文本输出
            new_query = llm.invoke(messages).content.strip() or question
            print("[Rewrite] structured output returned None, fallback to plain text")
    retry_count = state["retry_count"] + 1
    print(f"[Rewrite] original query: {current_query}")
    print(f"[Rewrite] rewritten query: {new_query}")
    print(f"[Retry] {retry_count} / {MAX_RETRY}")
    return {"retrieval_query": new_query, "retry_count": retry_count}


# ---------------------------------------------------------------------
# Build ToolMessage Node：relevant_documents → Context → ToolMessage
# ---------------------------------------------------------------------
def build_tool_message_node(state: AgentState) -> dict:
    with timer("build_tool_message"):
        relevant_documents = state["relevant_documents"]
        if relevant_documents:
            content = format_documents(relevant_documents)
        else:
            content = NO_RELEVANT_CONTEXT
        # tool_call_id 必须等于 Agent 最初那条 AIMessage 里 tool_call 的 id
        tool_message = ToolMessage(content=content, tool_call_id=state["pending_tool_call_id"])
    if relevant_documents:
        ids = ", ".join(d.metadata["chunk_id"] for d in relevant_documents)
        print(f"[Tool] context ({len(relevant_documents)} documents: {ids}) written as ToolMessage")
    else:
        print(f"[Tool] no relevant context, ToolMessage: {NO_RELEVANT_CONTEXT}")
    return {"messages": [tool_message]}
