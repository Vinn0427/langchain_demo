"""
AgentState（Version 3）。

Version 2 在 messages 之外维护 Corrective RAG Workflow 的业务状态；
Version 3 保留全部 V2 字段，只新增 Retrieval Gate 的两个结果字段：gate_level、top_rerank_score。
这些中间数据只在 Workflow 内部流转，不写进 messages，因此 LLM（Agent Node）看不到它们。

说明：这里直接用 TypedDict 写出 messages 字段，效果等同于 `class AgentState(MessagesState)`。
（在 Python 3.9 上继承 MessagesState 时，LangGraph 解析父类类型注解会报 NameError；
 显式写出来还能把 add_messages reducer 直接展示在代码里。）
"""
from typing import Annotated, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Agent 对话历史：HumanMessage / AIMessage / ToolMessage
    # reducer = add_messages：节点返回 {"messages": [新消息]} 时追加，而不是覆盖

    # ---- 以下字段没有 reducer：节点返回 {"字段": 新值} 时直接覆盖旧值 ----

    retrieval_query: Optional[str]
    # 当前真正交给 Hybrid Retrieval 的 Query。
    # 第一次 = original query（Agent 在 tool_call.args 里给出的）；Rewrite 之后 = rewritten query

    documents: list[Document]
    # V3：Gate 判为 MEDIUM 时，需要 LLM Grader 逐个判断的候选文档（rerank_score ∈ [LOW, HIGH)）

    relevant_documents: list[Document]
    # 最终进入 ToolMessage 的文档：HIGH 时由 Gate 直接给出；MEDIUM 时由 LLM Grader 筛出

    gate_level: Optional[str]
    # V3 新增：Retrieval Gate 的判断结果 high / medium / low

    top_rerank_score: Optional[float]
    # V3 新增：本次检索 Reranker 的最高分，Gate 的判断依据

    retry_count: int
    # 已经执行 Query Rewrite / Retry 的次数，用于 Stop Condition

    pending_tool_call_id: Optional[str]
    # Agent 发起 search_knowledge_base 时的 tool_call id，
    # 最终构造 ToolMessage 时必须使用它，与原始 AIMessage 中的 tool_call 对应
