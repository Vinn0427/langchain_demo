"""
Agent 可调用的 Tool：search_knowledge_base。

@tool 会读取 函数名 + 参数类型 + docstring，生成 JSON Schema 交给 LLM。
LLM 看到的"工具说明书"就是下面的 docstring，它据此判断要不要调用。

Version 2 中这个 Tool 的角色变化：
- 它的 name / description / args schema 仍然通过 bind_tools 交给 LLM，告诉 LLM"什么时候该搜知识库"；
- 但 Graph 不再直接执行这个函数，而是由 retrieve_node 读取 tool_call，
  调用底层的 retrieve_documents()，把 query / documents / tool_call_id 分别写入 AgentState，
  经过 Grader 过滤后，再由 build_tool_message_node 生成 ToolMessage。
- 函数体保留完整实现，单独调用时（search_knowledge_base.invoke(...)）仍然可以返回检索结果。
"""
from langchain_core.tools import tool

from rag.retriever import format_documents, retrieve_documents


@tool
def search_knowledge_base(query: str) -> str:
    """检索团队内部知识库。知识库包含：团队 Redis 的用途与使用规范、RAG 的基本流程、Agent 的基本定义。
    当用户问题涉及这些主题或明确提到"知识库"时调用；与这些主题无关的常识问题不要调用。"""
    return format_documents(retrieve_documents(query))


tools = [search_knowledge_base]
