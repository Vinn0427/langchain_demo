"""
Agent 可调用的 Tool：search_knowledge_base。

@tool 会读取 函数名 + 参数类型 + docstring，生成 JSON Schema 交给 LLM。
LLM 看到的"工具说明书"就是下面的 docstring，它据此判断要不要调用。

与 Version 2 相同：这个 Tool 的 name / description / args schema 通过 bind_tools 交给 LLM，
但 Graph 不直接执行它，而是由 retrieve_node 读取 tool_call，走 Hybrid Retrieval → Rerank → Gate 流程，
最后由 build_tool_message_node 生成 ToolMessage。
函数体保留完整实现，单独调用时（search_knowledge_base.invoke(...)）仍然可以返回检索结果。
"""
from langchain_core.tools import tool

from rag.pipeline import format_documents, retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """检索团队内部知识库。知识库包含：Redis 用途/限流/故障排查、RAG 流程与文档切分、向量数据库选型、
    Agent 定义与工具调用规范、发布流程（发布窗口/灰度/回滚）、值班与故障处理、MySQL 使用规范、API 错误码与鉴权规范。
    当用户问题涉及这些团队内部规范或明确提到"知识库"时调用；与这些主题无关的常识问题不要调用。"""
    return format_documents(retrieve(query).reranked)


tools = [search_knowledge_base]
