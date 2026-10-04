"""
Agent 可调用的 Tool：把 Retriever 封装成 search_knowledge_base。

@tool 会读取 函数名 + 参数类型 + docstring，生成 JSON Schema 交给 LLM。
LLM 看到的"工具说明书"就是下面的 docstring，它据此判断要不要调用。
"""
from langchain_core.tools import tool

from rag.retriever import build_retriever

# 索引阶段：import 本模块时执行一次（与原 demo.py 一样在启动时构建）
retriever = build_retriever()


@tool
def search_knowledge_base(query: str) -> str:
    """检索团队内部知识库。知识库包含：团队 Redis 的用途与使用规范、RAG 的基本流程、Agent 的基本定义。
    当用户问题涉及这些主题或明确提到"知识库"时调用；与这些主题无关的常识问题不要调用。"""
    print(f"[Tool] query: {query}")
    docs = retriever.invoke(query)
    print(f"[Tool] retrieved {len(docs)} documents")
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


tools = [search_knowledge_base]
