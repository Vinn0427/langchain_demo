"""
RAG 索引构建：加载 → 切分 → Embedding → VectorStore → Retriever
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

KNOWLEDGE_PATH = Path(__file__).parent.parent / "data" / "knowledge.md"


def build_retriever():
    # 1.1 加载 Markdown 文档（本质就是读一个文本文件）
    markdown_text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

    # 1.2 文本切分：按二级标题 "##" 切分，每个主题成为一个 chunk（Document）
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "topic")], strip_headers=False)
    chunks = splitter.split_text(markdown_text)
    print(f"[RAG] knowledge.md split into {len(chunks)} chunks")

    # 1.3 Embedding 模型：把文本变成向量
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        check_embedding_ctx_length=False,  # 直接发送原始文本，兼容非 OpenAI 的兼容服务商
    )

    # 1.4 VectorStore：在内存中保存 (向量, 文本)。add_documents 内部会调用 Embedding 模型
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    print(f"[RAG] {len(chunks)} chunks embedded into InMemoryVectorStore")

    # 1.5 Retriever：输入 query 字符串 → 向量化 → 相似度检索 → 返回最相关的 k 个 Document
    return vector_store.as_retriever(search_kwargs={"k": 2})
