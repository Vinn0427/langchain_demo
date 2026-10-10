"""
Retrieval Rewrite（与 query/understanding.py 的 Contextual Rewrite 是两件事，不合并成一个 Node）

只在检索失败后使用：Retrieval Gate = LOW，或 MEDIUM + Grader 判定 0 个相关。
输入：standalone_query（第一次检索用的 Query）+ current_retrieval_query + 检索证据摘要 + retry_count
目的：只提高 Retrieval Quality（补充同义词 / 专有名词 / 更具体的关键词）—— 不再做 Intent Recognition，不回答问题。

一次 LLM 调用；structured output 解析失败时对同一个 raw response 容错解析（纯文本也接受），
V3 的"parsed=None → 再调用一次 LLM"的双调用在这里被消除（见 llm/client.structured_call）。
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

import config
from llm.client import StructuredResult, structured_call
from query.schema import RetrievalRewriteResult

PROMPT = SystemMessage(
    "你是检索 Query 改写助手。上一次用当前 Query 在团队内部知识库中检索，结果与问题无关或相关度很低。"
    "请改写出一个更具体、关键词更明确、更适合检索的 Query：可以补充同义词、专有名词、相关术语，"
    "但必须保留原问题中的实体、数字、版本、否定条件和 ID，不要改变问题的意图，不要回答问题。"
    '只输出 JSON：{"query": "改写后的检索 Query"}'
)


def evidence_summary(docs: list, n: int = 3) -> str:
    if not docs:
        return "（没有召回任何文档）"
    lines = []
    for d in docs[:n]:
        m = d.metadata
        path = " > ".join(m.get("section_path") or [])
        lines.append(f"- {m['chunk_id']}（{m.get('document_title', '')} > {path}，rerank_score={m.get('rerank_score', 0):.3f}）")
    return "\n".join(lines)


def retrieval_rewrite(standalone_query: str, current_query: str, evidence: str, retry_count: int
                      ) -> tuple[Optional[str], StructuredResult]:
    messages = [
        PROMPT,
        HumanMessage(
            f"原问题：{standalone_query}\n当前检索 Query：{current_query}\n"
            f"当前检索到的前几条结果（都不够相关）：\n{evidence}\n已重试次数：{retry_count}"
        ),
    ]
    res = structured_call("query_rewrite", RetrievalRewriteResult, messages, timeout=config.REWRITE_TIMEOUT,
                          reserve=config.FINAL_ANSWER_RESERVE + config.RETRIEVAL_RESERVE, plain_text_field="query")
    if res.value is None:
        return None, res
    return res.value.query.strip(), res
