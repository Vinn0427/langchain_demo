"""
Retrieval Gate：根据 rerank_score 决定下一步，把"确定性的分数判断"放在 LLM Grader 之前。

    top1 = 最高的 rerank_score

    top1 ≥ HIGH                → HIGH   ：直接通过，score ≥ HIGH 的文档进入 ToolMessage，不调用 LLM
    LOW ≤ top1 < HIGH          → MEDIUM ：LOW ≤ score < HIGH 的文档交给 LLM Grader 逐个判断（score < LOW 的直接丢弃）
    top1 < LOW 或没有候选       → LOW    ：没有值得判断的文档，直接 Query Rewrite（重试次数用完则返回"没找到"）

阈值集中在 config.py，是 Demo 初始值，正式系统应基于 Eval Dataset 调参。
"""
from dataclasses import dataclass, field

from langchain_core.documents import Document

import config
from observability.metrics import timer

HIGH, MEDIUM, LOW = "high", "medium", "low"


@dataclass
class GateDecision:
    level: str
    top_score: float
    passed: list[Document] = field(default_factory=list)    # HIGH：直接通过的文档
    to_grade: list[Document] = field(default_factory=list)  # MEDIUM：需要 LLM Grader 判断的文档


def retrieval_gate(
    reranked: list[Document],
    high: float = config.HIGH_CONFIDENCE_THRESHOLD,
    low: float = config.LOW_CONFIDENCE_THRESHOLD,
) -> GateDecision:
    with timer("retrieval_gate"):
        top_score = reranked[0].metadata["rerank_score"] if reranked else 0.0
        if top_score >= high:
            passed = [d for d in reranked if d.metadata["rerank_score"] >= high]
            return GateDecision(HIGH, top_score, passed=passed)
        if top_score >= low:
            to_grade = [d for d in reranked if d.metadata["rerank_score"] >= low]
            return GateDecision(MEDIUM, top_score, to_grade=to_grade)
        return GateDecision(LOW, top_score)
