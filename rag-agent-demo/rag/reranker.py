"""
Reranker：Query + Document → relevance score，对 Hybrid 候选重新排序。

与 Embedding（Bi-Encoder，query 和文档分别编码再算相似度）不同，
Cross-Encoder 把 (query, document) 拼在一起输入同一个 Transformer，能看到两者之间逐词的交互，
排序更准，但每个候选都要跑一次模型，所以只用于少量候选（RRF Top N）的精排。

默认实现：BAAI/bge-reranker-base（中英双语 Cross-Encoder），通过 fastembed 以 ONNX Runtime 在本地 CPU 运行。
模型输出的是 logit，这里用 sigmoid 映射到 [0, 1]，作为 rerank_score，供 Retrieval Gate 使用。

替换方案：实现同样签名的 rerank(query, documents, top_k) 并在 get_reranker() 中返回即可。
"""
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document

import config
from observability.metrics import timer


@dataclass
class RerankResult:
    document: Document
    rerank_score: float  # [0, 1]，越大越相关


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[RerankResult]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, cache_dir: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[RerankResult]:
        if not documents:
            return []
        logits = list(self.model.rerank(query, [doc.page_content for doc in documents]))
        results = [
            RerankResult(document=doc, rerank_score=1.0 / (1.0 + math.exp(-logit)))
            for doc, logit in zip(documents, logits)
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return CrossEncoderReranker(config.RERANKER_MODEL, config.FASTEMBED_CACHE_DIR)


def rerank(query: str, documents: list[Document], top_k: int = config.RERANK_TOP_K) -> list[RerankResult]:
    with timer("rerank"):
        return get_reranker().rerank(query, documents, top_k)
