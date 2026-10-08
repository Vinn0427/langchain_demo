"""
Sparse Retrieval（关键词召回）：BM25，存储和检索都交给 Qdrant 的 sparse vector。

BM25(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 - b + b·|d|/avgdl) )
             └ Qdrant ┘   └──────────── 本模块离线算好，作为 sparse vector 的 value ────────────┘

- 文档侧（离线）：jieba 分词 → 每个词的 BM25 TF 部分 → sparse vector {token_id: weight}
- 查询侧（在线）：jieba 分词 → 每个词权重 1.0 → Qdrant 点积，再乘以 Qdrant 端维护的 IDF（modifier=IDF）
- token_id：对词做稳定哈希（md5 前 32 位），不需要维护词表文件；新词在查询时自然匹配不到任何文档。

avgdl 只在离线 Indexing 时需要，在线查询不需要重新扫描文档。
"""
import hashlib
import logging
import re
from collections import Counter

import jieba
from langchain_core.documents import Document
from qdrant_client import models

import config
from observability.metrics import timer
from rag.store import SPARSE_VECTOR, get_client, point_to_document

jieba.setLogLevel(logging.WARNING)

# 只过滤最常见的虚词 / 疑问词；其余高频词交给 IDF 自然降权
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "就", "都", "也", "要", "会", "吗", "呢", "吧",
    "什么", "哪个", "哪些", "怎么", "如何", "多少", "一般", "我们", "根据", "知识库", "请问",
}
_IS_WORD = re.compile(r"[\w\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in jieba.lcut_for_search(text.lower()):
        token = token.strip()
        if token and _IS_WORD.search(token) and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def token_id(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def _to_sparse(weights: dict[int, float]) -> models.SparseVector:
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])


def bm25_document_vector(tokens: list[str], avgdl: float) -> models.SparseVector:
    k1, b = config.BM25_K1, config.BM25_B
    doc_len = len(tokens)
    weights: dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        weight = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        tid = token_id(token)
        weights[tid] = weights.get(tid, 0.0) + weight  # 极少数哈希冲突时合并
    return _to_sparse(weights)


def bm25_query_vector(tokens: list[str]) -> models.SparseVector:
    return _to_sparse({token_id(token): 1.0 for token in set(tokens)})


def sparse_search(query: str, top_k: int = config.SPARSE_TOP_K) -> list[Document]:
    # sparse_retrieval 计时包含：查询分词 + Qdrant sparse 检索
    with timer("sparse_retrieval"):
        tokens = tokenize(query)
        if not tokens:
            return []
        response = get_client().query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=bm25_query_vector(tokens),
            using=SPARSE_VECTOR,
            limit=top_k,
            with_payload=True,
        )
    documents = []
    for rank, point in enumerate(response.points, start=1):
        doc = point_to_document(point)
        doc.metadata.update(sparse_score=point.score, sparse_rank=rank)
        documents.append(doc)
    return documents
