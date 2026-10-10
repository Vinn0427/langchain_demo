"""
Sparse Retrieval（关键词召回）：BM25，存储和检索都交给 Qdrant 的 sparse vector。

BM25(q, d) = Σ_{t ∈ q}  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 - b + b·|d|/avgdl) )
             └ Qdrant ┘   └──────────── 本模块离线算好，作为 sparse vector 的 value ────────────┘

- 文档侧（离线）：jieba 分词 → 每个词的 BM25 TF 部分 → sparse vector {token_id: weight}
- 查询侧（在线）：jieba 分词 → 每个词权重 1.0 → Qdrant 点积，再乘以 Qdrant 端维护的 IDF（modifier=IDF）
- token_id：对词做稳定哈希（md5 前 32 位），不需要维护词表文件；新词在查询时自然匹配不到任何文档。

======================== V4：增量更新时的全局依赖分析 ========================
| 组成部分             | 是否依赖整个语料 | 增量新增 / 删除文档时                                              |
|----------------------|------------------|--------------------------------------------------------------------|
| jieba 分词 / 停用词  | 否（只依赖词典） | 不受影响；词典或停用词变化 → BM25_VERSION 变化 → fingerprint 变化   |
| token_id（md5）      | 否               | 不受影响                                                           |
| TF 饱和（k1）        | 否（只看本文档） | 不受影响                                                           |
| 长度归一化（b·|d|/avgdl） | **是：avgdl** | V3 每次全量重建时重新计算 avgdl。若增量时用新 avgdl 只重算新文档， |
|                      |                  | 新旧文档的 value 就用了不同的 avgdl —— 表面增量、数学上不一致。      |
| IDF                  | **是：N、df(t)** | Qdrant 在查询时根据 collection 现状计算，新增文档立刻生效；          |
|                      |                  | 但实测：删除 / 覆盖写的旧 point 在 segment 被 vacuum 之前仍计入 IDF |
|                      |                  | 统计（默认 vacuum_min_vector_number=1000，小 collection 永远不清理）|

V4 的选择（方案 A：保留自研实现，明确什么时候需要重建）：
1. avgdl 在 collection 创建（全量构建）时冻结，写入 manifest；之后所有增量写入都用同一个冻结值。
   这样每个文档的 sparse value 只依赖"它自己 + 固定常数"，增量结果与"用同一 avgdl 全量重建"逐位相同。
   （Qdrant/FastEmbed 的标准 Bm25 也是同样思路：avg_len 是固定参数，不随语料变化。）
2. IDF 继续交给 Qdrant；collection 创建时设置 deleted_threshold≈0 / vacuum_min_vector_number=1，
   让删除后的 IDF 统计及时收敛；Indexing 结束时等待 collection 回到 green 再返回。
3. 什么时候必须重建 Sparse Index（Blue-Green 全量重建）：
   - tokenizer / 停用词 / token_id / k1 / b 变化（→ BM25_VERSION / fingerprint 变化，自动走 Blue-Green）
   - 实际 avgdl 偏离冻结值超过 BM25_AVGDL_DRIFT_THRESHOLD（每次增量构建都会报告 drift，超过阈值提示重建）
"""
import hashlib
import logging
import re
from collections import Counter
from typing import Optional

import jieba
from langchain_core.documents import Document
from qdrant_client import models

import config
from observability.metrics import timer
from rag.store import SPARSE_VECTOR, get_client, point_to_document, serving_collection

jieba.setLogLevel(logging.WARNING)

# 只过滤最常见的虚词 / 疑问词；其余高频词交给 IDF 自然降权
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "就", "都", "也", "要", "会", "吗", "呢", "吧",
    "什么", "哪个", "哪些", "怎么", "如何", "多少", "一般", "我们", "根据", "知识库", "请问",
}
_IS_WORD = re.compile(r"[\w\u4e00-\u9fff]")


def tokenizer_signature() -> dict:
    return {
        "tokenizer": f"jieba-{jieba.__version__}-lcut_for_search-lower",
        "stopwords_md5": hashlib.md5("|".join(sorted(STOPWORDS)).encode()).hexdigest()[:8],
        "token_id": "md5-32bit",
    }


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


def sparse_search(query: str, top_k: int = config.SPARSE_TOP_K, collection: Optional[str] = None) -> list[Document]:
    # sparse_retrieval 计时包含：查询分词 + Qdrant sparse 检索
    with timer("sparse_retrieval"):
        tokens = tokenize(query)
        if not tokens:
            return []
        response = get_client().query_points(
            collection_name=serving_collection(collection),
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
