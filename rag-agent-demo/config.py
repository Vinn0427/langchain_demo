"""
集中配置：模型、Qdrant、检索参数、Gate 阈值、重试次数都只在这里定义，可被 .env 覆盖。

rag/、agent/、scripts/、eval/ 都从这里读取，代码里不再散落任何检索 / Gate 常量。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# ---- 模型（OpenAI-compatible）----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or OPENAI_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or OPENAI_BASE_URL
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 10)  # 部分兼容服务商单次最多 10 条

# ---- Qdrant ----
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_demo_v3")

# ---- Indexing ----
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

# ---- Retrieval（知识库只有约 20 个 chunk，默认值按这个规模缩小；语料变大后应相应调大）----
DENSE_TOP_K = _int("DENSE_TOP_K", 10)    # Dense 召回条数
SPARSE_TOP_K = _int("SPARSE_TOP_K", 10)  # Sparse / BM25 召回条数
RRF_K = _int("RRF_K", 60)                # RRF 平滑常数：score(d) += 1 / (RRF_K + rank)
RRF_TOP_K = _int("RRF_TOP_K", 10)        # RRF 融合后送入 Reranker 的候选数
RERANK_TOP_K = _int("RERANK_TOP_K", 5)   # Reranker 之后保留的条数

# ---- BM25 ----
BM25_K1 = _float("BM25_K1", 1.2)
BM25_B = _float("BM25_B", 0.75)

# ---- Reranker（本地 Cross-Encoder，fastembed / ONNX Runtime）----
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
FASTEMBED_CACHE_DIR = os.path.expanduser(os.getenv("FASTEMBED_CACHE_DIR", "~/.cache/fastembed"))

# ---- Retrieval Gate（作用于 rerank_score ∈ [0, 1]）----
# 注意：这是 Demo 的初始值，只在本仓库的小知识库 + bge-reranker-base 上观察过分数分布
# （明确问题 top1 ≥ 0.98；模糊问题 0.17~0.86；跑偏 / 不存在的问题 ≤ 0.12），不具备普适性。
# 正式系统应基于 Eval Dataset 的分数分布调参。
HIGH_CONFIDENCE_THRESHOLD = _float("HIGH_CONFIDENCE_THRESHOLD", 0.9)
LOW_CONFIDENCE_THRESHOLD = _float("LOW_CONFIDENCE_THRESHOLD", 0.15)

# ---- Stop Condition ----
MAX_RETRY = _int("MAX_RETRY", 1)
