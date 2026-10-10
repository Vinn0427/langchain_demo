"""
集中配置：模型、Qdrant、检索参数、Gate 阈值、重试次数、Timeout / Deadline、Chunk 策略、Prompt 版本都只在这里定义，
可被 .env 覆盖。rag/、agent/、query/、tools/、llm/、scripts/、eval/ 都从这里读取。

注意：本文件中所有 timeout / 阈值的默认值都只来自本仓库在当前 Provider（阿里云百炼兼容模式 + qwen flash 系列）
上的实测分布，不具备普适性。换 Provider / 模型 / 语料后需要重新测。
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


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---- 模型（OpenAI-compatible）----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = _int("EMBEDDING_DIMENSION", 1024)  # 写进 pipeline_fingerprint；实际维度与它不一致时 Indexing 直接报错
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or OPENAI_API_KEY
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or OPENAI_BASE_URL
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 10)  # 部分兼容服务商单次最多 10 条

# ---- V4：分阶段 Timeout（秒）----
# V3 的教训：所有调用共用一个 timeout，且重试由 openai SDK 在内部静默完成（max_retries），
# 一次 60s 卡顿 + 一次静默重试在日志里只表现为"这次请求很慢"，无法定位到 phase / attempt。
# V4：SDK 内部重试关闭（max_retries=0），由 llm/client.py 自己做重试，每个 attempt 都单独记录 trace。
#
# 对 openai SDK / httpx 而言，timeout 是"每一次读操作"的上限：
#   - 非流式请求：≈ 等完整响应的时间（模型生成完才返回）
#   - 流式请求  ：≈ 等第一个 chunk（TTFT）以及任意两个 chunk 之间的间隔
# 初始值 = 本 Provider 实测 p95 的约 5~8 倍（V3 实测：agent_decision p95≈1.4s，grader 单次≈0.8s，
# rewrite≈1.3s，model_ttft p95≈0.9s）。不具备普适性。
QUERY_UNDERSTANDING_TIMEOUT = _float("QUERY_UNDERSTANDING_TIMEOUT", 12)  # qwen3.8-flash 实测单次 2.0~7.9s
AGENT_DECISION_TIMEOUT = _float("AGENT_DECISION_TIMEOUT", 12)
GRADER_TIMEOUT = _float("GRADER_TIMEOUT", 5)
REWRITE_TIMEOUT = _float("REWRITE_TIMEOUT", 5)
FINAL_ANSWER_TIMEOUT = _float("FINAL_ANSWER_TIMEOUT", 10)
EMBEDDING_TIMEOUT = _float("EMBEDDING_TIMEOUT", 5)
TOOL_TIMEOUT = _float("TOOL_TIMEOUT", 5)            # 非 RAG Tool（Qdrant 精确查询 / 元数据）的执行上限
QDRANT_TIMEOUT = _int("QDRANT_TIMEOUT", 5)

# 兼容 V3 的变量名：只用于 Indexing 的批量 Embedding
API_TIMEOUT = _float("API_TIMEOUT", 5)

# ---- V4：Retry Policy（只对 timeout / 连接异常 / 429 / 5xx 重试）----
# parse_error / validation_error / Grader=false / 检索结果为空 都不会触发新的 LLM 请求。
LLM_MAX_ATTEMPTS = _int("LLM_MAX_ATTEMPTS", 2)          # 每次逻辑调用最多 2 次 attempt（= 最多重试 1 次）
LLM_RETRY_BACKOFF = _float("LLM_RETRY_BACKOFF", 0.3)    # 第 n 次重试前等待 backoff * 2^(n-1) 秒
EMBEDDING_MAX_ATTEMPTS = _int("EMBEDDING_MAX_ATTEMPTS", 3)
# Structured Output：同一个 raw response 容错解析仍失败时，是否允许"再问一次模型"（需同时满足预算）
STRUCTURED_MAX_REASK = _int("STRUCTURED_MAX_REASK", 1)
STRUCTURED_OUTPUT_METHOD = os.getenv("STRUCTURED_OUTPUT_METHOD", "json_schema")  # json_schema / function_calling

# ---- V4：Request Deadline / Latency Budget（秒）----
REQUEST_DEADLINE = _float("REQUEST_DEADLINE", 30)
FINAL_ANSWER_RESERVE = _float("FINAL_ANSWER_RESERVE", 6)  # 给最终回答预留的预算：中间阶段不能吃掉这部分
RETRIEVAL_RESERVE = _float("RETRIEVAL_RESERVE", 2)        # 一次 Hybrid Retrieval + Rerank 的预留
MIN_ATTEMPT_TIMEOUT = _float("MIN_ATTEMPT_TIMEOUT", 1.0)  # 剩余预算小于它时不再发起新的 attempt
# Grader 因预算不足被跳过时的明确 fallback：MEDIUM 文档中 rerank_score ≥ 该值的直接放行，其余丢弃
GRADER_SKIP_PASS_THRESHOLD = _float("GRADER_SKIP_PASS_THRESHOLD", 0.5)
GRADER_CONCURRENCY = _int("GRADER_CONCURRENCY", 4)

# ---- V4：故障注入（只用于复现 / 验证 timeout 定位；默认关闭）----
# 格式：phase:attempt:mode[,phase:attempt:mode]   mode = stall / 429 / 500 / connect
# 例：LLM_FAULT_INJECTION=final_answer:1:stall  → final_answer 第 1 次 attempt 的 HTTP 请求挂起直到 read timeout
LLM_FAULT_INJECTION = os.getenv("LLM_FAULT_INJECTION", "")

# ---- Tracing ----
TRACE_LOG_PATH = PROJECT_ROOT / os.getenv("TRACE_LOG_PATH", "logs/trace.jsonl")
TRACE_PRINT = _bool("TRACE_PRINT", True)

# ---- Qdrant ----
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# V4：在线服务只读 alias；物理 collection 按 pipeline_fingerprint 命名（Blue-Green）。
# 注意：V3 的 QDRANT_COLLECTION=rag_demo_v3 不再被读取，旧 collection 原样保留，可以手动删除。
QDRANT_ALIAS = os.getenv("QDRANT_ALIAS", "rag_demo")
COLLECTION_PREFIX = os.getenv("COLLECTION_PREFIX", "rag_demo_v4")

# ---- Indexing ----
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
MANIFEST_PATH = PROJECT_ROOT / os.getenv("MANIFEST_PATH", "index_state/manifest.db")
# Blue-Green：新 collection 切换前必须通过的 Retrieval Eval 门槛（hybrid_rerank 指标）
EVAL_GATE_DATASET = PROJECT_ROOT / os.getenv("EVAL_GATE_DATASET", "eval/retrieval_dataset.json")
BLUE_GREEN_MIN_RECALL5 = _float("BLUE_GREEN_MIN_RECALL5", 0.85)
BLUE_GREEN_MAX_MRR_DROP = _float("BLUE_GREEN_MAX_MRR_DROP", 0.05)  # 相对当前 serving collection 允许的 MRR 下降

# ---- Chunking（V4）----
# A = raw chunk；B = 文档标题 + 当前小节标题 + raw；C = 文档标题 + 完整标题路径 + raw
# 默认值由 eval/run_chunk_eval.py 的实测结果决定（见 README V4 §Chunk Strategy）。
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "C")
CHUNK_STRATEGY_VERSION = "heading-v1"     # 切分算法版本：改了切分逻辑就要改这里
RETRIEVAL_TEXT_VERSION = "rt-v1"          # retrieval_text 拼接格式版本
CHUNK_SPLIT_LEVEL = _int("CHUNK_SPLIT_LEVEL", 3)    # 按 ## 和 ### 切分（# 是文档标题）
CHUNK_MAX_CHARS = _int("CHUNK_MAX_CHARS", 500)      # 与 data/rag.md 里的团队规范一致：每个 chunk 不超过 500 字
CHUNK_OVERLAP_CHARS = _int("CHUNK_OVERLAP_CHARS", 50)
CHUNK_TOO_SHORT_TOKENS = _int("CHUNK_TOO_SHORT_TOKENS", 32)   # Chunk Analysis 阈值（jieba token）
CHUNK_TOO_LONG_TOKENS = _int("CHUNK_TOO_LONG_TOKENS", 400)    # bge-reranker-base 上限 512 token，留余量

# ---- Retrieval（知识库只有约 30 个 chunk，默认值按这个规模缩小；语料变大后应相应调大）----
DENSE_TOP_K = _int("DENSE_TOP_K", 10)    # Dense 召回条数
SPARSE_TOP_K = _int("SPARSE_TOP_K", 10)  # Sparse / BM25 召回条数
RRF_K = _int("RRF_K", 60)                # RRF 平滑常数：score(d) += 1 / (RRF_K + rank)
RRF_TOP_K = _int("RRF_TOP_K", 10)        # RRF 融合后送入 Reranker 的候选数
RERANK_TOP_K = _int("RERANK_TOP_K", 5)   # Reranker 之后保留的条数

# ---- BM25 ----
BM25_K1 = _float("BM25_K1", 1.2)
BM25_B = _float("BM25_B", 0.75)
BM25_VERSION = "bm25-jieba-md5-v1"       # tokenizer / token_id / 停用词 / 公式实现的版本
# avgdl 在 collection 创建（全量构建）时冻结并写入 manifest；增量更新沿用冻结值，
# 实际 avgdl 偏离冻结值超过该比例时提示 Blue-Green 重建。
BM25_AVGDL_DRIFT_THRESHOLD = _float("BM25_AVGDL_DRIFT_THRESHOLD", 0.2)

# ---- Reranker（本地 Cross-Encoder，fastembed / ONNX Runtime）----
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
FASTEMBED_CACHE_DIR = os.path.expanduser(os.getenv("FASTEMBED_CACHE_DIR", "~/.cache/fastembed"))

# ---- Retrieval Gate（作用于 rerank_score ∈ [0, 1]）----
# Demo 初始值，只在本仓库的小知识库 + bge-reranker-base 上观察过分数分布，不具备普适性。
HIGH_CONFIDENCE_THRESHOLD = _float("HIGH_CONFIDENCE_THRESHOLD", 0.9)
LOW_CONFIDENCE_THRESHOLD = _float("LOW_CONFIDENCE_THRESHOLD", 0.15)

# ---- Stop Condition ----
MAX_RETRY = _int("MAX_RETRY", 1)

# ---- V4：Routing / Query Understanding ----
ROUTING_MODE = os.getenv("ROUTING_MODE", "query_understanding")  # v3_agent / query_understanding
QU_HISTORY_TURNS = _int("QU_HISTORY_TURNS", 3)          # 最近几轮对话交给 Query Understanding
QU_MIN_CONFIDENCE = _float("QU_MIN_CONFIDENCE", 0.6)    # 低于它不直接派发 Tool，交给 Tool Agent 决定
QU_DIRECT_DISPATCH = _bool("QU_DIRECT_DISPATCH", True)  # 单 Tool 且参数已确定时，跳过 Agent Decision 直接执行

# ---- V4：Multi-Tool Loop Safety ----
MAX_TOOL_CALLS_PER_REQUEST = _int("MAX_TOOL_CALLS_PER_REQUEST", 4)
DUPLICATE_TOOL_CALL_LIMIT = _int("DUPLICATE_TOOL_CALL_LIMIT", 1)  # 同一 Tool + 完全相同参数最多执行几次
AGENT_PARALLEL_TOOL_CALLS = _bool("AGENT_PARALLEL_TOOL_CALLS", True)
MAX_DOCUMENT_CHARS = _int("MAX_DOCUMENT_CHARS", 4000)  # get_document 返回给 LLM 的正文上限

# ---- V4：Prompt / Schema 版本（Regression 时用来定位变化来源）----
AGENT_PROMPT_VERSION = "agent-v4.1"
QUERY_UNDERSTANDING_PROMPT_VERSION = "qu-v1.1"
GRADER_PROMPT_VERSION = "grader-v2"
REWRITE_PROMPT_VERSION = "retrieval-rewrite-v1"
TOOL_SCHEMA_VERSION = "tools-v1"
