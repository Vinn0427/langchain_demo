"""
V4：运行版本信息。写进 Eval Report 和 Trace，出现 Regression 时能知道"当时跑的是什么"。
不包含任何密钥 / Base URL。
"""
from importlib.metadata import PackageNotFoundError, version

import config

_PACKAGES = ("langchain-core", "langchain-openai", "langgraph", "openai", "qdrant-client", "fastembed", "jieba")


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "n/a"


def runtime_versions(collection: str = None) -> dict:
    from rag.store import resolve_alias

    try:
        serving = resolve_alias(config.QDRANT_ALIAS)
    except Exception:  # noqa: BLE001
        serving = None
    return {
        "chat_model": config.CHAT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "reranker_model": config.RERANKER_MODEL,
        "bm25": {"version": config.BM25_VERSION, "k1": config.BM25_K1, "b": config.BM25_B},
        "chunk_strategy": config.CHUNK_STRATEGY,
        "chunk_strategy_version": config.CHUNK_STRATEGY_VERSION,
        "retrieval_text_version": config.RETRIEVAL_TEXT_VERSION,
        "routing_mode": config.ROUTING_MODE,
        "structured_output_method": config.STRUCTURED_OUTPUT_METHOD,
        "prompt_versions": {
            "agent": config.AGENT_PROMPT_VERSION,
            "query_understanding": config.QUERY_UNDERSTANDING_PROMPT_VERSION,
            "grader": config.GRADER_PROMPT_VERSION,
            "rewrite": config.REWRITE_PROMPT_VERSION,
            "tool_schema": config.TOOL_SCHEMA_VERSION,
        },
        "gate": {"high": config.HIGH_CONFIDENCE_THRESHOLD, "low": config.LOW_CONFIDENCE_THRESHOLD},
        "timeouts": {
            "query_understanding": config.QUERY_UNDERSTANDING_TIMEOUT, "agent_decision": config.AGENT_DECISION_TIMEOUT,
            "grader": config.GRADER_TIMEOUT, "rewrite": config.REWRITE_TIMEOUT,
            "final_answer": config.FINAL_ANSWER_TIMEOUT, "embedding": config.EMBEDDING_TIMEOUT,
            "tool": config.TOOL_TIMEOUT, "request_deadline": config.REQUEST_DEADLINE,
            "llm_max_attempts": config.LLM_MAX_ATTEMPTS,
        },
        "collection": collection or serving,
        "alias": config.QDRANT_ALIAS,
        "packages": {p: _pkg(p) for p in _PACKAGES},
    }
