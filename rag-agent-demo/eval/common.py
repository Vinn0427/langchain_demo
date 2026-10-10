"""Eval 公共工具：指标计算、延迟统计、表格打印、报告落盘。"""
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
REPORT_DIR = EVAL_DIR / "reports"
KS = (1, 3, 5)
STAGES = ("dense", "sparse", "hybrid_rrf", "hybrid_rerank")


def load(name: str) -> list[dict]:
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def save_report(name: str, data: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def stats(values: Iterable[Optional[float]]) -> dict:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "avg": None, "p50": None, "p95": None, "max": None}
    return {
        "n": int(arr.size), "avg": round(float(arr.mean()), 1), "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1), "max": round(float(arr.max()), 1),
    }


def score_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=float)
    return {"n": int(arr.size), "min": round(float(arr.min()), 3), "p50": round(float(np.percentile(arr, 50)), 3),
            "max": round(float(arr.max()), 3)}


def print_latency_table(title: str, rows: dict[str, list]) -> dict:
    print(f"\n---------- {title} (ms) ----------")
    print(f"{'stage':<26}{'avg':>10}{'p50':>10}{'p95':>10}{'max':>10}{'n':>6}")
    out = {}
    for name, values in rows.items():
        s = stats(values)
        out[name] = s
        if s["n"] == 0:
            print(f"{name:<26}{'n/a':>10}")
        else:
            print(f"{name:<26}{s['avg']:>10.1f}{s['p50']:>10.1f}{s['p95']:>10.1f}{s['max']:>10.1f}{s['n']:>6}")
    return out


def rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def evaluate_retrieval(dataset: list[dict], collection: Optional[str] = None, with_latency: bool = True) -> dict:
    """对每条 query 调 rag.pipeline.retrieve()，分别评估 dense / sparse / hybrid_rrf / hybrid_rerank 四个阶段。"""
    from observability.metrics import start_request
    from rag.pipeline import retrieve

    labeled = [d for d in dataset if d["relevant_chunk_ids"]]
    scores = {s: {**{f"recall@{k}": [] for k in KS}, "mrr": []} for s in STAGES}
    by_category: dict[str, dict] = {}
    latency = {k: [] for k in ("query_embedding", "dense_retrieval", "sparse_retrieval", "rrf_fusion",
                               "hybrid_retrieval", "rerank", "retrieval_total")}
    misses, top1_positive, top1_negative = [], [], []

    for item in dataset:
        metrics = start_request()
        result = retrieve(item["query"], collection=collection)
        metrics.finish()
        top1 = result.reranked[0].metadata["rerank_score"] if result.reranked else 0.0
        if not item["relevant_chunk_ids"]:
            top1_negative.append(top1)
            continue
        top1_positive.append(top1)
        relevant = set(item["relevant_chunk_ids"])
        ranked = {
            "dense": [d.metadata["chunk_id"] for d in result.dense],
            "sparse": [d.metadata["chunk_id"] for d in result.sparse],
            "hybrid_rrf": [d.metadata["chunk_id"] for d in result.fused],
            "hybrid_rerank": [d.metadata["chunk_id"] for d in result.reranked],
        }
        for stage, ids in ranked.items():
            for k in KS:
                scores[stage][f"recall@{k}"].append(recall_at_k(ids, relevant, k))
            scores[stage]["mrr"].append(reciprocal_rank(ids, relevant))
        cat = by_category.setdefault(item.get("category", "-"), {s: {"recall@5": [], "mrr": []} for s in STAGES})
        for stage, ids in ranked.items():
            cat[stage]["recall@5"].append(recall_at_k(ids, relevant, 5))
            cat[stage]["mrr"].append(reciprocal_rank(ids, relevant))
        if not set(ranked["hybrid_rerank"][:1]) & relevant:
            misses.append({"query": item["query"], "expected": sorted(relevant), "got_top3": ranked["hybrid_rerank"][:3]})
        if with_latency:
            for stage in latency:
                latency[stage].append(metrics.total_ms if stage == "retrieval_total" else metrics.stage_ms(stage))

    summary = {s: {m: round(float(np.mean(v)), 4) for m, v in scores[s].items()} for s in STAGES}
    categories = {
        c: {"n": len(v["hybrid_rerank"]["mrr"]),
            **{s: {m: round(float(np.mean(x)), 3) for m, x in v[s].items()} for s in STAGES}}
        for c, v in by_category.items()
    }
    return {
        "n_labeled": len(labeled), "n_no_answer": len(dataset) - len(labeled),
        "summary": summary, "by_category": categories, "misses": misses,
        "rerank_top1": {"positive": score_stats(top1_positive),
                        "no_answer": score_stats(top1_negative),
                        "positive_raw": [round(x, 3) for x in top1_positive],
                        "no_answer_raw": [round(x, 3) for x in top1_negative]},
        "latency": {k: stats(v) for k, v in latency.items()} if with_latency else None,
    }


def print_retrieval_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    header = "".join(f"{f'Recall@{k}':>11}" for k in KS) + f"{'MRR':>9}"
    print(f"{'stage':<16}{header}")
    for stage in STAGES:
        row = "".join(f"{summary[stage][f'recall@{k}']:>11.3f}" for k in KS)
        print(f"{stage:<16}{row}{summary[stage]['mrr']:>9.3f}")


# =====================================================================
# 端到端 Agent Case：Query / Tool / No-Answer / Latency Eval 共用
# =====================================================================
INTENT_BY_TOOL = {
    "search_knowledge_base": "KNOWLEDGE_SEARCH", "get_document": "DOCUMENT_LOOKUP", "get_chunk": "CHUNK_LOOKUP",
    "list_documents": "DOCUMENT_LIST", "get_index_status": "INDEX_STATUS",
}


def infer_intent(tools_requested: list[str]) -> str:
    """v3_agent 模式没有显式 Intent：由 Agent 实际选择的 Tool 反推（没有 Tool = DIRECT，多类 Tool = MULTI_TOOL）。"""
    distinct = list(dict.fromkeys(tools_requested))
    if not distinct:
        return "DIRECT"
    if len(distinct) > 1:
        return "MULTI_TOOL"
    return INTENT_BY_TOOL.get(distinct[0], "UNKNOWN")


def norm(text: str) -> str:
    return "".join(str(text).split()).lower()


def contains_all(text: str, needles: list[str]) -> bool:
    """needle 支持 "a|b" 表示任一写法即可；比较时去掉空白、忽略大小写。"""
    t = norm(text)
    return all(any(norm(alt) in t for alt in needle.split("|")) for needle in needles)


def run_agent_case(query: str, history: Optional[list] = None, mode: Optional[str] = None) -> dict:
    import contextlib
    import io
    import re

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from agent.nodes import SOURCES_HEADER
    from main import ask

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        state, metrics = ask(query, [tuple(h) for h in (history or [])], mode, show=False)
    messages = state["messages"]
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[start:]
    tool_outputs = "\n".join(m.content for m in turn if isinstance(m, ToolMessage))
    last = messages[-1]
    answer = last.content if isinstance(last, AIMessage) else ""
    body, _, sources_block = answer.partition(SOURCES_HEADER)
    printed_sources = re.findall(r"^- (\S+)", sources_block, flags=re.M)
    tool_log = state.get("tool_log") or []
    tool_source_ids = {s for e in tool_log for s in (e.get("source_ids") or [])}
    id_re = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*_[0-9]{3})(?![A-Za-z0-9_])")
    answer_ids = set(id_re.findall(body))
    hallucinated = sorted(i for i in answer_ids if i not in tool_outputs and i not in query
                          and not any(i in u or i in a for u, a in (history or [])))
    requested = [e["name"] for e in tool_log]
    marks = metrics.marks
    tool_selection_ms = ((marks["first_tool_start"] - marks["request_start"]) * 1000
                         if "first_tool_start" in marks else None)
    return {
        "query": query, "mode": mode,
        "intent": state.get("intent") if (mode or "") == "query_understanding" else infer_intent(requested),
        "qu_status": state.get("qu_status"), "standalone_query": state.get("standalone_query"),
        "candidate_tools": state.get("candidate_tools"), "qu_violations": state.get("qu_violations"),
        "tools_requested": requested, "tool_log": tool_log,
        "answer": answer, "answer_body": body, "final_answer_kind": state.get("final_answer_kind"),
        "printed_sources": printed_sources, "sources_not_from_tools": sorted(set(printed_sources) - tool_source_ids),
        "hallucinated_ids_in_answer": hallucinated,
        "latency": {
            "total": metrics.total_ms, "e2e_ttft": metrics.e2e_ttft_ms, "model_ttft": metrics.model_ttft_ms,
            "query_understanding": metrics.stage_ms("query_understanding"),
            "agent_decision": metrics.stage_ms("agent_decision"),
            "first_agent_decision": (metrics.stages["agent_decision"][0] * 1000 if "agent_decision" in metrics.stages else None),
            "tool_selection": tool_selection_ms,
        },
        "stages": metrics.summary(),
        "llm_calls": [r.to_json() for r in metrics.llm_calls],
        "events": list(metrics.events),
        "request_id": metrics.request_id,
        "log_tail": buf.getvalue()[-3000:],
    }


def stability_metrics(cases: list[dict]) -> dict:
    attempts = [c for case in cases for c in case["llm_calls"] if c["status"] != "deadline_exceeded"]
    by_call: dict[str, list] = {}
    for a in attempts:
        by_call.setdefault(a["call_id"], []).append(a)
    structured = [e for case in cases for e in case["events"] if e["event"] == "structured_parse"]
    rag_cases = [c for c in cases if any(e["name"] == "search_knowledge_base" for e in c["tool_log"])]
    tool_calls = [e for c in cases for e in c["tool_log"]]
    status_counts: dict[str, int] = {}
    for a in attempts:
        status_counts[a["status"]] = status_counts.get(a["status"], 0) + 1
    return {
        "requests": len(cases),
        "llm_attempts": len(attempts),
        "llm_logical_calls": len(by_call),
        "attempt_status_counts": status_counts,
        "timeout_rate": rate(sum(a["status"] == "timeout" for a in attempts), len(attempts)),
        "retry_rate": rate(sum(len(v) > 1 for v in by_call.values()), len(by_call)),
        "structured_calls": len(structured),
        "structured_parse_failure_rate": rate(sum(e["status"] != "ok" for e in structured), len(structured)),
        "structured_unrecovered_rate": rate(sum(e["status"] in ("parse_error", "validation_error", "llm_error")
                                                for e in structured), len(structured)),
        "structured_reask_count": sum(bool(e.get("reask")) for e in structured),
        "rag_requests": len(rag_cases),
        "query_rewrite_rate": rate(sum("query_rewrite" in c["stages"] for c in rag_cases), len(rag_cases)),
        "grader_trigger_rate": rate(sum("llm_grader" in c["stages"] for c in rag_cases), len(rag_cases)),
        "tool_calls": len(tool_calls),
        "tool_error_rate": rate(sum(e["error_code"] not in (None, "NOT_FOUND") for e in tool_calls), len(tool_calls)),
        "duplicate_tool_call_rate": rate(sum(e["error_code"] == "DUPLICATE_TOOL_CALL" for e in tool_calls), len(tool_calls)),
        "no_answer_rate": rate(sum(c["final_answer_kind"] == "no_answer" for c in cases), len(cases)),
        "llm_error_answers": sum(c["final_answer_kind"] == "llm_error" for c in cases),
    }


def print_stability(title: str, s: dict) -> None:
    print(f"\n---------- {title} ----------")
    for k in ("requests", "llm_attempts", "llm_logical_calls", "attempt_status_counts", "timeout_rate", "retry_rate",
              "structured_calls", "structured_parse_failure_rate", "structured_unrecovered_rate", "structured_reask_count",
              "rag_requests", "query_rewrite_rate", "grader_trigger_rate", "tool_calls", "tool_error_rate",
              "duplicate_tool_call_rate", "no_answer_rate", "llm_error_answers"):
        print(f"{k:<32}{s[k]}")
