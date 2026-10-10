"""
Version 4 Graph 的节点与路由（在 V3 基础上扩展，V3 的 Corrective RAG 节点全部保留）。

    query_understanding_node  V4：Contextual Rewrite + Intent + Entity + Candidate Tools（一次 LLM 调用）；
                              单 Tool 且参数确定时直接生成 tool_call（跳过 Agent Decision）
    agent_node                Reason：动态绑定候选 Tool；决定直接回答 / 发起 tool_call；最终回答流式输出 + Sources
    tool_dispatch_node        V4：顺序执行 AIMessage 中的每一个 tool_call（Validation / Loop Safety / 结构化错误）；
                              遇到 search_knowledge_base 时转入 V3 的 RAG 子流程，结束后回来继续执行剩余 tool_call
    retrieve_node             V3：Hybrid Retrieval（Dense + BM25 → RRF）→ Rerank → Retrieval Gate
    grade_documents_node      V3：只判断 MEDIUM 置信度的文档（V4：并发 + Deadline 降级）
    rewrite_query_node        V4：Retrieval Rewrite（query/rewrite.py），只在检索失败后使用
    build_tool_message_node   V4：Context Dedup → ToolResult → ToolMessage（tool_call_id = pending_tool_call_id）

协议保证：AIMessage 中的每一个 tool_call_id（包括参数无法解析的 invalid_tool_calls、被 Loop Safety 拦截的调用）
最终都会有一条对应的 ToolMessage，之后才会再次进入 agent_node。
"""
import contextvars
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END

import config
from agent.state import AgentState
from llm.client import LLMCallError, StreamCall, attempt_timeout, get_chat_model, structured_call
from observability.metrics import current, mark, timer
from observability.tracing import record_event, record_tool
from query.rewrite import evidence_summary, retrieval_rewrite
from query.schema import GradeResult
from query.understanding import plan_dispatch, understand
from rag.gate import HIGH, MEDIUM, retrieval_gate
from rag.pipeline import dedup_documents, retrieve, section_label
from tools.registry import ALL_TOOLS, TOOLS, execute, langchain_tools, validate
from tools.schemas import ToolResult

MAX_RETRY = config.MAX_RETRY  # Stop Condition：最多 Retrieval Rewrite 几次

NO_ANSWER_MESSAGE = "知识库中没有足够信息支持回答。"
RETRIEVAL_FAILED_MESSAGE = "知识库检索暂时不可用，无法基于知识库回答，请稍后重试。"
LLM_ERROR_MESSAGE = "抱歉，模型服务暂时不可用（{phase}: {status}），请稍后重试。"
SOURCES_HEADER = "\n\nSources:\n"

AGENT_PROMPT = (
    "你是团队内部知识库助手。\n"
    "1. 问题涉及团队内部的系统、组件、规范、流程、运维、故障处理，或者涉及知识库文档 / chunk / 索引本身时，"
    "必须先调用工具获取信息，再基于工具结果回答；调用工具前不要输出任何文字。\n"
    "2. 与团队知识无关的通用问题（寒暄、翻译、数学、通用编程语言常识）直接回答，不要调用工具。\n"
    "3. 按每个工具说明中的 Use when / Do NOT use when 选择工具；只有一个问题确实需要多个工具时才调用多个。\n"
    "4. 基于工具结果回答时，只能使用工具返回的内容；工具结果不足以回答时，明确回答"
    f"「{NO_ANSWER_MESSAGE}」，不要用你自己的常识补充成知识库答案。\n"
    "5. 不要在回答中编写来源列表，也不要编造 chunk_id，系统会根据工具结果自动附上来源。"
)

GRADER_PROMPT = SystemMessage(
    "你是检索结果评估员。判断给定文档是否包含能够帮助回答用户问题的信息。"
    "文档中有与问题直接相关的内容则 relevant=true；只是主题沾边、但无法帮助回答该问题，则 relevant=false。"
    '只输出 JSON：{"relevant": true} 或 {"relevant": false}'
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _last_human_index(messages) -> int:
    return max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))


def _strip_sources(text: str) -> str:
    return text.split(SOURCES_HEADER)[0]


def history_pairs(messages) -> list[tuple[str, str]]:
    """本轮之前的 (用户问题, 助手最终回答)；只用于 Query Understanding 的上下文。"""
    pairs, pending = [], None
    for m in messages[:_last_human_index(messages)]:
        if isinstance(m, HumanMessage):
            pending = m.content
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls and pending is not None:
            pairs.append((pending, _strip_sources(m.content)))
            pending = None
    return pairs


def _render_tool_result(result: ToolResult) -> str:
    """ToolMessage.content：结构化 JSON，只包含 success / error_code / message / data，不含 traceback。"""
    payload = {k: v for k, v in result.model_dump(include={"success", "error_code", "message", "data"}).items()
               if v is not None}
    return json.dumps(payload, ensure_ascii=False)


def _sources_text(sources: list[dict]) -> str:
    if not sources:
        return ""
    return SOURCES_HEADER + "\n".join(f"- {s['id']} ({s['label']})" for s in sources)


def _merge_sources(state: AgentState, items: list[dict]) -> list[dict]:
    seen = {s["id"] for s in state.get("sources") or []}
    merged = list(state.get("sources") or [])
    for item in items:
        if item["id"] not in seen:
            merged.append(item)
            seen.add(item["id"])
    return merged


def _tool_entry(call: dict, result: ToolResult, latency_ms: float, args_valid: bool, **extra) -> dict:
    entry = {
        "name": call["name"], "tool_call_id": call["id"], "args": call.get("args"), "args_valid": args_valid,
        "success": result.success, "error_code": result.error_code, "retrieval_status": result.retrieval_status,
        "latency_ms": round(latency_ms, 1), "source_ids": result.source_ids, "chunk_ids": result.chunk_ids,
        "document_ids": result.document_ids,
        "dispatched_by": "query_understanding" if str(call["id"]).startswith("qu_") else "agent", **extra,
    }
    record_tool(entry)
    status = "ok" if result.success else result.error_code
    print(f"[Tool] {call['name']}({json.dumps(call.get('args'), ensure_ascii=False)}) → {status} "
          f"{latency_ms:.0f}ms sources={result.source_ids}")
    return entry


# ---------------------------------------------------------------------
# Query Understanding Node（ROUTING_MODE=query_understanding）
# ---------------------------------------------------------------------
def query_understanding_node(state: AgentState) -> dict:
    messages = state["messages"]
    raw = messages[_last_human_index(messages)].content
    with timer("query_understanding"):
        u = understand(raw, history_pairs(messages))
    print(f"[QU] status={u.status} intent={u.intent} confidence={u.confidence:.2f} "
          f"standalone_query={u.standalone_query!r} candidate_tools={u.candidate_tools}"
          + (f" violations={u.violations}" if u.violations else ""))
    update = {
        "standalone_query": u.standalone_query, "intent": u.intent, "entities": u.entities,
        "constraints": u.constraints, "document_id": u.document_id, "chunk_id": u.chunk_id, "topic": u.topic,
        "candidate_tools": u.candidate_tools, "qu_confidence": u.confidence, "qu_status": u.status,
        "qu_violations": u.violations, "dispatch_next": "agent",
    }
    call = plan_dispatch(u)
    if call:
        call_id = f"qu_{uuid.uuid4().hex[:12]}"
        tool_call = {"name": call["name"], "args": call["args"], "id": call_id, "type": "tool_call"}
        print(f"[QU] dispatch tool_call: {call['name']} args={call['args']}（跳过 Agent Decision）")
        update.update(messages=[AIMessage(content="", tool_calls=[tool_call])], tool_queue=[tool_call],
                      dispatch_next="tool_dispatch")
    return update


def route_start(state: AgentState) -> str:
    mode = state.get("routing_mode") or config.ROUTING_MODE
    return "query_understanding" if mode == "query_understanding" else "agent"


def route_after_qu(state: AgentState) -> str:
    return state["dispatch_next"]


# ---------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------
def _bound_tools(state: AgentState) -> tuple[list[str], Optional[str]]:
    """Candidate Tool Gating：本轮 Agent 能看到哪些 Tool。返回 (tool_names, 不绑定 Tool 的原因)。"""
    metrics = current()
    if state.get("tools_disabled"):
        return [], "loop_safety"
    if state.get("tool_calls_count", 0) >= config.MAX_TOOL_CALLS_PER_REQUEST:
        return [], "tool_budget"
    if metrics and metrics.remaining() < config.FINAL_ANSWER_RESERVE and state.get("tool_log"):
        record_event("tools_disabled", reason="deadline", remaining_s=round(metrics.remaining(), 2))
        return [], "deadline"
    if (state.get("routing_mode") or config.ROUTING_MODE) == "v3_agent":
        return list(ALL_TOOLS), None
    if state.get("intent") == "DIRECT":
        return [], "intent_direct"
    return list(state.get("candidate_tools") or ALL_TOOLS), None


def _deterministic_answer(tool_log: list[dict]) -> Optional[str]:
    """No-Answer Policy：本轮所有 Tool 都是知识库检索，且都没有找到 / 失败 → 不调用 LLM，返回固定答复。"""
    if not tool_log or any(e["success"] for e in tool_log):
        return None
    if not all(e["name"] == "search_knowledge_base" for e in tool_log):
        return None
    statuses = {e.get("retrieval_status") for e in tool_log}
    if statuses <= {"NOT_FOUND", None} and "NOT_FOUND" in statuses:
        return "no_answer"
    if "FAILED" in statuses:
        return "retrieval_failed"
    return None


def _emit_fixed_answer(text: str, kind: str) -> dict:
    writer = get_stream_writer()
    mark("final_llm_start")
    mark("final_first_token")
    writer({"answer_token": text})
    mark("final_llm_end")
    record_event("deterministic_answer", kind=kind)
    print(f"[Agent] deterministic answer ({kind}), no LLM call")
    return {"messages": [AIMessage(content=text)], "final_answer_kind": kind, "tool_queue": []}


def agent_node(state: AgentState) -> dict:
    metrics = current()
    turn_messages = state["messages"][_last_human_index(state["messages"]):]
    has_tool_results = any(isinstance(m, ToolMessage) for m in turn_messages)

    fixed = _deterministic_answer(state.get("tool_log") or [])
    if fixed:
        return _emit_fixed_answer(NO_ANSWER_MESSAGE if fixed == "no_answer" else RETRIEVAL_FAILED_MESSAGE, fixed)

    tool_names, no_tool_reason = _bound_tools(state)
    system = AGENT_PROMPT
    if (state.get("routing_mode") or config.ROUTING_MODE) == "query_understanding" and state.get("intent"):
        system += (f"\n\n【查询理解】standalone_query：{state.get('standalone_query')}；intent：{state.get('intent')}；"
                   f"entities：{state.get('entities')}")
    if not tool_names and has_tool_results:
        system += "\n\n现在请基于以上工具结果直接给出最终回答，不要再调用工具。"
        if no_tool_reason in ("loop_safety", "tool_budget", "deadline"):
            system += f"（工具调用已停止：{no_tool_reason}）"

    llm = get_chat_model()
    runnable = (llm.bind_tools(langchain_tools(tool_names), parallel_tool_calls=config.AGENT_PARALLEL_TOOL_CALLS)
                if tool_names else llm)
    # 看到第一个 chunk 之前不知道这次调用会产出 tool_call 还是回答：已有 Tool 结果时大概率是最终回答。
    # 首个 chunk 到达后 call.phase 会被改成真实 phase；在此之前失败的 attempt 按 planned phase 记录。
    planned = "agent_decision" if tool_names and not has_tool_results else "final_answer"
    timeout = config.AGENT_DECISION_TIMEOUT if planned == "agent_decision" else config.FINAL_ANSWER_TIMEOUT
    print(f"[Agent] LLM called (tools={tool_names or 'none'}{', reason=' + no_tool_reason if no_tool_reason else ''})")

    writer = get_stream_writer()
    call = StreamCall(planned, runnable, [SystemMessage(system)] + state["messages"], timeout=timeout)
    call_start = perf_counter()
    kind, response, failure = None, None, None
    try:
        for chunk in call:
            response = chunk if response is None else response + chunk
            if kind is None:
                if chunk.tool_call_chunks:
                    kind, call.phase = "tool_call", "agent_decision"
                elif chunk.content:
                    kind, call.phase = "answer", "final_answer"
                    mark("final_llm_start", call_start)
                    mark("final_first_token")
            if kind == "answer" and chunk.content:
                writer({"answer_token": chunk.content})
    except LLMCallError as exc:
        failure = exc
    call_end = perf_counter()

    if failure is not None and kind != "answer":
        metrics and metrics.add(planned, call_end - call_start)
        text = LLM_ERROR_MESSAGE.format(phase=failure.phase, status=failure.status)
        mark("final_llm_start", call_end)
        mark("final_first_token", call_end)
        writer({"answer_token": text})
        mark("final_llm_end")
        return {"messages": [AIMessage(content=text)], "final_answer_kind": "llm_error", "tool_queue": []}

    tool_calls = list(response.tool_calls or []) if response is not None else []
    invalid = list(getattr(response, "invalid_tool_calls", None) or []) if response is not None else []
    if failure is None and (tool_calls or invalid):
        if kind == "answer":
            # 模型先输出了一段文字、随后又发起 tool_call：这段文字不是最终回答，撤回 TTFT 标记
            writer({"retract": True})
            for event in ("final_llm_start", "final_first_token", "first_visible_token"):
                metrics and metrics.marks.pop(event, None)
            metrics and metrics.notes.setdefault("preamble", "模型在 tool_call 前输出了前导文本，已撤回，不计入 TTFT")
        metrics and metrics.add("agent_decision", call_end - call_start)
        queue = [{"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"} for c in tool_calls]
        queue += [{"name": c.get("name") or "", "args": c.get("args"), "id": c["id"], "invalid": True,
                   "error": c.get("error")} for c in invalid]
        for c in queue:
            print(f"[Agent] tool_call: {c['name']} args={c['args']}" + (" (INVALID JSON)" if c.get("invalid") else ""))
        message = AIMessage(content=response.content, tool_calls=tool_calls, invalid_tool_calls=invalid, id=response.id)
        return {"messages": [message], "tool_queue": queue}

    # ---- 最终回答 ----
    content = response.content if response is not None else ""
    answer_kind = "llm"
    if failure is not None:  # 流式输出中途中断：已展示的部分保留，追加说明
        suffix = f"\n（回答输出中断：{failure.status}）"
        writer({"answer_token": suffix})
        content += suffix
        answer_kind = "llm_interrupted"
    metrics and metrics.add("final_llm", call_end - call_start)
    mark("final_llm_end", call_end)
    if kind is None:  # 空回答
        mark("final_llm_start", call_start)
        mark("final_first_token", call_end)
    sources = state.get("sources") or []
    if sources and NO_ANSWER_MESSAGE not in content:
        text = _sources_text(sources)
        writer({"answer_token": text})
        content += text
    if metrics is not None and response is not None:
        metrics.final_message_id = response.id
    return {"messages": [AIMessage(content=content, id=response.id if response is not None else None)],
            "final_answer_kind": answer_kind, "tool_queue": []}


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None) or getattr(last, "invalid_tool_calls", None):
        return "tool_dispatch"
    return END


# ---------------------------------------------------------------------
# Tool Dispatch Node：顺序执行 tool_queue
# ---------------------------------------------------------------------
def _signature(call: dict) -> str:
    return f"{call['name']}:{json.dumps(call.get('args'), sort_keys=True, ensure_ascii=False)}"


def tool_dispatch_node(state: AgentState) -> dict:
    queue = list(state.get("tool_queue") or [])
    count = state.get("tool_calls_count", 0)
    signatures = list(state.get("tool_call_signatures") or [])
    disabled = bool(state.get("tools_disabled"))
    messages, log, used = [], [], list(state.get("used_chunk_ids") or [])
    sources = list(state.get("sources") or [])
    searched_before = any(e["name"] == "search_knowledge_base" for e in state.get("tool_log") or [])
    metrics = current()

    def finish(call: dict, result: ToolResult, t0: float, args_valid: bool) -> None:
        latency = (perf_counter() - t0) * 1000
        if TOOLS.get(call["name"]) and TOOLS[call["name"]].kind != "rag" and args_valid:
            metrics and metrics.add("tool_execution", latency / 1000)
        messages.append(ToolMessage(content=_render_tool_result(result), tool_call_id=call["id"], name=call["name"],
                                    artifact=result.model_dump(), status="success" if result.success else "error"))
        log.append(_tool_entry(call, result, latency, args_valid))

    while queue:
        call = queue.pop(0)
        t0 = perf_counter()
        if metrics is not None and "first_tool_start" not in metrics.marks:
            metrics.mark("first_tool_start", t0)  # request_start → first_tool_start = tool_selection_latency
        signature = _signature(call)
        if call.get("invalid"):
            count += 1
            finish(call, ToolResult.error("INVALID_ARGUMENT", f"tool_call 参数不是合法 JSON：{call.get('error')}"), t0, False)
            continue
        if count >= config.MAX_TOOL_CALLS_PER_REQUEST:
            disabled = True
            record_event("tool_budget_exceeded", tool=call["name"], limit=config.MAX_TOOL_CALLS_PER_REQUEST)
            finish(call, ToolResult.error("TOOL_BUDGET_EXCEEDED",
                                          f"本次请求的 Tool 调用已达上限 {config.MAX_TOOL_CALLS_PER_REQUEST}，请直接回答"), t0, True)
            continue
        if signatures.count(signature) >= config.DUPLICATE_TOOL_CALL_LIMIT:
            disabled = True
            record_event("duplicate_tool_call", tool=call["name"], args=call.get("args"))
            print(f"[LoopSafety] duplicate tool call blocked: {signature}")
            finish(call, ToolResult.error("DUPLICATE_TOOL_CALL",
                                          "与之前完全相同的 Tool 调用，结果已在前面的 ToolMessage 中，请直接基于已有结果回答"), t0, True)
            continue
        count += 1
        signatures.append(signature)
        args, error = validate(call["name"], call.get("args"))
        if error is not None:
            finish(call, error, t0, False)
            continue
        if TOOLS[call["name"]].kind == "rag":
            query, source = args.query, "tool_args"
            mode = state.get("routing_mode") or config.ROUTING_MODE
            if (mode == "query_understanding" and state.get("intent") == "KNOWLEDGE_SEARCH" and not searched_before
                    and state.get("standalone_query")):
                query, source = state["standalone_query"], "standalone_query"  # 第一次 RAG Retrieval 必须用 standalone_query
            return {
                "messages": messages, "tool_log": log, "tool_queue": queue, "tool_calls_count": count,
                "tool_call_signatures": signatures, "tools_disabled": disabled, "dispatch_next": "retrieve",
                "pending_tool_call_id": call["id"], "retrieval_query": query, "rag_original_query": query,
                "query_source": source, "retry_count": 0, "rewrite_failed": False, "retrieval_error": None,
                "rag_started_at": perf_counter(), "documents": [], "relevant_documents": [],
                "used_chunk_ids": used, "sources": sources,
            }
        result = execute(call["name"], args)
        if result.success:
            used += [c for c in result.chunk_ids if c not in used]
            items = []
            for sid in result.source_ids:
                data = result.data or {}
                label = data.get("section") or f"{data.get('title', '')} | {data.get('source', '')}"
                items.append({"id": sid, "source": data.get("source"), "label": label})
            sources = _merge_sources({"sources": sources}, items)
        finish(call, result, t0, True)

    return {"messages": messages, "tool_log": log, "tool_queue": [], "tool_calls_count": count,
            "tool_call_signatures": signatures, "tools_disabled": disabled, "dispatch_next": "agent",
            "used_chunk_ids": used, "sources": sources}


def route_after_dispatch(state: AgentState) -> str:
    return state["dispatch_next"]


# ---------------------------------------------------------------------
# Retrieve Node：query → Hybrid Retrieval → Rerank → Retrieval Gate
# ---------------------------------------------------------------------
def _short(docs, score_key: str, n: int = 5) -> str:
    return ", ".join(f"{d.metadata['chunk_id']}({d.metadata[score_key]:.3f})" for d in docs[:n]) or "-"


def retrieve_node(state: AgentState) -> dict:
    query = state["retrieval_query"]
    label = "rewritten retrieval query" if state.get("retry_count") else f"from {state.get('query_source')}"
    print(f"[Retrieve] query: {query}  ({label})")
    try:
        result = retrieve(query)
    except LLMCallError as exc:  # query embedding 重试后仍失败
        print(f"[Retrieve] FAILED: {exc}")
        return {"gate_level": "error", "retrieval_error": "TIMEOUT" if exc.status == "timeout" else "DEPENDENCY_ERROR",
                "documents": [], "relevant_documents": []}
    except Exception as exc:  # noqa: BLE001  Qdrant 等依赖异常
        print(f"[Retrieve] FAILED: {type(exc).__name__}")
        return {"gate_level": "error", "retrieval_error": "DEPENDENCY_ERROR", "documents": [], "relevant_documents": []}
    print(f"[Retrieve] dense  top: {_short(result.dense, 'dense_score')}")
    print(f"[Retrieve] sparse top: {_short(result.sparse, 'sparse_score')}")
    print(f"[Retrieve] rrf    top: {_short(result.fused, 'rrf_score')}")
    print(f"[Rerank]   top: {_short(result.reranked, 'rerank_score')}")

    decision = retrieval_gate(result.reranked)
    print(
        f"[Gate] {decision.level.upper()} (top rerank_score={decision.top_score:.3f}, "
        f"high>={config.HIGH_CONFIDENCE_THRESHOLD}, low<{config.LOW_CONFIDENCE_THRESHOLD})"
        + (f" → pass {len(decision.passed)} documents" if decision.level == HIGH else "")
        + (f" → {len(decision.to_grade)} documents to LLM Grader" if decision.level == MEDIUM else "")
    )
    update = {
        "gate_level": decision.level, "top_rerank_score": decision.top_score,
        "documents": decision.to_grade, "relevant_documents": decision.passed,
        "evidence_summary": evidence_summary(result.reranked), "retrieval_error": None,
    }
    if not state.get("retry_count"):  # 第一次检索（Rewrite 之前）的 Top-K：Eval 用来衡量 "RAG Query Quality"
        update["first_retrieval_ids"] = [d.metadata["chunk_id"] for d in result.reranked]
    return update


def _rewrite_or_stop(state: AgentState) -> str:
    if state["retry_count"] >= MAX_RETRY:
        print(f"[Retry] max retry reached ({state['retry_count']} / {MAX_RETRY}), stop rewriting")
        return "build_tool_message"
    if attempt_timeout(config.REWRITE_TIMEOUT, config.FINAL_ANSWER_RESERVE + config.RETRIEVAL_RESERVE) is None:
        metrics = current()
        record_event("rewrite_skipped", reason="deadline", remaining_s=round(metrics.remaining(), 2) if metrics else None)
        print("[Deadline] remaining budget too small, skip Retrieval Rewrite")
        return "build_tool_message"
    return "rewrite_query"


def route_after_gate(state: AgentState) -> str:
    if state["gate_level"] == "error":
        return "build_tool_message"
    if state["gate_level"] == HIGH:
        return "build_tool_message"   # 高置信度：跳过 LLM Grader
    if state["gate_level"] == MEDIUM:
        return "grade_documents"      # 边界情况：交给 LLM 做语义判断
    return _rewrite_or_stop(state)    # 低置信度：直接 Retrieval Rewrite


# ---------------------------------------------------------------------
# Retrieval Grader（只处理 MEDIUM）
# ---------------------------------------------------------------------
def _grade_one(question: str, doc) -> object:
    content = f"[{doc.metadata['chunk_id']}] {section_label(doc.metadata)}\n{doc.page_content}"
    return structured_call("grader", GradeResult, [GRADER_PROMPT, HumanMessage(f"用户问题：{question}\n\n文档：\n{content}")],
                           timeout=config.GRADER_TIMEOUT, reserve=config.FINAL_ANSWER_RESERVE)


def grade_documents_node(state: AgentState) -> dict:
    question = state.get("rag_original_query") or state["retrieval_query"]
    docs = state["documents"]
    threshold = config.GRADER_SKIP_PASS_THRESHOLD
    if attempt_timeout(config.GRADER_TIMEOUT, config.FINAL_ANSWER_RESERVE) is None:
        passed = [d for d in docs if d.metadata["rerank_score"] >= threshold]
        record_event("grader_skipped", reason="deadline", passed=[d.metadata["chunk_id"] for d in passed])
        print(f"[Deadline] skip LLM Grader → fallback: rerank_score >= {threshold} 的 {len(passed)} 个文档直接放行")
        return {"relevant_documents": passed}

    with timer("llm_grader"):
        with ThreadPoolExecutor(max_workers=config.GRADER_CONCURRENCY) as pool:
            futures = [pool.submit(contextvars.copy_context().run, _grade_one, question, d) for d in docs]
            results = [f.result() for f in futures]
    relevant_documents = []
    for doc, res in zip(docs, results):
        score = doc.metadata["rerank_score"]
        if res.value is None:  # LLM 失败或解析失败：明确 fallback，而不是再发一次请求
            relevant = score >= threshold
            note = f" (grader {res.status} → fallback score>={threshold})"
        else:
            relevant, note = res.value.relevant, ""
        print(f"[Grader] {doc.metadata['chunk_id']} (rerank_score={score:.3f}): "
              f"{'relevant' if relevant else 'irrelevant'}{note}")
        if relevant:
            relevant_documents.append(doc)
    print(f"[Grader] {len(relevant_documents)} relevant documents")
    return {"relevant_documents": relevant_documents}


def route_after_grading(state: AgentState) -> str:
    if state["relevant_documents"]:
        return "build_tool_message"
    return _rewrite_or_stop(state)


# ---------------------------------------------------------------------
# Retrieval Rewrite Node
# ---------------------------------------------------------------------
def rewrite_query_node(state: AgentState) -> dict:
    current_query = state["retrieval_query"]
    with timer("query_rewrite"):
        new_query, res = retrieval_rewrite(state.get("rag_original_query") or current_query, current_query,
                                           state.get("evidence_summary") or "", state["retry_count"])
    retry_count = state["retry_count"] + 1
    print(f"[Rewrite] original query: {current_query}")
    if not new_query or new_query == current_query:
        record_event("rewrite_failed", status=res.status, same_query=new_query == current_query)
        print(f"[Rewrite] no usable rewritten query (status={res.status}) → stop retry")
        return {"retry_count": retry_count, "rewrite_failed": True}
    print(f"[Rewrite] rewritten query: {new_query}  (llm_requests={res.llm_requests}, parse={res.status})")
    print(f"[Retry] {retry_count} / {MAX_RETRY}")
    return {"retrieval_query": new_query, "retry_count": retry_count, "rewrite_failed": False}


def route_after_rewrite(state: AgentState) -> str:
    return "build_tool_message" if state.get("rewrite_failed") else "retrieve"


# ---------------------------------------------------------------------
# Build ToolMessage Node：Context Dedup → ToolResult → ToolMessage
# ---------------------------------------------------------------------
def build_tool_message_node(state: AgentState) -> dict:
    call = {"name": "search_knowledge_base", "id": state["pending_tool_call_id"],
            "args": {"query": state.get("rag_original_query")}}
    t_start = state.get("rag_started_at") or perf_counter()
    used = list(state.get("used_chunk_ids") or [])
    sources = list(state.get("sources") or [])
    with timer("build_tool_message"):
        if state.get("gate_level") == "error":
            code = state.get("retrieval_error") or "DEPENDENCY_ERROR"
            result = ToolResult.error(code, f"知识库检索失败（{code}）", retrieval_status="FAILED")
        elif state["relevant_documents"]:
            kept, removed = dedup_documents(state["relevant_documents"], set(used))
            docs = [{"chunk_id": d.metadata["chunk_id"], "source": d.metadata["source"],
                     "section": section_label(d.metadata), "text": d.page_content} for d in kept]
            message = None
            if removed:
                message = f"Context Dedup：{removed} 与已提供的内容重复，已省略"
                record_event("context_dedup", removed=removed)
            result = ToolResult(
                success=True, data={"documents": docs}, message=message, retrieval_status="FOUND",
                source_ids=[d["chunk_id"] for d in docs], chunk_ids=[d["chunk_id"] for d in docs],
                document_ids=sorted({d.metadata["document_id"] for d in kept}),
            )
            used += [d["chunk_id"] for d in docs if d["chunk_id"] not in used]
            sources = _merge_sources({"sources": sources}, [
                {"id": d["chunk_id"], "source": d["source"], "label": f"{d['source']} | {d['section']}"} for d in docs])
        else:
            result = ToolResult.error("NOT_FOUND", NO_ANSWER_MESSAGE, retrieval_status="NOT_FOUND")
        tool_message = ToolMessage(content=_render_tool_result(result), tool_call_id=call["id"], name=call["name"],
                                   artifact=result.model_dump(), status="success" if result.success else "error")
    entry = _tool_entry(call, result, (perf_counter() - t_start) * 1000, True,
                        query_source=state.get("query_source"), final_query=state.get("retrieval_query"),
                        retry_count=state.get("retry_count", 0), gate_level=state.get("gate_level"),
                        top_rerank_score=state.get("top_rerank_score"),
                        first_retrieval_ids=state.get("first_retrieval_ids"))
    return {"messages": [tool_message], "tool_log": [entry], "retrieval_status": result.retrieval_status,
            "used_chunk_ids": used, "sources": sources}
