"""
V4：所有外部模型调用（Chat / Embedding）的统一入口：分阶段 Timeout + 自己实现的 Retry + Request Deadline + Trace。

为什么不用 openai SDK 自带的 max_retries：
    V3 配置 timeout=60, max_retries=2。SDK 在内部静默重试，调用方只看到"这一次调用花了 60.47s"，
    看不到"第 1 次 attempt 读超时 60s、第 2 次 attempt 0.47s 成功"。V4 把 SDK 重试关掉（max_retries=0），
    每一次 attempt 都由这里发起并单独记录 AttemptRecord（observability/tracing.py）。

Retry Policy（classify_error）：
    timeout / connection_error / 429（非 insufficient_quota）/ 500·502·503·504  → 可重试
    其他 4xx / parse_error / validation_error / 业务结果为空 / Grader=false     → 不重试
    流式输出已经产出过 token 之后的中断                                       → 不重试（已展示给用户）

Deadline：每次 attempt 的 timeout = min(phase_timeout, remaining_budget - reserve)。
    reserve 是给后续必需阶段（最终回答）预留的预算；算出来小于 MIN_ATTEMPT_TIMEOUT 就不再发起请求，
    状态记为 deadline_exceeded，由调用方走明确的 fallback。

Structured Output：一次 LLM 调用 → raw response → 严格解析 → 失败则对同一个 raw response 容错解析
    （去 code fence / 截取 JSON / tool_call args / 单字段纯文本）→ 仍失败且有预算才允许再问一次模型。
"""
import json
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Optional, Type

import httpx
import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ValidationError

import config
from observability.metrics import current
from observability.tracing import AttemptRecord, record_attempt, record_event

# ---------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------
RETRYABLE_5XX = {500, 502, 503, 504}


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """→ (status, retryable)"""
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return "timeout", True
    if isinstance(exc, openai.RateLimitError):
        body = str(getattr(exc, "body", "") or exc)
        return "rate_limit", "insufficient_quota" not in body   # 额度用完的 429 重试没有意义
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code >= 500:
            return "server_error", exc.status_code in RETRYABLE_5XX
        return "client_error", False
    if isinstance(exc, (openai.APIConnectionError, httpx.TransportError)):
        return "connection_error", True
    return "internal_error", False


class LLMCallError(Exception):
    def __init__(self, phase: str, status: str, call_id: str, message: str = ""):
        super().__init__(f"{phase} failed: status={status} call_id={call_id} {message}")
        self.phase, self.status, self.call_id = phase, status, call_id


# ---------------------------------------------------------------------
# 故障注入（只用于复现 / 验证；默认关闭）：在 httpx transport 层制造真实的超时 / 429 / 5xx / 连接错误，
# 让 openai SDK 走它自己真实的异常路径，而不是在业务代码里 raise 一个假异常。
# ---------------------------------------------------------------------
_fault_ctx: ContextVar[Optional[tuple]] = ContextVar("llm_fault_ctx", default=None)  # (phase, attempt)


def parse_fault_spec(spec: str) -> dict:
    faults = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        phase, attempt, mode = item.split(":")
        faults[(phase, int(attempt))] = mode
    return faults


class FaultInjectingTransport(httpx.BaseTransport):
    """
    stall   ：请求发出后服务端一直不返回 —— 阻塞到本次请求的 read timeout，然后抛 httpx.ReadTimeout
    429/500 ：返回对应 HTTP 状态码
    connect ：抛 httpx.ConnectError
    stall_first_requests=N：不看 phase，前 N 个请求一律 stall（复现 V3 的 SDK 静默重试用）
    """

    def __init__(self, faults: Optional[dict] = None, stall_first_requests: int = 0):
        self._inner = httpx.HTTPTransport()
        self.faults = faults or {}
        self.stall_first_requests = stall_first_requests
        self.requests_seen = 0

    def _mode(self) -> Optional[str]:
        self.requests_seen += 1
        if self.requests_seen <= self.stall_first_requests:
            return "stall"
        ctx = _fault_ctx.get()
        return self.faults.get(ctx) if ctx else None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        mode = self._mode()
        if mode == "stall":
            read_timeout = (request.extensions.get("timeout") or {}).get("read") or 60
            time.sleep(read_timeout)
            raise httpx.ReadTimeout(f"[fault-injection] no response within read timeout {read_timeout}s", request=request)
        if mode in ("429", "500", "503"):
            return httpx.Response(int(mode), json={"error": {"message": f"[fault-injection] HTTP {mode}"}}, request=request)
        if mode == "connect":
            raise httpx.ConnectError("[fault-injection] connection refused", request=request)
        return self._inner.handle_request(request)


# ---------------------------------------------------------------------
# 模型客户端
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    kwargs: dict[str, Any] = dict(
        model=config.CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=0,
        timeout=config.FINAL_ANSWER_TIMEOUT,   # 默认值；每次调用都会显式传入分阶段 timeout
        max_retries=0,                         # V4：关闭 SDK 内部静默重试，由本模块重试并逐次记录
        extra_body={"enable_thinking": False},  # 关闭 Qwen3 的深度思考
    )
    faults = parse_fault_spec(config.LLM_FAULT_INJECTION)
    if faults:
        kwargs["http_client"] = httpx.Client(transport=FaultInjectingTransport(faults))
        print(f"[FaultInjection] enabled: {config.LLM_FAULT_INJECTION}")
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL,
        chunk_size=config.EMBEDDING_BATCH_SIZE,
        check_embedding_ctx_length=False,  # 直接发送原始文本，兼容非 OpenAI 的兼容服务商
        timeout=config.EMBEDDING_TIMEOUT,
        max_retries=0,
    )


# ---------------------------------------------------------------------
# Retry Loop
# ---------------------------------------------------------------------
def attempt_timeout(phase_timeout: float, reserve: float = 0.0) -> Optional[float]:
    """本次 attempt 可用的 timeout；None = 预算不足，不应再发起请求。"""
    metrics = current()
    remaining = metrics.remaining() if metrics else float("inf")
    timeout = min(phase_timeout, remaining - reserve)
    return round(timeout, 2) if timeout >= config.MIN_ATTEMPT_TIMEOUT else None


def _new_record(phase: str, model: str, attempt: int, call_id: str, timeout: Optional[float]) -> AttemptRecord:
    metrics = current()
    return AttemptRecord(
        request_id=metrics.request_id if metrics else None, call_id=call_id, phase=phase, model=model,
        attempt=attempt, start_time=time.time(), timeout_s=timeout,
    )


def _finish(rec: AttemptRecord, p0: float, status: str, exc: Optional[BaseException] = None) -> None:
    rec.end_time = time.time()
    rec.latency_ms = (perf_counter() - p0) * 1000
    rec.status = status
    if exc is not None:
        rec.error_type = type(exc).__name__
        rec.error_message = str(exc)[:300]


def _call_id() -> str:
    metrics = current()
    return metrics.next_call_id() if metrics else f"nometrics-{int(time.time() * 1000) % 100000}"


def _deadline_record(phase: str, model: str, attempt: int, call_id: str) -> None:
    rec = _new_record(phase, model, attempt, call_id, None)
    _finish(rec, perf_counter(), "deadline_exceeded")
    record_attempt(rec)


def run_with_retry(
    phase: str,
    fn: Callable[[float], Any],
    *,
    timeout: float,
    reserve: float = 0.0,
    max_attempts: Optional[int] = None,
    model: str = config.CHAT_MODEL,
) -> tuple[Any, AttemptRecord]:
    """fn(timeout_s) 发起一次请求。失败按 Retry Policy 重试，每次 attempt 一条 Trace。"""
    max_attempts = max_attempts or config.LLM_MAX_ATTEMPTS
    call_id = _call_id()
    for attempt in range(1, max_attempts + 1):
        t = attempt_timeout(timeout, reserve)
        if t is None:
            _deadline_record(phase, model, attempt, call_id)
            raise LLMCallError(phase, "deadline_exceeded", call_id)
        rec = _new_record(phase, model, attempt, call_id, t)
        token = _fault_ctx.set((phase, attempt))
        p0 = perf_counter()
        try:
            result = fn(t)
            _finish(rec, p0, "success")
            record_attempt(rec)
            return result, rec
        except Exception as exc:  # noqa: BLE001
            status, retryable = classify_error(exc)
            _finish(rec, p0, status, exc)
            backoff = config.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
            can_retry = (
                retryable and attempt < max_attempts and attempt_timeout(timeout, reserve + backoff) is not None
            )
            rec.retry, rec.retry_reason = can_retry, (status if can_retry else None)
            record_attempt(rec)
            if not can_retry:
                raise LLMCallError(phase, status, call_id, str(exc)[:200]) from exc
            time.sleep(backoff)
        finally:
            _fault_ctx.reset(token)
    raise LLMCallError(phase, "exhausted", call_id)  # 不会执行到这里


def invoke_llm(phase: str, runnable, messages: list[BaseMessage], *, timeout: float, reserve: float = 0.0) -> AIMessage:
    message, _ = run_with_retry(phase, lambda t: runnable.invoke(messages, timeout=t), timeout=timeout, reserve=reserve)
    return message


class StreamCall:
    """
    流式调用 + Retry：只有在"还没有收到任何有意义的 chunk"时失败才会重试；
    已经产出 token 之后中断记为 stream_interrupted，不重试（token 已经展示给用户）。
    phase 可以在迭代过程中由调用方改写（Agent Node 看到第一个 chunk 才知道是 agent_decision 还是 final_answer）。
    """

    def __init__(self, phase: str, runnable, messages: list[BaseMessage], *, timeout: float, reserve: float = 0.0):
        self.phase = phase
        self.runnable, self.messages = runnable, messages
        self.timeout, self.reserve = timeout, reserve
        self.call_id = _call_id()
        self.last_record: Optional[AttemptRecord] = None

    def __iter__(self):
        model = config.CHAT_MODEL
        for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
            t = attempt_timeout(self.timeout, self.reserve)
            if t is None:
                _deadline_record(self.phase, model, attempt, self.call_id)
                raise LLMCallError(self.phase, "deadline_exceeded", self.call_id)
            rec = _new_record(self.phase, model, attempt, self.call_id, t)
            self.last_record = rec
            token = _fault_ctx.set((self.phase, attempt))
            p0 = perf_counter()
            produced = False
            try:
                for chunk in self.runnable.stream(self.messages, timeout=t):
                    if not produced and (chunk.content or getattr(chunk, "tool_call_chunks", None)):
                        produced = True
                        rec.first_token_time = time.time()
                    yield chunk
                rec.phase = self.phase
                _finish(rec, p0, "success")
                record_attempt(rec)
                return
            except GeneratorExit:
                raise
            except Exception as exc:  # noqa: BLE001
                status, retryable = classify_error(exc)
                if produced:
                    status, retryable = "stream_interrupted", False
                rec.phase = self.phase
                _finish(rec, p0, status, exc)
                backoff = config.LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
                can_retry = (
                    retryable and attempt < config.LLM_MAX_ATTEMPTS
                    and attempt_timeout(self.timeout, self.reserve + backoff) is not None
                )
                rec.retry, rec.retry_reason = can_retry, (status if can_retry else None)
                record_attempt(rec)
                if not can_retry:
                    raise LLMCallError(self.phase, status, self.call_id, str(exc)[:200]) from exc
                time.sleep(backoff)
            finally:
                _fault_ctx.reset(token)


# ---------------------------------------------------------------------
# Structured Output：LLM Call Once → Raw Response → Strict Parse → Tolerant Parse（同一个 raw）
# ---------------------------------------------------------------------
@dataclass
class StructuredResult:
    value: Optional[BaseModel]
    status: str                 # ok / recovered / parse_error / validation_error / llm_error
    recovered_by: Optional[str] = None
    raw_text: str = ""
    llm_requests: int = 0       # 这次 structured 调用实际发出的 HTTP 请求数（含 retry 和 reask）
    error: Optional[str] = None


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


def _json_candidates(message: AIMessage) -> list[tuple[str, Any]]:
    out = []
    for call in message.tool_calls or []:
        out.append(("tool_call_args", call.get("args")))
    content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
    text = _FENCE.sub("", content.strip()).strip()
    if text:
        try:
            out.append(("tolerant_json", json.loads(text)))
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    out.append(("json_substring", json.loads(text[start:end + 1])))
                except json.JSONDecodeError:
                    pass
    return out


def _parse(schema: Type[BaseModel], message: AIMessage, plain_text_field: Optional[str]) -> tuple[Optional[BaseModel], str, Optional[str], str]:
    """→ (value, status, recovered_by, error)"""
    content = message.content if isinstance(message.content, str) else ""
    # 1) 严格解析：json_schema 模式下 content 必须本身就是合法 JSON
    try:
        if config.STRUCTURED_OUTPUT_METHOD == "function_calling" and message.tool_calls:
            return schema.model_validate(message.tool_calls[0]["args"]), "ok", "strict", ""
        return schema.model_validate_json(content), "ok", "strict", ""
    except (ValidationError, ValueError):
        pass
    # 2) 容错解析：同一个 raw response
    last_error, saw_json = "no JSON object found in raw response", False
    for source, data in _json_candidates(message):
        if not isinstance(data, dict):
            continue
        saw_json = True
        try:
            return schema.model_validate(data), "recovered", source, ""
        except ValidationError as exc:
            last_error = f"validation: {exc.errors()[:2]}"
    # 3) 单字段 schema（例如 Rewrite 的 query）：模型直接输出纯文本时，把纯文本当作该字段
    text = _FENCE.sub("", content.strip()).strip()
    if plain_text_field and text and not saw_json and len(text) <= 300 and "\n\n" not in text:
        try:
            return schema.model_validate({plain_text_field: text.strip('"“”')}), "recovered", "plain_text", ""
        except ValidationError as exc:
            last_error = f"validation: {exc.errors()[:2]}"
    return None, ("validation_error" if saw_json else "parse_error"), None, last_error


def _structured_runnable(schema: Type[BaseModel]):
    llm = get_chat_model()
    if config.STRUCTURED_OUTPUT_METHOD == "function_calling":
        return llm.bind_tools([schema], tool_choice=schema.__name__, parallel_tool_calls=False)
    return llm.bind(response_format={
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": False},
    })


def structured_call(
    phase: str,
    schema: Type[BaseModel],
    messages: list[BaseMessage],
    *,
    timeout: float,
    reserve: float = 0.0,
    plain_text_field: Optional[str] = None,
) -> StructuredResult:
    runnable = _structured_runnable(schema)
    call_ids: list[str] = []

    def _invoke(msgs: list[BaseMessage]) -> AIMessage:
        try:
            message, rec = run_with_retry(phase, lambda t: runnable.invoke(msgs, timeout=t), timeout=timeout, reserve=reserve)
            call_ids.append(rec.call_id)
            return message
        except LLMCallError as exc:
            call_ids.append(exc.call_id)
            raise

    try:
        message = _invoke(messages)
    except LLMCallError as exc:
        result = StructuredResult(value=None, status="llm_error", error=exc.status, llm_requests=_count_requests(call_ids))
        _record_parse(phase, schema, result, reask=False)
        return result

    value, status, recovered_by, error = _parse(schema, message, plain_text_field)
    result = StructuredResult(value, status, recovered_by, raw_text=str(message.content)[:500], error=error or None)
    reasked = False
    if value is None and config.STRUCTURED_MAX_REASK > 0 and attempt_timeout(timeout, reserve) is not None:
        # 只有"同一个 raw response 无法恢复"且"仍有预算"时，才再问一次模型
        reasked = True
        fix = HumanMessage(f"你上一次的输出无法解析为符合要求的 JSON（{error}）。请只输出一个符合 schema 的 JSON 对象。")
        try:
            message = _invoke(messages + [message, fix])
            value, _, recovered_by, error = _parse(schema, message, plain_text_field)
            if value is not None:
                result = StructuredResult(value, "recovered", f"reask+{recovered_by}", raw_text=str(message.content)[:500])
        except LLMCallError:
            pass
    result.llm_requests = _count_requests(call_ids)
    _record_parse(phase, schema, result, reask=reasked)
    return result


def _count_requests(call_ids: list[str]) -> int:
    """这次 structured 调用实际发出的 HTTP 请求数（所有 attempt，不含因预算不足而没有发出的）。"""
    metrics = current()
    if metrics is None:
        return len(call_ids)
    ids = set(call_ids)
    return sum(1 for r in list(metrics.llm_calls) if r.call_id in ids and r.status != "deadline_exceeded")


def _record_parse(phase: str, schema: Type[BaseModel], result: StructuredResult, reask: bool) -> None:
    record_event(
        "structured_parse", phase=phase, schema=schema.__name__, status=result.status,
        recovered_by=result.recovered_by, reask=reask, llm_requests=result.llm_requests,
        error=result.error, raw=None if result.status in ("ok",) else result.raw_text[:200],
    )
    if result.status not in ("ok",) and config.TRACE_PRINT:
        print(f"[Structured] phase={phase} schema={schema.__name__} status={result.status} "
              f"recovered_by={result.recovered_by} llm_requests={result.llm_requests} reask={reask}", flush=True)


# ---------------------------------------------------------------------
# Embedding（同样逐次 trace + retry；OpenAIEmbeddings 不支持逐次 timeout，使用客户端级 EMBEDDING_TIMEOUT）
# ---------------------------------------------------------------------
def embed_query(text: str) -> list[float]:
    vector, _ = run_with_retry(
        "query_embedding", lambda _t: get_embeddings().embed_query(text),
        timeout=config.EMBEDDING_TIMEOUT, max_attempts=config.EMBEDDING_MAX_ATTEMPTS, model=config.EMBEDDING_MODEL,
    )
    return vector


def embed_documents(texts: list[str]) -> list[list[float]]:
    """离线 Indexing：调用方负责按 EMBEDDING_BATCH_SIZE 分批，这里一批一次请求。"""
    vectors, _ = run_with_retry(
        "index_embedding", lambda _t: get_embeddings().embed_documents(texts),
        timeout=config.EMBEDDING_TIMEOUT, max_attempts=config.EMBEDDING_MAX_ATTEMPTS, model=config.EMBEDDING_MODEL,
    )
    return vectors
