"""
V4：外部模型调用 / Tool 调用的逐次 Trace。

每一次 attempt（不是每一次逻辑调用）记录一条 AttemptRecord：

    request_id  call_id  phase  model  attempt  start_time  first_token_time  end_time  latency_ms
    status      error_type  timeout_s  retry  retry_reason

- call_id   ：同一次逻辑调用的多次 attempt 共用一个 call_id（例如 ab12cd-03）
- phase     ：query_understanding / agent_decision / grader / query_rewrite / final_answer / query_embedding / ...
- status    ：success / timeout / rate_limit / server_error / connection_error / client_error /
              parse_error / validation_error / deadline_exceeded / stream_interrupted
- retry     ：本次 attempt 失败后是否会再发起下一次 attempt；retry_reason = 触发重试的 status

记录同时写入：
  1) 当前请求的 RequestMetrics.llm_calls（Eval 汇总用）
  2) logs/trace.jsonl（一行一个 JSON，事后定位）
  3) 终端一行 [LLM] ...（TRACE_PRINT=true 时）

这里不 import 任何业务模块。
"""
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import config
from observability.metrics import current

_lock = threading.Lock()


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class AttemptRecord:
    request_id: Optional[str]
    call_id: str
    phase: str
    model: str
    attempt: int
    start_time: float                      # time.time()
    end_time: Optional[float] = None
    first_token_time: Optional[float] = None
    latency_ms: Optional[float] = None
    status: str = "pending"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timeout_s: Optional[float] = None
    retry: bool = False
    retry_reason: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def first_token_ms(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    def to_json(self) -> dict:
        data = asdict(self)
        data.update(
            type="llm_attempt",
            start_time=_iso(self.start_time),
            end_time=_iso(self.end_time),
            first_token_time=_iso(self.first_token_time),
            first_token_ms=None if self.first_token_ms is None else round(self.first_token_ms, 1),
            latency_ms=None if self.latency_ms is None else round(self.latency_ms, 1),
        )
        return data


def write_jsonl(record: dict) -> None:
    path = config.TRACE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def record_attempt(rec: AttemptRecord) -> None:
    metrics = current()
    if metrics is not None:
        metrics.llm_calls.append(rec)
    write_jsonl(rec.to_json())
    if config.TRACE_PRINT:
        ft = f" first_token={rec.first_token_ms:.0f}ms" if rec.first_token_ms is not None else ""
        err = f" error={rec.error_type}" if rec.error_type else ""
        retry = f" → retry (reason={rec.retry_reason})" if rec.retry else ""
        newline = "\n" if rec.first_token_time is not None and rec.phase == "final_answer" else ""  # 流式回答之后另起一行
        print(
            f"{newline}[LLM] req={rec.request_id} call={rec.call_id} phase={rec.phase} attempt={rec.attempt} "
            f"status={rec.status} latency={rec.latency_ms:.0f}ms timeout={rec.timeout_s}s{ft}{err}{retry}",
            flush=True,
        )


def record_tool(entry: dict) -> None:
    """Tool 调用 Trace：name / args / success / error_code / latency_ms / source_ids。"""
    metrics = current()
    entry = {"type": "tool_call", "request_id": metrics.request_id if metrics else None, **entry}
    if metrics is not None:
        metrics.tool_calls.append(entry)
    write_jsonl(entry)


def record_event(_event: str, **data) -> None:
    """其他值得留痕的决策：deadline 跳过 Grader、duplicate tool call、no-answer 等。"""
    metrics = current()
    entry = {"type": "event", "event": _event, "request_id": metrics.request_id if metrics else None, **data}
    if metrics is not None:
        metrics.events.append(entry)
    write_jsonl(entry)
