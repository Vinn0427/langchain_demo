"""
请求级性能监控：只用 time.perf_counter()，不引入任何监控组件。

用法：
    metrics = start_request()          # 收到用户 Query 的时刻 = request_start
    with timer("dense_retrieval"):     # 任意层（rag / agent）记录一个阶段耗时
        ...
    mark("final_llm_start")            # 记录一个时间点
    metrics.finish()                   # request_end
    print(metrics.report())

两类数据：
- stages：阶段耗时（秒）。同一阶段可执行多次（例如 Rewrite 后再检索一次），全部保留，报告里显示 总和 + 次数。
          没出现在 stages 里的阶段 = 本次请求没有执行（报告显示 skipped），和"执行了但很快"（0.1 ms）区分开。
- marks ：时间点（perf_counter 读数）。TTFT 等指标由时间点相减得到，而不是阶段相加。

当前请求的 RequestMetrics 放在 ContextVar 里，rag 层不需要显式接收 metrics 参数，也不需要 import agent。
LangGraph 在线程池中执行节点时会复制 Context，所以节点里拿到的是同一个 RequestMetrics 对象。
"""
import itertools
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Optional

_current: ContextVar[Optional["RequestMetrics"]] = ContextVar("request_metrics", default=None)


class RequestMetrics:
    def __init__(self, deadline_s: Optional[float] = None) -> None:
        self.stages: dict[str, list[float]] = defaultdict(list)
        self.marks: dict[str, float] = {"request_start": perf_counter()}
        self.notes: dict[str, str] = {}
        self.final_message_id: Optional[str] = None
        # ---- V4 ----
        self.request_id: str = uuid.uuid4().hex[:8]
        self.deadline_s: Optional[float] = deadline_s   # 请求级 Latency Budget；None = 不限制
        self.llm_calls: list = []    # observability.tracing.AttemptRecord（每个 attempt 一条）
        self.tool_calls: list = []   # Tool 调用 trace
        self.events: list = []       # deadline 降级 / duplicate tool call / no-answer 等决策
        self._call_seq = itertools.count(1)

    # ---------- V4：Deadline / Budget ----------
    def elapsed(self) -> float:
        return perf_counter() - self.marks["request_start"]

    def remaining(self) -> float:
        if self.deadline_s is None:
            return float("inf")
        return self.deadline_s - self.elapsed()

    def next_call_id(self) -> str:
        return f"{self.request_id}-{next(self._call_seq):02d}"

    # ---------- 写入 ----------
    def add(self, stage: str, seconds: float) -> None:
        self.stages[stage].append(seconds)

    def mark(self, event: str, t: Optional[float] = None) -> None:
        self.marks[event] = perf_counter() if t is None else t

    def finish(self) -> None:
        self.mark("request_end")

    # ---------- 读取 ----------
    def stage_ms(self, stage: str) -> Optional[float]:
        """阶段总耗时（ms）；None = 没有执行。"""
        if stage not in self.stages:
            return None
        return sum(self.stages[stage]) * 1000

    def interval_ms(self, start: str, end: str) -> Optional[float]:
        if start not in self.marks or end not in self.marks:
            return None
        return (self.marks[end] - self.marks[start]) * 1000

    @property
    def model_ttft_ms(self) -> Optional[float]:
        # 最终回答那次 LLM 请求发出 → 收到该请求的第一个输出 token
        return self.interval_ms("final_llm_start", "final_first_token")

    @property
    def e2e_ttft_ms(self) -> Optional[float]:
        # 收到用户 Query → 用户看到最终回答的第一个 token（终端打印出来的时刻）
        return self.interval_ms("request_start", "first_visible_token")

    @property
    def generation_ms(self) -> Optional[float]:
        # 最终回答第一个 token → 最后一个 token（decode 阶段）
        return self.interval_ms("final_first_token", "final_llm_end")

    @property
    def total_ms(self) -> Optional[float]:
        # 墙钟时间：request_start → request_end，不是各阶段之和
        return self.interval_ms("request_start", "request_end")

    def summary(self) -> dict:
        """供 Eval 汇总使用的扁平字典（ms）。"""
        data = {name: self.stage_ms(name) for name in self.stages}
        data.update(
            model_ttft=self.model_ttft_ms,
            e2e_ttft=self.e2e_ttft_ms,
            generation=self.generation_ms,
            total=self.total_ms,
        )
        return data

    # ---------- 报告 ----------
    def _line(self, name: str, indent: int = 0) -> str:
        label = " " * indent + name
        value = self.stage_ms(name)
        if value is None:
            return f"{label:<24}{'skipped':>12}"
        count = len(self.stages[name])
        suffix = f"   (x{count})" if count > 1 else ""
        number = f"{value:.1f}" if value >= 1 else f"{value:.3f}"  # 执行了但很快：0.012 ms，而不是 skipped
        return f"{label:<24}{number:>9} ms{suffix}"

    def _value_line(self, name: str, value: Optional[float], note: str = "") -> str:
        text = "n/a" if value is None else f"{value:.1f} ms"
        return f"{name:<24}{text:>12}{('   ' + note) if note else ''}"

    def report(self) -> str:
        lines = ["========== Performance =========="]
        lines.append(self._line("query_understanding"))
        if "agent_decision" in self.stages or "query_understanding" in self.stages:
            lines.append(self._line("agent_decision"))
        else:
            lines.append(f"{'agent_decision':<24}{'n/a':>12}   直接回答：决策与回答在同一次 final_llm 中完成")
        lines.append(self._line("hybrid_retrieval") + ("   = 下面 4 项的墙钟时间" if "hybrid_retrieval" in self.stages else ""))
        for stage in ("query_embedding", "dense_retrieval", "sparse_retrieval", "rrf_fusion"):
            lines.append(self._line(stage, indent=2))
        for stage in ("rerank", "retrieval_gate", "llm_grader", "query_rewrite", "build_tool_message", "tool_execution"):
            lines.append(self._line(stage))
        lines.append("")
        lines.append(self._line("final_llm") + ("   = model_ttft + generation" if "final_llm" in self.stages else ""))
        lines.append(self._value_line("model_ttft", self.model_ttft_ms))
        lines.append(self._value_line("e2e_ttft", self.e2e_ttft_ms))
        lines.append(self._value_line("generation", self.generation_ms))
        lines.append("")
        lines.append(self._value_line("total", self.total_ms, "墙钟时间，不等于各阶段相加"))
        if self.deadline_s is not None:
            lines.append(f"{'deadline':<24}{self.deadline_s * 1000:>9.0f} ms   request_id={self.request_id}")
        if self.llm_calls:
            lines.append("")
            lines.append("---------- LLM / Embedding attempts ----------")
            for rec in self.llm_calls:
                ft = f" ttft={rec.first_token_ms:.0f}" if rec.first_token_ms is not None else ""
                retry = f" retry({rec.retry_reason})" if rec.retry else ""
                lines.append(
                    f"{rec.call_id:<12}{rec.phase:<20}#{rec.attempt} {rec.status:<17}"
                    f"{rec.latency_ms:>8.0f} ms{ft}{retry}"
                )
        for event in self.events:
            detail = {k: v for k, v in event.items() if k not in ("type", "event", "request_id")}
            lines.append(f"[event] {event['event']} {detail}")
        for key, value in self.notes.items():
            lines.append(f"[note] {key}: {value}")
        lines.append("=================================")
        return "\n".join(lines)


# ---------------- 当前请求（ContextVar）----------------
def start_request(deadline_s: Optional[float] = None) -> RequestMetrics:
    metrics = RequestMetrics(deadline_s)
    _current.set(metrics)
    return metrics


def current() -> Optional[RequestMetrics]:
    return _current.get()


@contextmanager
def timer(stage: str):
    """记录一个阶段的耗时；当前没有请求上下文（例如离线 Indexing）时只计时不记录。"""
    start = perf_counter()
    try:
        yield
    finally:
        metrics = _current.get()
        if metrics is not None:
            metrics.add(stage, perf_counter() - start)


def mark(event: str, t: Optional[float] = None) -> None:
    metrics = _current.get()
    if metrics is not None:
        metrics.mark(event, t)
