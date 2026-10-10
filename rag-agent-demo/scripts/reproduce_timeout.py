"""
复现并定位 V3 的约 60s Chat API timeout，并验证 V4 的分阶段 Timeout / Retry / Deadline。

    python scripts/reproduce_timeout.py              # 全部场景（Part A 会真实等待约 60s）
    python scripts/reproduce_timeout.py --skip-v3    # 只跑 V4 场景

故障注入在 httpx transport 层完成（llm/client.FaultInjectingTransport）：请求真实发出前被挂起直到 read timeout，
openai SDK 走它自己真实的 ReadTimeout → APITimeoutError 路径；不是在业务代码里 raise 一个假异常。

Part A  V3 配置重放：timeout=60, max_retries=2（SDK 内部静默重试）+ 第 1 个 HTTP 请求挂起
        → 调用方只看到"一次 60.x s 的调用"，看不到第几次 attempt、什么错误
Part B  V4：同样的故障注入到指定 phase / attempt，trace 逐 attempt 输出 phase / attempt / status / latency / retry
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

REPORT = ROOT / "eval" / "reports" / "timeout_reproduction.log"
SCENARIOS = [
    ("final_answer 首 token 挂起（复现 V3 的 60s 位置）", "final_answer:1:stall", "query_understanding", {},
     "Redis 怎么扩容？"),
    ("query_understanding 返回 429", "query_understanding:1:429", "query_understanding", {}, "灰度放量分几个阶段？"),
    ("v3_agent 模式 agent_decision 返回 500", "agent_decision:1:500", "v3_agent", {}, "灰度放量分几个阶段？"),
    ("query_rewrite 连接失败", "query_rewrite:1:connect", "query_understanding", {}, "Kafka ISR 是怎么实现的？"),
    ("REQUEST_DEADLINE=12s 时 final_answer 连续挂起（尾延迟被 deadline 截断）", "final_answer:1:stall,final_answer:2:stall",
     "query_understanding", {"REQUEST_DEADLINE": "12"}, "Redis 怎么扩容？"),
]


def out(line: str = "") -> None:
    print(line, flush=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def part_a(v3_timeout: float) -> None:
    import httpx
    from langchain_openai import ChatOpenAI

    from llm.client import FaultInjectingTransport

    out(f"\n==================== Part A：V3 配置重放（timeout={v3_timeout}, max_retries=2）====================")
    transport = FaultInjectingTransport(stall_first_requests=1)
    llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL,
                     temperature=0, timeout=v3_timeout, max_retries=2, extra_body={"enable_thinking": False},
                     http_client=httpx.Client(transport=transport))
    start = time.perf_counter()
    first = None
    for chunk in llm.stream("用一句话介绍 Redis。"):
        if chunk.content and first is None:
            first = time.perf_counter() - start
    total = time.perf_counter() - start
    out(f"[V3 view] final_llm 调用：model_ttft={first * 1000:.0f} ms  total={total * 1000:.0f} ms  —— 调用方能看到的只有这两个数")
    out(f"[V3 hidden] SDK 实际发出 HTTP 请求 {transport.requests_seen} 次：第 1 次在 read timeout {v3_timeout}s 后 "
        f"httpx.ReadTimeout → openai.APITimeoutError，SDK 静默重试；第 2 次成功（≈{(first - v3_timeout) * 1000:.0f} ms 首 token）")


def part_b() -> None:
    out("\n==================== Part B：V4 分阶段 Timeout + 逐 attempt Trace ====================")
    for title, fault, mode, extra_env, question in SCENARIOS:
        env = {**os.environ, "LLM_FAULT_INJECTION": fault, "ROUTING_MODE": mode, **extra_env}
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, str(ROOT / "main.py"), question], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=180)
        wall = time.perf_counter() - start
        out(f"\n--- {title}  [LLM_FAULT_INJECTION={fault} ROUTING_MODE={mode} {extra_env or ''}] ---")
        out(f"[User] {question}")
        for line in proc.stdout.splitlines():
            if re.match(r"^\[(LLM|Structured|Deadline|Assistant|FaultInjection)\]", line) or "request_id=" in line \
                    or re.match(r"^(total|e2e_ttft|model_ttft)\s", line) or re.match(r"^\[event\] (tools_disabled|grader_skipped|rewrite_skipped)", line):
                out("  " + line[:220])
        if proc.returncode != 0:
            out(f"  [exit={proc.returncode}] {proc.stderr[-500:]}")
        out(f"  (process wall time incl. startup {wall:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-v3", action="store_true")
    parser.add_argument("--v3-timeout", type=float, default=60.0)
    args = parser.parse_args()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("", encoding="utf-8")
    if not args.skip_v3:
        part_a(args.v3_timeout)
    part_b()
    out(f"\n[Report] {REPORT}")


if __name__ == "__main__":
    main()
