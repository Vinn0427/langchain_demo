"""
Online Query Pipeline 入口：question → graph.stream(...) → 流式打印最终回答（+ Sources）→ Performance Report

    python main.py                                   # 依次运行内置 Case（含多轮对话 Case）
    python main.py "用户问题"                         # 自定义问题
    python main.py --chat                            # 交互式多轮对话
    python main.py --mode v3_agent "Redis 怎么扩容？"  # 切换 Routing Mode（默认取 config.ROUTING_MODE）

前置条件：Qdrant 已启动，且已运行过 python scripts/build_index.py（alias 已指向一个 collection）。
"""
import argparse
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

import config
from agent.graph import RECURSION_LIMIT, graph
from observability.metrics import RequestMetrics, start_request
from rag.pipeline import warmup

CASES = [
    # 普通问题：DIRECT，不进入任何 Tool
    ["用一句话介绍一下 Python 这门编程语言。"],
    # 知识问题 → search_knowledge_base
    ["Redis 怎么扩容？"],
    # 多轮：Contextual Rewrite（"它" → Redis）
    ["Redis 怎么部署？", "那它扩容呢？"],
    # Exact Lookup / System Tool
    ["把 redis 文档完整内容给我"],
    ["看一下 redis_003"],
    ["有哪些 MySQL 文档？"],
    ["当前一共有多少 chunk？"],
    # No-Answer：知识库里没有
    ["Kafka ISR 是怎么实现的？"],
    # Multi-Tool：先检索定位文档，再读取该文档全文
    ["知识库里讲 Redis 脑裂处理的内容出自哪篇文档？把那篇文档的完整内容也给我。"],
]


def ask(question: str, history: Optional[list[tuple[str, str]]] = None, routing_mode: Optional[str] = None,
        show: bool = True, deadline_s: Optional[float] = config.REQUEST_DEADLINE) -> tuple[dict, RequestMetrics]:
    """运行一次完整请求，返回最终 State 和本次请求的 RequestMetrics。history = [(用户问题, 助手回答), ...]"""
    metrics = start_request(deadline_s)  # request_start：收到用户 Query
    final_state: dict = {}
    streaming = False
    messages = []
    for user, assistant in history or []:
        messages += [HumanMessage(user), AIMessage(assistant)]
    messages.append(HumanMessage(question))
    inputs = {"messages": messages, "routing_mode": routing_mode or config.ROUTING_MODE}

    # custom：agent_node 通过 stream writer 推送的最终回答 token（tool_call 永远不会出现在这里）
    # values：每一步之后的完整 State，取最后一个作为最终结果
    for mode, payload in graph.stream(inputs, stream_mode=["custom", "values"],
                                      config={"recursion_limit": RECURSION_LIMIT}):
        if mode == "values":
            final_state = payload
            continue
        if payload.get("retract"):
            if streaming and show:
                print("\n[Agent] ↑ 以上是 tool_call 前的前导文本，已撤回，不计入最终回答", flush=True)
            streaming = False
            continue
        token = payload["answer_token"]
        if not streaming:
            streaming = True
            if show:
                print("[Assistant] ", end="", flush=True)
        if show:
            print(token, end="", flush=True)
        if "first_visible_token" not in metrics.marks:
            metrics.mark("first_visible_token")  # 第一个 token 真正打印到终端之后
    if streaming and show:
        print(flush=True)

    metrics.finish()  # request_end
    return final_state, metrics


def final_answer(state: dict) -> str:
    last = state["messages"][-1]
    return last.content if isinstance(last, AIMessage) else ""


def run(question: str, history: Optional[list] = None, routing_mode: Optional[str] = None) -> dict:
    print("\n" + "=" * 70)
    print(f"[User] {question}" + (f"   (history: {len(history)} turns)" if history else ""))
    print("=" * 70)
    result, metrics = ask(question, history, routing_mode)
    print("[Trace] " + " → ".join(type(m).__name__ for m in result["messages"][-12:]))
    print(f"[State] routing_mode={result.get('routing_mode')} intent={result.get('intent')} "
          f"standalone_query={result.get('standalone_query')!r} candidate_tools={result.get('candidate_tools')}")
    for e in result.get("tool_log") or []:
        print(f"[State] tool {e['name']} args={e['args']} success={e['success']} error={e['error_code']} "
              f"retrieval_status={e.get('retrieval_status')} sources={e['source_ids']}")
    print(f"[State] final_answer_kind={result.get('final_answer_kind')} used_chunk_ids={result.get('used_chunk_ids')}")
    print(metrics.report())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--mode", choices=["v3_agent", "query_understanding"], default=None)
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    info = warmup()  # 进程启动时：检查索引、加载 Reranker（不计入任何一次请求的 latency）
    timings = ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in info["timings"].items())
    print(f"[Startup] alias={config.QDRANT_ALIAS} → {info['collection']} points={info['points']}  {timings}  "
          f"routing_mode={args.mode or config.ROUTING_MODE}")

    if args.chat:
        history: list[tuple[str, str]] = []
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question:
                state = run(question, history, args.mode)
                history.append((question, final_answer(state)))
    elif args.question:
        run(" ".join(args.question), routing_mode=args.mode)
    else:
        for conversation in CASES:
            history = []
            for question in conversation:
                state = run(question, history, args.mode)
                history.append((question, final_answer(state)))


if __name__ == "__main__":
    main()
