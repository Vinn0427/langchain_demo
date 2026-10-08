"""
Online Query Pipeline 入口：question → graph.stream(...) → 流式打印最终回答 → Performance Report

    python main.py                 # 依次运行内置 Case
    python main.py "用户问题"       # 自定义问题

前置条件：Qdrant 已启动，且已运行过 python scripts/build_index.py。
本进程只读取已有索引，不会对文档重新做 Embedding。
"""
import sys

from langchain_core.messages import HumanMessage

from agent.graph import graph
from observability.metrics import RequestMetrics, start_request
from rag.pipeline import warmup

CASES = [
    # Case 1：普通问题 —— Agent → END，完全不进入 RAG
    "用一句话介绍一下 Python 这门编程语言。",
    # Case 2：清晰的知识库问题 —— Hybrid → Rerank → Gate HIGH → ToolMessage（不调用 LLM Grader）
    "根据知识库，我们团队的 Redis 集群代号是什么？主要用来做什么？",
    # Case 3：表述模糊 —— 预期 Gate MEDIUM → LLM Grader
    "我们这边一般怎么上线？",
    # Case 4："Agent 应用" 容易把检索带偏 —— 预期 Gate LOW → Rewrite → Retry
    "根据知识库，我们的 Agent 应用里，用户会话一般存在团队的哪个组件里？",
    # Case 5：知识库中不存在 —— LOW → Rewrite → LOW → Max Retry → "没找到"
    "根据知识库，我们团队的 Kafka 集群代号是什么？",
]


def ask(question: str, show: bool = True) -> tuple[dict, RequestMetrics]:
    """运行一次完整请求，返回最终 State 和本次请求的 RequestMetrics。"""
    metrics = start_request()  # request_start：收到用户 Query
    final_state: dict = {}
    streaming = False

    # custom：agent_node 通过 stream writer 推送的最终回答 token（tool_call 永远不会出现在这里）
    # values：每一步之后的完整 State，取最后一个作为最终结果
    for mode, payload in graph.stream(
        {"messages": [HumanMessage(question)]}, stream_mode=["custom", "values"]
    ):
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


def run(question: str) -> None:
    print("\n" + "=" * 70)
    print(f"[User] {question}")
    print("=" * 70)
    result, metrics = ask(question)
    print("[Trace] " + " → ".join(type(m).__name__ for m in result["messages"]))
    if result.get("pending_tool_call_id"):
        print(
            f"[State] retrieval_query={result['retrieval_query']!r} gate_level={result['gate_level']} "
            f"top_rerank_score={result['top_rerank_score']:.3f} retry_count={result['retry_count']} "
            f"relevant_documents={[d.metadata['chunk_id'] for d in result['relevant_documents']]}"
        )
    print(metrics.report())


if __name__ == "__main__":
    info = warmup()  # 进程启动时：检查索引、加载 Reranker（不计入任何一次请求的 latency）
    timings = ", ".join(f"{k}={v * 1000:.0f}ms" for k, v in info["timings"].items())
    print(f"[Startup] Qdrant points={info['points']}  {timings}")
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        for case in CASES:
            run(case)
