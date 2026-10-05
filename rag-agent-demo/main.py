"""
程序入口：接收 question → graph.invoke(...) → 打印结果

    python main.py                 # 依次运行 Case 1 ~ Case 4
    python main.py "用户问题"       # 自定义问题
"""
import sys

from langchain_core.messages import HumanMessage

from agent.graph import graph

CASES = [
    # Case 1：普通问题 —— Agent → END，不进入 RAG Workflow
    "用一句话介绍一下 Python 这门编程语言。",
    # Case 2：知识库问题，第一次检索即成功 —— Retrieve → Grade(relevant) → ToolMessage → Agent
    "根据知识库，我们团队的 Redis 集群代号是什么？主要用来做什么？",
    # Case 3：问题里的 "Agent 应用" 容易把第一次检索带偏到 Agent / RAG 主题
    #         —— 预期 Retrieve → Grade(0 relevant) → Rewrite → Retrieve → Grade(relevant)
    "根据知识库，我们的 Agent 应用里，用户会话一般存在团队的哪个组件里？",
    # Case 4：知识库里根本没有的信息 —— Rewrite 一次后仍然找不到，触发 MAX_RETRY 终止条件
    "根据知识库，我们团队的 Kafka 集群代号是什么？",
]


def run(question: str) -> None:
    print("\n" + "=" * 70)
    print(f"[User] {question}")
    print("=" * 70)
    result = graph.invoke({"messages": [HumanMessage(question)]})
    print("[Trace] " + " → ".join(type(m).__name__ for m in result["messages"]))
    if result.get("pending_tool_call_id"):
        print(
            f"[State] retrieval_query={result['retrieval_query']!r} "
            f"retry_count={result['retry_count']} "
            f"relevant_documents={len(result['relevant_documents'])}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        for case in CASES:
            run(case)
