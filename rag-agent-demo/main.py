"""
程序入口：接收 question → graph.invoke(...) → 打印结果

    python main.py                 # 依次运行 Case 1、Case 2
    python main.py "用户问题"       # 自定义问题
"""
import sys

from langchain_core.messages import HumanMessage

from agent.graph import graph


def run(question: str) -> None:
    print("\n" + "=" * 70)
    print(f"[User] {question}")
    print("=" * 70)
    result = graph.invoke({"messages": [HumanMessage(question)]})
    print("[Trace] " + " → ".join(type(m).__name__ for m in result["messages"]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        run("用一句话介绍一下 Python 这门编程语言。")  # Case 1：普通问题，预期不调用工具
        run("根据知识库，我们团队的 Redis 集群代号是什么？主要用来做什么？")  # Case 2：预期调用工具
