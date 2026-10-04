## Redis 的用途

Redis 是一个基于内存的 Key-Value 数据库，读写速度极快。常见用途包括：缓存、分布式锁、计数器、排行榜、简单消息队列（List / Stream）。
在我们团队，Redis 集群的内部代号是「青鸟」（Bluebird），主要用于缓存用户会话（Session）和接口限流。
团队规范要求：所有 key 必须以 `demo:` 作为前缀，并且必须设置过期时间。

## RAG 的基本流程

RAG（Retrieval-Augmented Generation，检索增强生成）是让大模型"先查资料再回答"的技术。
本知识库将 RAG 流程总结为"五步法"：
1. 加载（Load）：读取原始文档；
2. 切分（Split）：把长文档切成较小的 chunk；
3. 向量化（Embed）：用 Embedding 模型把 chunk 变成向量，存入向量库；
4. 检索（Retrieve）：把用户问题向量化，找出最相似的若干 chunk；
5. 生成（Generate）：把检索到的 chunk 和问题一起交给 LLM 生成答案。

## Agent 的基本定义

Agent（智能体）是一个能够自主决定"下一步做什么"的 LLM 程序。它的核心是一个循环：
- Reason（思考）：LLM 根据当前上下文决定是直接回答，还是调用某个工具；
- Act（行动）：程序执行 LLM 选择的工具；
- Observe（观察）：把工具结果放回上下文，再交给 LLM 继续思考。
当 LLM 不再请求调用工具、而是直接给出答案时，循环结束。
本知识库把这个循环称为"ReAct 三拍子"。
