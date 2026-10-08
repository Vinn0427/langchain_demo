# RAG 与向量检索

## RAG 的基本流程

RAG（Retrieval-Augmented Generation，检索增强生成）是让大模型"先查资料再回答"的技术。
本知识库将 RAG 流程总结为"五步法"：
1. 加载（Load）：读取原始文档；
2. 切分（Split）：把长文档切成较小的 chunk；
3. 向量化（Embed）：用 Embedding 模型把 chunk 变成向量，存入向量库；
4. 检索（Retrieve）：把用户问题向量化，找出最相似的若干 chunk；
5. 生成（Generate）：把检索到的 chunk 和问题一起交给 LLM 生成答案。

## 文档切分规范

团队约定：文档优先按 Markdown 二级标题切分，每个 chunk 不超过 500 个汉字；超长段落再按句子切分，相邻 chunk 之间保留 50 字重叠。
每个 chunk 必须携带来源文件名和标题作为元数据，方便在回答中标注出处。

## 向量数据库选型

团队统一使用 Qdrant 作为向量数据库，生产集群的内部代号是「灯塔」（Lighthouse）。
Embedding 向量维度统一为 1024，距离度量使用余弦相似度（Cosine）。
本地开发使用 Docker 启动单节点 Qdrant，REST 端口 6333，gRPC 端口 6334。
