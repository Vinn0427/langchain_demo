# LangGraph Demo 学习笔记：核心知识点 Q&A

## 一、LangChain Tool 与 JSON Schema

Q1：什么是 Docstring？

Docstring 是 Python 函数、类或模块的文档字符串，通常通过三引号 `"""..."""` 编写，用于描述功能、参数和返回值。

```
@tool(parse_docstring=True)def search(query: str, top_k: int = 5) -> str:    """检索知识库文档。    Args:        query: 用户查询内容        top_k: 返回结果数量    """    return "检索结果"
```

Q2：LangChain 如何将 Python 函数转换为 Tool？

通过 `@tool` 装饰器，提取以下信息：

- 函数名 → Tool 名称
- 类型注解 → 参数类型
- Docstring → 工具描述
- Pydantic → 参数 Schema 与校验

其中 `parse_docstring=True` 可以解析符合格式的参数说明。

Q3：JSON Schema 有什么好处？

1. 统一接口契约：标准化描述 Tool 参数。
2. 约束参数生成：定义类型、必填项、枚举等约束。
3. 运行时校验：结合 Pydantic 拦截非法参数。
4. 跨语言兼容：不同语言编写的工具可使用统一 Schema。

JSON Schema 提供结构约束，但无法保证 LLM 生成的参数在业务语义上一定正确。

## 二、LangChain Message 类型

Q4：LangChain 的四种 Message 有什么区别？

| Message         | 作用                |
| --------------- | ----------------- |
| `SystemMessage` | 定义系统指令、角色与行为规则    |
| `HumanMessage`  | 表示用户输入            |
| `AIMessage`     | 表示 LLM 的回答或工具调用请求 |
| `ToolMessage`   | 表示工具执行结果          |

Q5：AIMessage 和 ToolMessage 有什么关系？

`AIMessage.tool_calls` 表示模型发起工具调用请求，`ToolMessage` 表示工具执行后的结果。

```
AIMessage(    content="",    tool_calls=[{        "name": "search",        "args": {"query": "RAG"},        "id": "call_001"    }])ToolMessage(    content="检索到相关文档",    tool_call_id="call_001")
```

两者通过 `tool_call_id` 关联。

Q6：ToolMessage 返回后，LLM 会自动继续执行吗？

不会。需要 Agent 框架或开发者再次调用 LLM。

典型 Agent Loop：

```
LLM → AIMessage(tool_calls)
             ↓
         执行 Tool
             ↓
         ToolMessage
             ↓
          再次调用 LLM
             ↓
       最终回答 / 继续调用工具
```

Q7：LangGraph 如何管理 Message 历史？

通常使用 `MessagesState`，或者通过 `add_messages` Reducer 管理消息列表。

```
from langgraph.graph import MessagesStateclass AgentState(MessagesState):    pass
```

`add_messages` 支持追加消息，以及根据消息 ID 更新已有消息，避免普通状态覆盖导致历史消息丢失。

## 三、Python functools.lru_cache

Q8：`@lru_cache` 是什么？

Python 提供的函数结果缓存装饰器，相同参数再次调用时直接返回缓存结果，避免重复计算。

```
from functools import lru_cache@lru_cache(maxsize=128)def calculate(x):    return x * x
```

Q9：LRU 缓存淘汰机制是什么？

LRU（Least Recently Used）表示最近最少使用。

缓存达到容量上限时，优先淘汰最长时间未被访问的数据。

Q10：lru_cache 的主要参数是什么？

- `maxsize=128`：最多缓存 128 项。
- `maxsize=None`：不限制缓存数量。
- `typed=True`：区分不同参数类型。

Q11：在 Agent/RAG 项目中有什么用途？

适合缓存初始化成本较高、需要复用的对象：

```
@lru_cache(maxsize=1)def get_embedding_model():    return HuggingFaceEmbeddings(        model_name="BAAI/bge-m3"    )
```

常见场景包括 Embedding 模型、Tokenizer、配置对象、客户端实例。

注意：缓存只在当前进程内有效，参数必须可哈希，多线程首次并发调用仍可能发生重复计算，也不应直接用于缓存 `async def` 返回的协程对象。

## 四、BM25 算法原理

Q12：BM25 是什么？

BM25 是一种基于关键词的文档相关性评分算法，综合考虑：

- TF：关键词在文档中的出现频率。
- IDF：关键词在整个文档集合中的稀有程度。
- 文档长度归一化：避免长文档天然获得更高分数。

Q13：BM25 的计算公式是什么？

\\[ BM25(D,Q)=\sum\_{q_i\in Q}IDF(q_i)\cdot \frac{f(q_i,D)(k_1+1)} {f(q_i,D)+k_1(1-b+b\frac{|D|}{avgdl})} \\]

其中：

| 参数                   | 含义              |
| -------------------- | --------------- |
| \\(f(q_i,D)\\)       | 关键词在文档中的词频      |
| \\(IDF(q_i)\\)       | 逆文档频率           |
| \\(\lvert D\rvert\\) | 当前文档长度          |
| \\(avgdl\\)          | 平均文档长度          |
| \\(k_1\\)            | 控制词频饱和，常用 1.2   |
| \\(b\\)              | 控制长度归一化，常用 0.75 |

Q14：为什么 BM25 要引入词频饱和？

关键词出现次数越多，相关性通常越高，但收益递减。

例如出现 10 次的价值通常不会是出现 1 次的 10 倍，因此 BM25 使用非线性公式控制词频贡献。

Q15：IDF 的作用是什么？

衡量关键词的区分能力。

例如：

- 「数据」在 90% 的文档中出现，区分度较低。
- 「HNSW」只在 1% 的文档中出现，区分度较高。

因此后者通常具有更高的 IDF 权重。

## 五、BM25 的索引与向量表示

Q16：BM25 需要建立倒排索引吗？

BM25 本身是评分算法，不强制依赖倒排索引。

但实际检索系统通常使用倒排索引加速关键词查询。

```
Redis → [Chunk1, Chunk3]
MySQL → [Chunk2]
HNSW  → [Chunk3, Chunk4]
```

通过 Term 快速定位候选文档，再计算 BM25 得分。

Q17：BM25 需要生成 Embedding 向量吗？

不需要稠密 Embedding。

传统 BM25 直接基于分词、词频、文档频率等统计信息计算得分。

但在 Qdrant 等向量数据库中，可以使用 Sparse Vector（稀疏向量） 表示 BM25 词项权重。

Q18：Dense Vector 和 Sparse Vector 有什么区别？

| 维度   | Dense Vector    | Sparse Vector |
| ---- | --------------- | ------------- |
| 来源   | Embedding 模型    | 词项统计或稀疏编码模型   |
| 特征   | 大多数维度非零         | 大多数维度为零       |
| 主要能力 | 语义相似度           | 关键词匹配         |
| 典型应用 | Dense Retrieval | BM25、SPLADE   |

BM25 Sparse Vector 一般通过词项权重编码实现，无需调用稠密 Embedding 模型。

Q19：Qdrant 如何实现 Dense + BM25 混合检索？

典型流程：

```
                  Query
                    |
           ┌────────┴────────┐
           ↓                 ↓
     Dense Embedding     BM25 Sparse
           ↓                 ↓
      Dense Search      Sparse Search
           └────────┬────────┘
                    ↓
                   RRF
                    ↓
                Reranker
                    ↓
              Retrieval Gate
                    ↓
                   LLM
```

Dense 负责语义召回，BM25 负责关键词召回，RRF 融合两路排名，Reranker 优化候选顺序。

BM25 Sparse 实现需要正确处理分词、IDF 统计和长度归一化。

## 六、需要额外记住的几个工程细节

Q20：BM25 和 Dense 的原始分数能直接相加吗？

通常不建议。

BM25 分数和 Dense 相似度的数值范围、分布不同，直接相加可能产生偏差。

常见方案是使用 RRF 根据排名融合：

\\[ RRF(d)=\sum_i\frac{1}{k+rank_i(d)} \\]

Q21：BM25 检索为什么依赖分词质量？

因为关键词的 TF、DF 等统计都基于分词结果。

中文技术文档中，需要尽量正确识别「向量数据库」「分布式锁」「HNSW」等词项，否则容易影响召回准确率。

Q22：Chunking 会影响 BM25 吗？

会。

Chunk 长度影响 TF、文档长度归一化及整个语料库的 DF 统计。

Chunk 太小可能丢失语义上下文，太大则可能稀释关键词的相关性。

因此需要结合 Recall\@K、MRR、nDCG 等指标进行评测，而不能单纯依靠固定的 Chunk 长度。

## 七、Reranker 本地部署（Cross-Encoder + ONNX Runtime）

Q23：本地部署模型和调用 API 有什么区别？

| | API 模型（Chat / Embedding） | 本地模型（Reranker） |
| --------------- | ------------------- | -------------- |
| 模型位置           | 服务商机房             | 本机磁盘          |
| 调用方式           | HTTP 请求           | 进程内函数调用      |
| 延迟构成           | 网络 + 排队 + 推理      | 纯本机计算         |
| 前置条件           | API Key            | 模型文件 + 推理引擎  |

Q24：模型文件是什么？从哪里来？

模型不是"代码"，而是一组权重文件（几亿个 float 参数），真正执行它的引擎是单独的库（fastembed 内部的 onnxruntime）。

首次构造 `TextCrossEncoder` 时，fastembed 会：

1. 先以 `local_files_only=True` 查本地缓存，命中则完全不联网；
2. 未命中才用 `snapshot_download` 从 HuggingFace 拉取所需文件：`onnx/model.onnx`（权重，约 1.04 GB）+ `tokenizer.json`（分词器）；
3. 存入 `~/.cache/fastembed/models--BAAI--bge-reranker-base/`，并记录文件大小等元数据，下次启动时校验文件完整性。

Q25：ONNX 和 ONNX Runtime 是什么关系？

ONNX（Open Neural Network Exchange）是模型交换格式：作者用 PyTorch 训练后导出 ONNX 文件（网络结构 + 权重数值）；ONNX Runtime 是执行它的推理引擎。

类比：`.mp4` 文件与播放器——不需要安装 PyTorch（2 GB+）也能跑模型（onnxruntime 仅几十 MB）。

Q26：模型是如何被加载并常驻内存的？

核心一行：

```
ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

`InferenceSession` 把权重读入内存并完成优化，之后每次 `run()` 只需喂输入、取输出，不重新加载。项目用两层机制避免重复加载：

- `@lru_cache(maxsize=1)` 的 `get_reranker()`：全进程只构造一次；
- `warmup()`：把加载挪到进程启动阶段，耗时计入 `reranker_load`，不计入任何请求的 latency。

Q27：每次 rerank() 调用时，CPU 里实际发生了什么？

1. 拼对：把 `(query, 文档)` 组成 N 个输入对（Cross-Encoder 的定义：query 与文档一起过模型，逐词交互，所以每个候选都要跑一次前向）；
2. 分词：必须用训练时同一个 tokenizer，否则 token id 对不上语义；
3. 推理：token 序列喂给 `InferenceSession`，按批（batch_size=64）执行 Transformer 前向计算；
4. 输出：每对得到一个 logit，项目里用 sigmoid 归一到 [0,1] 作为 `rerank_score`，交给 Gate 使用。

10 个候选在本机 CPU 上实测约 300~420 ms——这就是只精排 RRF Top 10、而不是全库的原因。

Q28：为什么本项目 rerank 选择本地而不是 API？

四个条件同时成立，本地才完胜 API：

1. 数值要稳：Gate 阈值（HIGH=0.9 / LOW=0.15）是基于 bge-reranker-base 在本知识库上观察到的分数分布调的。本地固定权重 = 冻结的分数坐标系；API 模型换版本会导致分布漂移，阈值在没有任何代码变更的情况下失效；
2. 长尾要小：rerank 挡在 TTFT 关键路径上（LOW 分支 Rewrite 重试时还会执行 x2）。本地 p95/p50 ≈ 1.06；公网 API 存在秒级甚至超时级长尾；
3. 调用要免费：评测和调试要反复大量调用，本地推理边际成本为零；
4. 负载扛得住：base 规模模型 + 每次仅约 10 个候选，CPU 足够，无需 GPU。

反之，候选集大、QPS 高、不想维护模型文件时应选 API Reranker。另外数据隐私在本项目不是真实约束——文档正文最终都会通过 ToolMessage 发给 Chat API。

Q29：换模型、离线部署、切换 API Reranker 分别怎么做？

- 换模型：改 `.env` 的 `RERANKER_MODEL`（需在 fastembed 支持列表内，如 `Xenova/ms-marco-MiniLM-L-6-v2`），首次运行自动下载；
- 完全离线部署：把 `~/.cache/fastembed` 整个目录拷贝到目标机器相同路径，启动时缓存命中，全程不联网；
- 切换 API Reranker：实现 `Reranker` Protocol 同签名类，在 `get_reranker()` 中返回即可，其余代码零改动；
- 内存：模型权重常驻进程约 1 GB，`lru_cache` 单例保证不会重复占用。

## 核心知识速记

| 知识点           | 一句话记忆                         |
| ------------- | ----------------------------- |
| Docstring     | 描述 Python 函数用途与参数的文档字符串       |
| JSON Schema   | 统一工具参数契约并支持结构校验               |
| SystemMessage | 定义系统行为规则                      |
| HumanMessage  | 用户输入                          |
| AIMessage     | LLM 输出或工具调用请求                 |
| ToolMessage   | 工具执行结果，通过 ID 匹配调用             |
| add_messages  | LangGraph 中管理消息追加和更新的 Reducer |
| lru_cache     | 基于参数缓存函数结果，LRU 淘汰             |
| BM25          | TF + IDF + 文档长度归一化            |
| 倒排索引          | 根据关键词快速定位包含它的文档               |
| Sparse Vector | 存储非零词项权重，适用于稀疏检索              |
| Dense Vector  | 表示文本语义，用于相似度召回                |
| RRF           | 使用排名倒数融合多路检索结果                |
| ONNX / ONNX Runtime | 模型交换格式与推理引擎，类比 mp4 与播放器   |
| InferenceSession   | 权重常驻内存的推理会话，进程内只加载一次    |
| Cross-Encoder | (query, doc) 一起过模型打分，精排更准但每候选一次前向 |
| rerank_score  | sigmoid(logit) ∈ [0,1]，Gate 阈值的判断依据  |
| 本地 vs API rerank  | 数值稳、长尾小、零成本、负载小四条件齐备才选本地 |

这份笔记覆盖了从 `@tool`、Message、`lru_cache`、BM25 计算，到倒排索引、稀疏向量，以及 Reranker 本地部署（模型文件、ONNX Runtime 加载与推理、本地 vs API 的决策）的全部问题，可直接作为 LangGraph Demo 的阶段学习总结。