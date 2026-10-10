# Version 4 进度总结：已完成的工作与后续待办

> 本文档记录 V4（Incremental Indexing & Reliability）在本次开发中**实际完成并验证**的内容，以及因时间限制**没有完成或只完成一部分**的内容。
> 所有数字都来自 `eval/reports/` 下的真实运行日志。详细设计见 `README.md` 的「Version 4」一节。

**重要背景：开发中途更换了 Chat 模型**
- Phase 1~10 的开发、Routing A/B、Query Eval、Tool Eval 都在 `qwen3.7-flash-2026-07-15` 上完成（`eval/reports/*_qwen3.7.*`）。
- 之后 `.env` 中的 `CHAT_MODEL` 改为 `qwen3.8-flash`。最终 Regression 只在新模型上跑了默认模式（`query_understanding`），**没有在新模型上重跑 A/B**。

---

## 一、已完成并验证

| Phase | 内容 | 验证证据 |
|---|---|---|
| 1 | 逐 attempt 的 LLM Trace（request_id / call_id / phase / model / attempt / start / first_token / end / latency / status / error_type / retry / retry_reason），写入 `logs/trace.jsonl` | `scripts/reproduce_timeout.py` → `eval/reports/timeout_reproduction.log` |
| 1 | 复现 V3 的 60s timeout：位于 **final_answer 等首 token**，`httpx.ReadTimeout → openai.APITimeoutError`，SDK 静默重试 1 次（重放实测 61.5s） | 同上，Part A |
| 2 | 分阶段 Timeout（QU 12 / Agent 12 / Grader 5 / Rewrite 5 / Final 10 / Embedding 5 / Tool 5 秒）+ 自己实现的 Retry（只重试 timeout / 连接错误 / 429 / 5xx，最多 2 次 attempt）+ `REQUEST_DEADLINE=30s` | 故障注入 5 个场景：timeout / 429 / 500 / connect / deadline 截断（12s deadline → 请求 12009 ms 结束） |
| 2 | Structured Output：`json_schema` 单次调用 → 严格解析 → 对同一个 raw response 容错解析；只有无法恢复且仍有预算时才 re-ask | Regression：`structured_parse_failure_rate=0`，`reask=0`；Rewrite 每次 `llm_requests=1`，**V3 的双调用已消除** |
| 3 | 按 `##`/`###` 标题层级切分，payload 区分 `raw_text` / `retrieval_text`；A/B/C 三种策略 + Chunk Analysis | `eval/reports/chunk_strategy.log`：hybrid_rerank MRR A 0.903 / B 0.939 / **C 0.953**，默认使用 C |
| 4 | SQLite Manifest + 文件级增量 Indexing（新增 / 不变 / 修改 / 删除）+ Pipeline Fingerprint | `scripts/verify_incremental.py` **19/19 通过**：第二次无修改时 0 次 chunk、0 次 embedding；修改 1 个文件只处理 7 个 chunk（其中 2 个重新 embedding、5 个复用旧向量）；删除文件后 Qdrant 中的 point 2 → 0 |
| 5 | Sparse 增量正确性：冻结 avgdl；配置 Qdrant vacuum，避免已删除的 point 继续参与 IDF 统计 | 增量结果与全量重建对比，65/65 个 query 的 BM25 分数一致；对照组（Qdrant 默认配置）有 63/65 个 query 分数不一致 |
| 5 | Blue-Green：alias + 按 fingerprint 命名的 collection + Eval Gate + 原子切换 | alias `rag_demo` 已从 (none) 切到 `rag_demo_v4_c_0073c1ee` |
| 6 | Query Understanding（Contextual Rewrite + Intent + Entity + Candidate Tools，一次 LLM 调用），由代码再做一层 Contract 检查 | 用户先问“Redis 怎么部署？”，下一轮问“那它扩容呢？”，生成的 standalone query 为 `Redis 怎么扩容？` |
| 6 | Retrieval Rewrite 与 Contextual Rewrite 分开实现：前者在 `query/rewrite.py`，后者在 `query/understanding.py` | — |
| 7 | 保留 `ROUTING_MODE=v3_agent`，并完成 A/B（qwen3.7） | 两种模式的 Tool Selection Accuracy：v3_agent 0.929，QU **1.000**；Intent Accuracy：0.952 vs **1.000**。Direct Query TTFT p50：926 ms vs 1681 ms |
| 8 | 5 个 Tool，均使用 Pydantic Args 校验，统一返回 ToolResult，并区分 5 类错误码 | 用 `tools/registry` 单独测试了 9 种调用情况 |
| 9 | Candidate Tool Gating + 动态 `bind_tools`；顺序执行同一条消息里的多个 tool_call，保证协议正确；Loop Safety（最多 4 次调用 + 重复调用检测） | 一条消息中的 2 个 `get_chunk` 都执行，并各自返回 ToolMessage；Multi-Tool 链路 `search → get_document` 跑通 |
| 10 | `eval/tool_dataset.json`（42 条）、`eval/query_dataset.json`（17 条），以及 Tool 指标和稳定性指标 | `eval/reports/tool_eval_ab_qwen3.7.log` |
| 11 | Context Dedup（chunk_id + 文本 hash）、No-Answer（固定文案，不调用 LLM）、Sources 由代码追加（只取自真实 ToolResult） | Kafka 相关问题稳定返回固定的无答案文案；`source_integrity_rate=1.0` |
| 12 | `eval/run_regression.py` 统一入口 | **Regression 在 qwen3.8-flash 上 12/12 PASS**（`eval/reports/regression.log`、`regression_summary.json`） |

### Regression 实测（qwen3.8-flash，ROUTING_MODE=query_understanding）

```text
retrieval.hybrid_rerank.recall@5 0.9778   mrr 0.9533
incremental 19/19
query.intent_accuracy 1.0   query_must_contain_rate 1.0
tool: intent 1.0 / selection 1.0 / arg_valid 1.0 / exec_success 1.0 / multi_tool 1.0 / no_answer 1.0 / source_integrity 1.0
stability: timeout_rate=0.0174 (2/115 attempts) retry_rate=0.0177 parse_failure_rate=0.0
           rewrite_rate=0.1875 grader_rate=0.1875 tool_error_rate=0.0 duplicate_rate=0.0 no_answer_rate=0.0714
latency (ms): e2e_ttft direct p50 2857 / p95 10292；rag p50 3739 / p95 10491；total p50 5457 / p95 14991 / max 20684
              query_understanding p50 2358 / p95 9797 / max 15182
```

新模型上出现了 2 次真实 timeout，trace 定位如下：
- `query_understanding` 第 1 次 attempt 在 12.29s 超时，第 2 次重试成功；
- `grader` 第 1 次 attempt 在 5.0s 超时，第 2 次重试成功。

---

## 二、未完成 / 部分完成（后续继续）

### P0：结论还不完整
1. **在 qwen3.8-flash 上重跑 Routing A/B**：`python eval/run_regression.py --ab`。目前 A/B 结论（QU 优于 V3）只在旧模型上成立。
2. **在新模型上做 V3 vs V4 的延迟对比**：V3 基线（`v3_baseline_eval.log`，total p95 3748 ms）用的是 qwen3.7，与新模型上的 V4 数据（p95 14991 ms）**不可直接比较**。需要在同一个模型上分别跑 V3（git tag / commit `5957b84`）和 V4。
3. **qwen3.8-flash 下 QU 的延迟问题**：QU 的 p95 达到 9.8s，已经是 TTFT 的主要成本。需要按新模型的分布重新确定 `QUERY_UNDERSTANDING_TIMEOUT`，或者评估更轻量的分类方案（规则预分类 / 更小的模型 / 只在多轮对话时做 QU）。
4. README 中「Regression 实测」一节还只是占位，需要把本文档第一部分的数字写进去。

### P1：功能缺口
5. Blue-Green 还缺 **回滚 CLI**（把 alias 切回 retired collection）和 **旧 collection 清理策略**（目前只标记为 retired，不会删除）。V3 的 `rag_demo_v3` collection 也还在。
6. Gate 阈值（0.9 / 0.15）**没有针对策略 C 重新校准**：无答案 query 的 rerank top1 最高达到 0.795，会进入 MEDIUM，需要依赖 Grader 兜底。
7. Context Dedup 只做了 chunk_id 和文本 hash 去重，**没有做相似度去重**。
8. Deadline 降级（跳过 Grader / 跳过 Rewrite / 停用 Tool）只在故障注入下验证过，Eval 中**没有专门的 deadline 场景统计**。
9. `logs/trace.jsonl` 没有做轮转和大小限制。
10. 没有单元测试（`pytest`），目前的验证都依赖 Eval 和脚本。

### P2：数据与评测
11. Eval 规模较小（检索 60 条、Tool 42 条、Query 17 条），而且 LLM 输出有随机性，单次运行之间的差异不具统计意义。需要扩充样本，并对每条样本多次运行取均值。
12. `exact_id` 类 query（问题里直接写 `redis_003`）的向量检索 MRR 只有 0.1。目前靠 QU 把这类问题路由到 `get_chunk` 来解决，检索层本身没有改进。
13. 没有评测 Embedding Provider 的不确定性（同一个 batch 重复请求 cos≈0.998）对 Recall 的影响。

---

## 三、本次提交的主要文件

```text
新增：llm/client.py  observability/{tracing,versions}.py  query/{schema,understanding,rewrite}.py
      tools/{schemas,registry,document,index_status}.py  rag/{chunking,manifest}.py
      scripts/{reproduce_timeout,verify_incremental}.py
      eval/{common,run_retrieval_eval,run_chunk_eval,run_query_eval,run_tool_eval,run_regression}.py
      eval/{retrieval,query,tool}_dataset.json  eval/reports/*  data/{redis_ops,mysql_ops}.md
修改：agent/{nodes,graph,state}.py  rag/{indexing,store,pipeline,dense_retriever,sparse_retriever,reranker}.py
      config.py  main.py  scripts/build_index.py  README.md  .env.example
删除：agent/tools.py（→ tools/registry.py）  eval/run_eval.py、eval/dataset.json（→ run_retrieval_eval.py、retrieval_dataset.json，包含原 24 条）
未提交（.gitignore）：.env  logs/  index_state/  qdrant_storage/
```
