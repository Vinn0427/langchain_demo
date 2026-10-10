"""
Query Understanding（V4，ROUTING_MODE=query_understanding）

输入：raw query + 最近 QU_HISTORY_TURNS 轮对话。
一次 LLM 调用（structured output）同时完成：
    Contextualization（Contextual Rewrite：指代消解 / 省略补全 / 实体与约束继承）
    + Intent Recognition + Entity / Constraint Extraction + Candidate Tool Routing

这里的 Rewrite 只是"把问题补全成可以单独理解的形式"，不扩大问题，也不是为了检索效果改写 ——
后者是 query/rewrite.py 的 Retrieval Rewrite，只在检索失败后才使用。

LLM 输出之后，代码再做一层确定性的 Contract 检查（enforce_contract），不完全依赖 Prompt：
    - raw query 中出现的 chunk_id / document_id 形式的 ID 必须保留在 standalone_query 中，并写入对应字段
    - 数字 / 否定词丢失 → 记录 violation；单轮（没有历史）时直接退回 raw query
    - CHUNK_LOOKUP 必须有 chunk_id；candidate_tools 只保留注册过的 Tool，并与 Intent 默认候选取并集
LLM 调用失败（超时 / 解析失败且无法恢复）→ status=fallback：standalone_query=raw，intent=None（交给 Tool Agent + 全部 Tool）
"""
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

import config
from llm.client import structured_call
from query.schema import QueryUnderstandingResult
from tools.registry import ALL_TOOLS, INTENT_TOOLS, TOOLS

_ID = r"(?<![A-Za-z0-9_])([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*_[0-9]{3})(?![A-Za-z0-9_])"
CHUNK_ID_RE = re.compile(_ID)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
NEGATIONS = ("不要", "不是", "不能", "不", "没有", "没", "禁止", "除了", "非")

PROMPT = """你是「团队内部知识库助手」的查询理解模块。根据【最近对话】和【当前问题】输出一个 JSON 对象，不要输出其他任何内容。

字段说明：
- standalone_query：把当前问题改写成不依赖上下文、可以单独理解的问题。
  只允许：补全省略、消解指代（它 / 这个 / 那篇 / 上面说的）、继承上文的实体和约束、规范表达。
  不得扩大或改变问题范围，不得加入用户没有问的内容，不要回答问题。
  必须原样保留：实体名、数字、时间、版本号、否定条件（不 / 禁止 / 除了 …）、document_id、chunk_id（例如 redis_003）。
  当前问题本身已经完整时，原样输出。
- intent：取值之一
  DIRECT           与团队内部知识无关的通用问题（寒暄、翻译、数学、通用编程语言常识等），不需要任何工具
  KNOWLEDGE_SEARCH 涉及团队的系统、组件、规范、流程、运维、故障处理的知识问题（包括知识库里可能没有的组件，如 Kafka）
  DOCUMENT_LOOKUP  用户明确要求某篇文档的完整内容 / 全文 / 大纲
  CHUNK_LOOKUP     用户明确引用某个 chunk_id（如 redis_003）
  DOCUMENT_LIST    询问知识库里有哪些文档，或某个主题有哪些文档
  INDEX_STATUS     询问索引本身的状态：chunk 数、文档数、embedding 模型、chunk 策略、索引版本、最近更新时间
  MULTI_TOOL       一个问题需要两类及以上工具才能完成（例如：先检索再取所在文档全文；既要列文档又要查具体知识；既要索引状态又要查知识）
- entities：关键实体（系统、组件、代号、ID）
- constraints：数字 / 时间 / 版本 / 否定条件的原文
- document_id：用户明确指定的文档，必须是【文档目录】中的 document_id；用户说的文档不在目录中时，按用户的说法给出小写 ID（如 kafka）；没有则 null
- chunk_id：用户明确提到的 chunk_id，没有则 null
- topic：DOCUMENT_LIST 时的主题过滤词（如 MySQL），没有则 null
- candidate_tools：可能用到的工具名，从【工具】中选择；DIRECT 时为 []
- confidence：0~1，你对 intent 判断的把握

【工具】（candidate_tools 只能填写下面的工具名本身，不要带括号和参数）
- search_knowledge_base：模糊知识问题的语义 / 关键词检索，参数 query
- get_document：读取某篇文档的完整内容，参数 document_id
- get_chunk：读取某个 chunk，参数 chunk_id
- list_documents：列出知识库文档，可选参数 topic
- get_index_status：索引状态，无参数

【文档目录】
{catalog}"""

_catalog_cache: dict = {"at": 0.0, "text": ""}


def _catalog_text() -> str:
    if time.time() - _catalog_cache["at"] > 60:
        try:
            from tools.document import document_catalog

            items = document_catalog()
            _catalog_cache["text"] = "\n".join(f"- {d['document_id']}：{d['title']}" for d in items) or "（空）"
        except Exception:  # noqa: BLE001
            _catalog_cache["text"] = "（目录暂不可用）"
        _catalog_cache["at"] = time.time()
    return _catalog_cache["text"]


def known_document_ids() -> set[str]:
    return {line.split("：")[0][2:] for line in _catalog_text().splitlines() if line.startswith("- ")}


@dataclass
class Understanding:
    raw_query: str
    standalone_query: str
    intent: Optional[str]
    entities: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    topic: Optional[str] = None
    candidate_tools: list = field(default_factory=list)
    confidence: float = 0.0
    status: str = "ok"                 # ok / recovered（容错解析）/ fallback（LLM 失败）
    violations: list = field(default_factory=list)
    llm_requests: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def format_history(history: list[tuple[str, str]], turns: int = config.QU_HISTORY_TURNS) -> str:
    if not history:
        return "（无）"
    lines = []
    for user, assistant in history[-turns:]:
        lines.append(f"用户：{user}")
        lines.append(f"助手：{assistant[:200]}{'…' if len(assistant) > 200 else ''}")
    return "\n".join(lines)


def enforce_contract(raw: str, result: QueryUnderstandingResult, has_history: bool) -> tuple[QueryUnderstandingResult, list[str]]:
    violations: list[str] = []
    data = result.model_dump()
    standalone = data["standalone_query"].strip()

    # 1) ID 必须保留：chunk_id 形式
    raw_chunk_ids = CHUNK_ID_RE.findall(raw)
    for cid in raw_chunk_ids:
        if cid not in standalone:
            violations.append(f"lost_id:{cid}")
            standalone = f"{standalone}（{cid}）"
    if raw_chunk_ids and not data["chunk_id"]:
        data["chunk_id"] = raw_chunk_ids[0]
        violations.append("chunk_id_filled_from_raw")
    # raw query 中出现的已知 document_id（例如 "redis_ops 文档"）也必须保留
    for doc_id in known_document_ids():
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(doc_id)}(?![A-Za-z0-9_])", raw) and doc_id not in standalone.lower() \
                and not any(c.startswith(doc_id + "_") for c in raw_chunk_ids):
            violations.append(f"lost_id:{doc_id}")
            standalone = f"{standalone}（{doc_id}）"

    # 2) 数字、否定词不能丢
    lost = [n for n in NUMBER_RE.findall(raw) if n not in standalone and not any(n in c for c in raw_chunk_ids)]
    lost += [w for w in NEGATIONS if w in raw and w not in standalone]
    if lost:
        violations.append(f"lost_constraints:{lost}")
        if not has_history:  # 单轮没有可继承的上下文：原问题本身就是最安全的 standalone_query
            standalone = raw.strip()

    # 3) Intent 与字段一致性
    if data["intent"] == "CHUNK_LOOKUP" and not data["chunk_id"]:
        violations.append("chunk_lookup_without_chunk_id")
    data["standalone_query"] = standalone

    # 4) Candidate Tool Gating：只保留注册过的 Tool（规范化 "name(args)" 这种写法），并与 Intent 默认候选取并集
    names = [re.sub(r"\(.*$", "", str(t)).strip() for t in data["candidate_tools"]]
    llm_tools = [t for t in names if t in TOOLS]
    unknown = [t for t in names if t not in TOOLS]
    if unknown:
        violations.append(f"unknown_tools:{unknown}")
    if data["intent"] == "DIRECT":
        data["candidate_tools"] = []
    else:
        merged = list(dict.fromkeys(INTENT_TOOLS[data["intent"]] + llm_tools))
        data["candidate_tools"] = merged
    return QueryUnderstandingResult.model_validate(data), violations


def understand(raw_query: str, history: Optional[list[tuple[str, str]]] = None) -> Understanding:
    history = history or []
    messages = [
        SystemMessage(PROMPT.format(catalog=_catalog_text())),
        HumanMessage(f"【最近对话】\n{format_history(history)}\n\n【当前问题】\n{raw_query}"),
    ]
    res = structured_call("query_understanding", QueryUnderstandingResult, messages,
                          timeout=config.QUERY_UNDERSTANDING_TIMEOUT, reserve=config.FINAL_ANSWER_RESERVE)
    if res.value is None:
        return Understanding(raw_query=raw_query, standalone_query=raw_query, intent=None,
                             candidate_tools=list(ALL_TOOLS), status="fallback",
                             violations=[f"llm:{res.status}:{res.error}"], llm_requests=res.llm_requests)
    value, violations = enforce_contract(raw_query, res.value, bool(history))
    return Understanding(
        raw_query=raw_query, standalone_query=value.standalone_query, intent=value.intent, entities=value.entities,
        constraints=value.constraints, document_id=value.document_id, chunk_id=value.chunk_id, topic=value.topic,
        candidate_tools=value.candidate_tools, confidence=value.confidence,
        status="ok" if res.status == "ok" else "recovered", violations=violations, llm_requests=res.llm_requests,
    )


def plan_dispatch(u: Understanding) -> Optional[dict]:
    """
    单 Tool 且参数已经由 Query Understanding 完全确定时，直接生成 tool_call，跳过一次 Agent Decision LLM 调用。
    低置信度 / MULTI_TOOL / 缺参数 / fallback → None（交给 Tool Agent，在候选 Tool 中选择）。
    """
    if not config.QU_DIRECT_DISPATCH or u.intent is None or u.confidence < config.QU_MIN_CONFIDENCE:
        return None
    if len(set(CHUNK_ID_RE.findall(u.raw_query))) > 1:  # 一个问题引用多个 chunk：需要多次调用，交给 Tool Agent
        return None
    if u.intent == "KNOWLEDGE_SEARCH":
        return {"name": "search_knowledge_base", "args": {"query": u.standalone_query}}
    if u.intent == "DOCUMENT_LOOKUP" and u.document_id:
        return {"name": "get_document", "args": {"document_id": u.document_id}}
    if u.intent == "CHUNK_LOOKUP" and u.chunk_id:
        return {"name": "get_chunk", "args": {"chunk_id": u.chunk_id}}
    if u.intent == "DOCUMENT_LIST":
        return {"name": "list_documents", "args": {"topic": u.topic} if u.topic else {}}
    if u.intent == "INDEX_STATUS":
        return {"name": "get_index_status", "args": {}}
    return None
