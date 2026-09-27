"""GraphWalker worker：图谱给假设，原文给证据（spec §5.1）。

流程固定四段，顺序不可颠倒：

1. 用 non-thinking 调用把子任务目标解析成 `GraphQuery`（干净实体名 + 关系类型 + 假设）；
2. `resolve_entity` 把实体名链接到图谱节点（多候选时记歧义，不静默选一个）；
3. `walk_graph` 沿白名单关系拿关系路径；
4. 把每条路径的 `text` 当作查询去 `search_documents`。

第 4 步是设计核心：`walk_graph` 的输出只有实体与关系，**不带 chunk_id**，
直接当引用用就会让引用支撑率（Task 33）失去意义、Verifier 也无原文可校验。
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from akr.agents.errors import ToolResultError
from akr.model.routing import mode_for
from akr.model.structured import make_repair_fn, parse_with_repair
from akr.orchestration.ledger import Citation, CostRecord, SubTask, WorkerReport
from akr.storage.graph_store import ALLOWED_RELATIONS

AMBIGUITY_MARGIN = 0.05
EVIDENCE_TOP_K = 3
SNIPPET_CHARS = 160

# 关系类型白名单直接写进 prompt：既让模型少给非法值，也让这段 prompt 保持静态
# （前缀缓存要求，spec §7.5）。白名单本身仍以 `ALLOWED_RELATIONS` 为准，prompt 只是提示。
QUERY_SYSTEM = f"""你要把一个问题转成图查询，只输出 JSON。

字段：
- mention：要从图谱里链接的实体名，必须是干净的机构名/公司名（例如「甲公司」），
  不要带疑问词、不要带整句话，否则实体链接会失败。
- relation_types：需要遍历的关系类型，只能从下面这个集合里选，可以多选：
  {", ".join(sorted(ALLOWED_RELATIONS))}
- hypothesis：一句话说明你想验证的图关系假设。

只输出 JSON 对象，不要解释，不要 Markdown 代码块。"""

GRAPH_SYSTEM = """你要根据图谱关系路径与对应的原文片段，回答子任务目标，只输出 JSON。

字段：
- findings：结论文本列表，每条都必须是原文证据支持的断言；没有证据就返回空列表。
- chunk_ids：结论引用的证据编号，只能从输入里出现过的 chunk_id 中选，禁止新增编号。
- confidence：0 到 1 的浮点数，表示结论被证据支撑的程度，证据不足就给低分。
- open_questions：仍需查证的点，例如实体歧义、关系方向不明、缺少证据。

只输出 JSON 对象，不要解释，不要 Markdown 代码块。"""


class GraphQuery(BaseModel):
    """模型给出的图查询意图。`mention` 一定是实体名，不是整句话。"""

    mention: str = ""
    relation_types: list[str] = Field(default_factory=list)
    hypothesis: str = ""


class GraphSummary(BaseModel):
    findings: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    open_questions: list[str] = Field(default_factory=list)


def _call_model(client: Any, *, system: str, user: str, schema, mode: str):
    """一次模型调用 + 结构化校验，返回 (解析结果, LLMResult)。

    用量由调用方累加：本 worker 有两次调用，只记最后一次会让成本归因系统性偏低。
    """

    result = client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        mode=mode,
        json_object=True,
    )
    parsed = parse_with_repair(
        result.text,
        schema,
        repair=make_repair_fn(client, system=system, mode=mode),
    )
    return parsed, result


def _parse_query(client: Any, goal: str) -> tuple[GraphQuery, Any]:
    return _call_model(
        client,
        system=QUERY_SYSTEM,
        user=goal,
        schema=GraphQuery,
        mode=mode_for("extract"),
    )


def _resolve_entity(
    gateway: Any,
    mention: str,
    *,
    open_questions: list[str],
) -> str | None:
    """实体链接。返回 entity_id；没链接到时返回 None（调用方据此提前收工）。

    歧义不抛异常也不静默：按最高分继续走图，但把「首选与次选分差很小」写进
    open_questions。多跳问题错一个实体，后面整条路径都是错的。

    候选条数不在这一层给：`ToolGateway.resolve_entity(mention, context="")` 是协议的全部
    入参，上限由 MCP 工具内部定死（`tools.resolve_entity` 用 `limit=5`）。2026-09-25 主
    实验里这里多传了一个 `limit=5`，生产网关当场抛
    `TypeError: ... got an unexpected keyword argument 'limit'`，整条运行作废。
    """

    payload = gateway.resolve_entity(mention)
    if isinstance(payload, dict) and payload.get("error"):
        raise ToolResultError("resolve_entity", str(payload["error"]))

    candidates = [c for c in payload.get("candidates", []) if isinstance(c, dict)]
    if not candidates:
        open_questions.append(f"图谱中未找到与 {mention!r} 匹配的实体，图谱维度无结论")
        return None

    best = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        gap = float(best.get("score", 0.0)) - float(runner_up.get("score", 0.0))
        if gap < AMBIGUITY_MARGIN:
            open_questions.append(
                f"实体链接存在歧义：{best.get('name')!r} 与 {runner_up.get('name')!r} "
                f"得分差 {gap:.3f} < {AMBIGUITY_MARGIN}，本次按最高分继续"
            )
    return str(best["entity_id"])


def _walk(
    gateway: Any,
    entity_id: str,
    *,
    relation_types: list[str],
    hops: int,
    open_questions: list[str],
) -> list[dict[str, Any]]:
    """按白名单关系走图，返回路径列表。

    relation_types 是 Cypher 里唯一允许插值的位置（Global Constraints），
    `ALLOWED_RELATIONS` 是唯一防线，因此模型给的类型必须在这里再过滤一次。
    """

    dropped = [r for r in relation_types if r not in ALLOWED_RELATIONS]
    if dropped:
        open_questions.append(
            f"模型给出的关系类型 {sorted(set(dropped))} 不在白名单内，已剔除"
        )
    allowed = [r for r in relation_types if r in ALLOWED_RELATIONS]

    payload = gateway.walk_graph(entity_id, hops=hops, relation_types=allowed)
    if isinstance(payload, dict) and payload.get("error"):
        raise ToolResultError("walk_graph", str(payload["error"]))

    paths = [p for p in payload.get("paths", []) if isinstance(p, dict)]
    for path in paths:
        text = path.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ToolResultError("walk_graph", "路径缺少 text 字段，无法据此检索原文证据")
    if not paths:
        open_questions.append("图谱未返回任何关系路径，图谱维度无结论")
    return paths


def _collect_evidence(
    gateway: Any,
    paths: list[dict[str, Any]],
    *,
    open_questions: list[str],
) -> list[dict[str, Any]]:
    """把每条路径的 `text` 当查询去检索原文，返回去重后的证据命中。

    去重按 `chunk_id`：多条关系路径常指向同一段原文，重复计数会让引用支撑率虚高。
    契约字段（chunk_id/doc_id/source/text）在这里就校验，缺字段立刻抛错——否则错误
    会以「引用没有来源」的形式一路飘到 Task 33 的指标计算里才炸出来。
    """

    evidence: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = gateway.search_documents(path["text"], top_k=EVIDENCE_TOP_K)
        if isinstance(payload, dict) and payload.get("error"):
            raise ToolResultError("search_documents", str(payload["error"]))
        for hit in payload.get("hits", []):
            if not isinstance(hit, dict):
                continue
            missing = [
                key
                for key in ("chunk_id", "doc_id", "source", "text")
                if not hit.get(key)
            ]
            if missing:
                raise ToolResultError(
                    "search_documents",
                    f"命中缺少字段 {missing}，引用无法溯源到原文",
                )
            evidence.setdefault(str(hit["chunk_id"]), hit)
    return list(evidence.values())


def _user_prompt(
    subtask: SubTask,
    query: GraphQuery,
    entity_id: str | None,
    paths: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> str:
    """动态内容全部放进 user 消息，system prompt 保持静态（前缀缓存要求）。"""

    lines = [
        f"子任务目标：{subtask.goal}",
        f"图查询假设：{query.hypothesis}",
        f"链接到的实体：{entity_id}",
        "图谱关系路径：",
    ]
    lines.extend(f"- {path['text']}" for path in paths)
    lines.append("原文证据片段：")
    lines.extend(
        f"- [{hit['chunk_id']}] 来源：{hit['source']} 内容：{str(hit['text'])[:SNIPPET_CHARS]}"
        for hit in evidence
    )
    return "\n".join(lines)


def _citations_from(
    chunk_ids: list[str],
    evidence: list[dict[str, Any]],
) -> tuple[list[Citation], int]:
    """把模型给的 chunk_id 映射成带来源的引用，返回 (引用列表, 被剔除的条数)。"""

    by_id = {str(hit["chunk_id"]): hit for hit in evidence}
    citations: list[Citation] = []
    seen: set[str] = set()
    dropped = 0
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            continue
        hit = by_id.get(chunk_id)
        if hit is None:
            dropped += 1
            continue
        seen.add(chunk_id)
        citations.append(
            Citation(doc_id=str(hit["doc_id"]), chunk_id=chunk_id, source=str(hit["source"]))
        )
    return citations, dropped


def run_graph_walk(
    subtask: SubTask,
    *,
    gateway: Any,
    client: Any | None = None,
    hops: int = 2,
) -> WorkerReport:
    """执行一个图步子任务，回传结构化摘要。

    `client=None` 走纯图模式：零模型成本，只回传图谱路径与命中证据，
    confidence 固定 0.0（不表态），供联调与「图谱单独能找到什么」的对照实验。
    """

    if subtask.kind != "graph":
        raise ValueError(f"GraphWalker 只接受 kind='graph' 的子任务，收到：{subtask.kind!r}")

    started = time.perf_counter()
    open_questions: list[str] = []
    model_calls = 0
    tool_calls = 0
    tokens = 0
    hit_tokens = 0
    miss_tokens = 0
    output_tokens = 0
    cost_cny = 0.0

    def account(result: Any) -> None:
        """累加用量：本 worker 最多两次模型调用，必须求和而不是取最后一次。"""

        nonlocal tokens, hit_tokens, miss_tokens, output_tokens, cost_cny
        tokens += int(result.usage.total_tokens)
        hit_tokens += int(result.usage.hit_tokens)
        miss_tokens += int(result.usage.miss_tokens)
        output_tokens += int(result.usage.output_tokens)
        cost_cny += float(result.usage.cost_cny(peak=True))

    if client is None:
        query = GraphQuery(mention=subtask.goal, hypothesis=subtask.goal)
        open_questions.append("未启用摘要模型：只回传图谱路径与命中证据，未生成文字结论")
    else:
        query, first = _parse_query(client, subtask.goal)
        model_calls += 1
        account(first)

    mention = (query.mention or subtask.goal).strip() or subtask.goal
    tool_calls += 1
    entity_id = _resolve_entity(gateway, mention, open_questions=open_questions)

    paths: list[dict[str, Any]] = []
    if entity_id is not None:
        tool_calls += 1
        paths = _walk(
            gateway,
            entity_id,
            relation_types=query.relation_types,
            hops=hops,
            open_questions=open_questions,
        )

    evidence: list[dict[str, Any]] = []
    if paths:
        evidence = _collect_evidence(gateway, paths, open_questions=open_questions)
        tool_calls += len(paths)
        if not evidence:
            open_questions.append("图谱路径没有任何原文证据支撑，图谱维度无结论")

    def report(
        *,
        findings: list[str],
        citations: list[Citation],
        confidence: float,
    ) -> WorkerReport:
        return WorkerReport(
            findings=list(findings),
            citations=list(citations),
            confidence=confidence,
            open_questions=list(open_questions),
            cost=CostRecord(
                steps=model_calls + tool_calls,
                tokens=tokens,
                hit_tokens=hit_tokens,
                miss_tokens=miss_tokens,
                output_tokens=output_tokens,
                cost_cny=cost_cny,
                latency_ms=int(round((time.perf_counter() - started) * 1000)),
            ),
        )

    if client is None:
        citations, _ = _citations_from([str(hit["chunk_id"]) for hit in evidence], evidence)
        return report(findings=[], citations=citations, confidence=0.0)

    if not evidence:
        # 没有材料就不调归纳模型：省一次调用，也避免模型在空上下文里编结论。
        return report(findings=[], citations=[], confidence=0.0)

    summary, second = _call_model(
        client,
        system=GRAPH_SYSTEM,
        user=_user_prompt(subtask, query, entity_id, paths, evidence),
        schema=GraphSummary,
        mode=mode_for("summarize"),
    )
    model_calls += 1
    account(second)

    citations, dropped = _citations_from(list(summary.chunk_ids), evidence)
    if dropped:
        open_questions.append(
            f"不在本次检索结果中的 chunk_id，已剔除（幻觉引用）：{dropped} 条"
        )
    open_questions.extend(summary.open_questions)
    return report(
        findings=list(summary.findings),
        citations=citations,
        confidence=float(summary.confidence),
    )
