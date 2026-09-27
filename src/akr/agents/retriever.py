"""Retriever worker：混合检索 + 结构化摘要（spec §5.1 / §5.4）。

职责边界刻意画得很窄：

* **工具由代码调用**：混合检索（向量 + BM25 + RRF）已经在工具层融合完成，
  worker 只调一次 `search_documents`，不自己分别召回再合并。
* **模型只做归纳**：一次 non-thinking 调用，把检索片段压成
  `WorkerReport` 的结构化字段。原文不进 Supervisor，只留 `chunk_id` 引用。
* **引用由代码映射**：模型选 chunk_id，`doc_id` / `source` 一律取自真实命中。

它收不到对话历史、收不到账本、收不到其他 worker 的输出——不是靠纪律，是靠签名。
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from akr.agents.errors import ToolResultError
from akr.model.client import LLMClient, LLMResult
from akr.model.structured import make_repair_fn, parse_with_repair
from akr.orchestration.ledger import Citation, CostRecord, SubTask, WorkerReport

SNIPPET_CHARS = 160
DEFAULT_TOP_K = 8
# 片段进 prompt 时给更宽的额度：太短会丢掉表格里的数字，而「结论对不对」几乎全靠数字。
PROMPT_SNIPPET_CHARS = 600

# 静态 system prompt：逐字固定、不含时间戳与随机 id。动态内容（子任务目标、
# 检索结果）全部放在 user 消息里，这样前缀缓存才可能命中（spec §7.5）。
SUMMARY_SYSTEM = (
    "你是企业年报检索结果的归纳器。"
    "输入是一份检索结果列表，每条包含 chunk_id、来源文件名、融合分数与原文片段。"
    "只允许依据这些片段陈述事实，不得引入外部知识，不得推测未出现的数据，不得合并口径不同的数字。"
    "输出 JSON 对象，字段为："
    "findings（数组，每条是一个结论性断言，一句话）；"
    "chunk_ids（数组，与 findings 一一对应，写该断言依据的 chunk_id）；"
    "confidence（0 到 1 的自评，仅按片段对问题的覆盖程度给分）；"
    "open_questions（数组，未能从片段中确定的问题，逐条写清缺什么）。"
    "不要输出 JSON 以外的任何内容。"
)


class RetrieverSummary(BaseModel):
    """模型必须返回的结构。用 pydantic 兜住字段缺失与类型错误（Task 7 的校验层）。"""

    findings: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    open_questions: list[str] = Field(default_factory=list)


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """压平空白并截断。检索片段里的换行会把结构化摘要挤成噪声。"""

    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _validate_hits(hits: list[dict[str, Any]], *, tool: str) -> list[dict[str, Any]]:
    """命中必须带溯源字段（spec §6.2 第 2 条）。

    缺字段说明工具层契约被破坏，属于编程错误，不该被当成"这条命中质量差"糊过去。
    """

    required = ("chunk_id", "doc_id", "source", "text")
    for hit in hits:
        missing = [field_name for field_name in required if not hit.get(field_name)]
        if missing:
            raise ToolResultError(tool, f"命中缺少字段 {missing}：{hit!r}")
    return hits


def _user_prompt(subtask: SubTask, hits: list[dict[str, Any]]) -> str:
    """worker 看到的全部动态内容：自己的子任务目标 + 自己的检索结果。"""

    lines = [
        f"子任务：{subtask.goal}",
        "",
        f"检索结果（共 {len(hits)} 条，已按 RRF 融合排序）：",
    ]
    for index, hit in enumerate(hits, start=1):
        score = float(hit.get("score", 0.0))
        lines.append(
            f"[{index}] chunk_id={hit['chunk_id']} source={hit['source']} score={score:.3f}"
        )
        lines.append(f"    {_snippet(hit['text'], PROMPT_SNIPPET_CHARS)}")
    lines.append("")
    lines.append("请只依据以上片段归纳，输出 JSON。")
    return "\n".join(lines)


def _summarize(
    client: LLMClient,
    *,
    subtask: SubTask,
    hits: list[dict[str, Any]],
    mode: str,
) -> tuple[RetrieverSummary, LLMResult]:
    """一次模型调用 + 校验与自修复。解析失败会抛 `StructuredOutputError`。"""

    result = client.chat(
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": _user_prompt(subtask, hits)}],
        mode=mode,
        json_object=True,
    )
    summary = parse_with_repair(
        result.text,
        RetrieverSummary,
        repair=make_repair_fn(client, system=SUMMARY_SYSTEM, mode=mode),
    )
    return summary, result  # type: ignore[return-value]


def _citations_from(
    chunk_ids: list[str],
    hits: list[dict[str, Any]],
) -> tuple[list[Citation], int]:
    """把模型给的 chunk_id 映射成带来源的引用，返回 (引用列表, 被剔除的数量)。

    只认命中里真实存在的 chunk_id 与真实来源：模型编造的位置无法被溯源，
    留着会让引用支撑率（Task 33）变成自证。
    """

    by_id = {hit["chunk_id"]: hit for hit in hits}
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
            Citation(doc_id=hit["doc_id"], chunk_id=hit["chunk_id"], source=hit["source"])
        )
    return citations, dropped


def run_retrieve(
    subtask: SubTask,
    *,
    gateway: Any,
    client: LLMClient | None = None,
    mode: str = "non-thinking",
    top_k: int = DEFAULT_TOP_K,
    query: str | None = None,
    two_stage_docs: int = 0,
    expand_neighbors: int = 0,
) -> WorkerReport:
    """执行一个检索子任务，回传结构化摘要。

    `client=None` 走纯检索模式：不调模型，findings 用命中原文的截断片段，
    confidence 固定 0.0（不表态）。Task 33 的检索类指标与零成本联调都用这条路径。

    两个检索策略参数由编排层给（默认值 = 旧行为，单测与纯检索路径不受影响）：

    * `query`：检索用的查询文本，默认取子任务目标。编排层传**原问句**——规划器改写过的
      子任务目标当查询实测更差（2026-09-25 只读 A/B，见 `orchestration.runner._retrieve`）。
    * `two_stage_docs > 0`：透传给工具层开启「文档内重搜」（见
      `akr.mcp_server.tools.search_documents`）。**不增加任何模型调用**。
    * `expand_neighbors > 0`：透传给工具层开启「同文档邻块扩窗」（见
      `akr.mcp_server.tools._neighbourhood`）。**不增加任何模型调用**，只多读一次本地 SQLite。

    注意：`query` 只影响检索，不改 `_user_prompt`——worker 看到的动态内容仍然只有
    自己的子任务目标与命中片段，prompt 隔离没有被这两条打开。
    """

    if subtask.kind != "retrieve":
        raise ValueError(f"Retriever 只接受 kind='retrieve' 的子任务，收到：{subtask.kind!r}")

    started = time.perf_counter()
    search_text = query or subtask.goal
    payload = gateway.search_documents(
        search_text,
        top_k=top_k,
        two_stage_docs=two_stage_docs,
        expand_neighbors=expand_neighbors,
    )
    if not isinstance(payload, dict):
        raise ToolResultError("search_documents", f"返回值不是对象：{type(payload).__name__}")
    if "error" in payload:
        raise ToolResultError("search_documents", payload["error"])
    hits = _validate_hits(list(payload.get("hits", [])), tool="search_documents")

    if not hits:
        # 没有材料可归纳：直接给诚实结论，省掉一次调用（worker 是唯一花钱的环节）。
        return WorkerReport(
            findings=[],
            citations=[],
            confidence=0.0,
            open_questions=[
                f"检索「{search_text}」无命中：语料中可能确实不存在相关内容，"
                "建议追加换关键词的子任务，或走拒答分支"
            ],
            cost=CostRecord(steps=1, tokens=0, cost_cny=0.0, latency_ms=_ms(started)),
        )

    if client is None:
        citations, _ = _citations_from([hit["chunk_id"] for hit in hits], hits)
        return WorkerReport(
            findings=[_snippet(hit["text"]) for hit in hits],
            citations=citations,
            confidence=0.0,
            open_questions=[
                "未启用摘要模型：findings 是命中原文的截断片段，confidence 不表态（0.0）"
            ],
            cost=CostRecord(steps=1, tokens=0, cost_cny=0.0, latency_ms=_ms(started)),
        )

    summary, result = _summarize(client, subtask=subtask, hits=hits, mode=mode)
    citations, dropped = _citations_from(list(summary.chunk_ids), hits)
    open_questions = list(summary.open_questions)
    if dropped:
        open_questions.append(
            f"模型给出了 {dropped} 个不在本次检索结果中的 chunk_id，已剔除（幻觉引用）"
        )

    return WorkerReport(
        findings=list(summary.findings),
        citations=citations,
        confidence=float(summary.confidence),
        open_questions=open_questions,
        cost=CostRecord(
            steps=2,  # 一次工具调用 + 一次模型调用
            # 与 Task 23 RunSummary.tokens 同口径：命中 + 未命中 + 输出
            tokens=int(result.usage.total_tokens),
            # 明细三列：编排层要按列写回 trace，缓存命中率只能从这里算（spec §7.5）
            hit_tokens=int(result.usage.hit_tokens),
            miss_tokens=int(result.usage.miss_tokens),
            output_tokens=int(result.usage.output_tokens),
            # 保守按高峰价估算：预算的目的是熔断，宁可高估也不低估
            cost_cny=float(result.usage.cost_cny(peak=True)),
            latency_ms=_ms(started),
        ),
    )
