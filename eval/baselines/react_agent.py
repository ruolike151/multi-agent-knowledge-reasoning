"""Baseline A：单 Agent ReAct（原生 tool calls）。

存在的意义是「同模型、同工具、同价格表」下的对照：所有能对齐的口径都对齐，
剩下的差异才是架构差异。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from akr.agents.errors import ToolResultError
from akr.agents.tool_gateway import ToolGateway
from akr.model.client import LLMClient
from akr.model.cost import Usage
from akr.orchestration.ledger import Citation

SYSTEM = "react"
MAX_STEPS = 12
# 与目标系统同一条墙钟（`akr.orchestration.budget.Budget.max_wall_seconds = 120`）。
# 2026-09-25 主实验里 `ours` 有 8 条被这条闸门打断进退化链，而当时的基线**根本没有**
# 时间闸门——两组在时间预算上不对称，「ours 慢」就既可能是架构代价、也可能是只有一组
# 被截断造成的假象。补上同一条闸门后，两组的资源盒子（步数 + 墙钟）才可比。
MAX_WALL_SECONDS = 120.0
TOOL_NAMES = (
    "search_documents",
    "walk_graph",
    "resolve_entity",
    "fetch_chunk",
    "list_documents",
)

REACT_SYSTEM = (
    "你是企业年报问答助手。你可以调用工具检索语料，然后给出结论。"
    "规则：1) 必须先用工具取证再回答；2) 结论只能来自工具返回的原文；"
    "3) 不再需要工具时直接给出最终答案，并在答案中标注所用到的 chunk_id；"
    "4) 证据不足就说无法确定，不要编造数字。"
)

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "按自然语言查询做混合检索（向量 + BM25 + RRF），返回带 chunk_id 的片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索语句"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 8"},
                    "doc_filters": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "walk_graph",
            "description": "从实体出发做多跳关系遍历（子公司 / 供应商 / 客户 / 竞争 / 提及）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "hops": {"type": "integer"},
                    "relation_types": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_entity",
            "description": "把口语化的实体名对齐到图谱里的标准实体。",
            "parameters": {
                "type": "object",
                "properties": {"mention": {"type": "string"}, "context": {"type": "string"}},
                "required": ["mention"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_chunk",
            "description": "按 chunk_id 取回 chunk 原文全文。",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出当前语料里的文档。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class BaselineResult(BaseModel):
    """基线的单次运行结果；字段与目标系统的 `RunResult` 对齐，方便三组共用一张结果表。"""

    system: str = SYSTEM
    question: str
    answer: str
    citations: list[Citation] = []
    steps: int = 0
    cost_cny: float = 0.0
    tokens: int = 0
    tool_calls: int = 0
    degraded: bool = False
    latency_ms: int = 0


def _add_usage(total: Usage, extra: Usage) -> Usage:
    """手工累加：`Usage` 是计价用的不可变值对象，不假设它实现了 `__add__`。"""
    return Usage(
        hit_tokens=total.hit_tokens + extra.hit_tokens,
        miss_tokens=total.miss_tokens + extra.miss_tokens,
        output_tokens=total.output_tokens + extra.output_tokens,
    )


def collect_payload_citations(payload: Any) -> list[Citation]:
    """递归扫工具返回值里任何带 chunk_id + doc_id 的条目。

    不针对每个工具的返回形状写死解析：图谱路径、检索命中、chunk 详情里都可能夹带证据，
    少解析一处，基线的 recall 就被低估一截。
    """
    found: dict[str, Citation] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            chunk_id = node.get("chunk_id")
            doc_id = node.get("doc_id")
            if isinstance(chunk_id, str) and isinstance(doc_id, str):
                found.setdefault(
                    chunk_id,
                    Citation(doc_id=doc_id, chunk_id=chunk_id,
                             source=str(node.get("source", ""))),
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return list(found.values())


def _fallback_answer(citations: dict[str, Citation], *, reason: str = "步数") -> str:
    """预算用尽时的降级答案：把已有证据摊开，并明确声明没有收敛。

    `reason` 必须写实际耗尽的维度（步数 / 时间）：把「时间到了」写成「步数用尽」，
    等于在答案里给判分器和读者喂错的归因——`ours` 的降级文案就是因为这个被修过一次
    （偏差日志第二十八节缺陷 C）。
    """
    if not citations:
        return f"在给定{reason}预算内未能完成检索，无法回答问题。"
    listed = "、".join(list(citations)[:3])
    return f"在给定{reason}预算内未能给出确定答案；已检索到的证据片段：{listed}"


class ReActAgent:
    """单 Agent ReAct 循环：模型 → 工具 → 模型 …… 直到不再请求工具或步数用尽。"""

    def __init__(
        self,
        client: LLMClient,
        gateway: ToolGateway,
        *,
        max_steps: int = MAX_STEPS,
        system: str = REACT_SYSTEM,
        clock: Callable[[], float] = time.monotonic,
        max_wall_seconds: float = MAX_WALL_SECONDS,
    ) -> None:
        self._client = client
        self._gateway = gateway
        self._max_steps = max_steps
        self._system = system
        self._clock = clock
        self._max_wall_seconds = max_wall_seconds

    def run(self, question: str) -> BaselineResult:
        started = self._clock()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        usage = Usage(hit_tokens=0, miss_tokens=0, output_tokens=0)
        citations: dict[str, Citation] = {}
        steps = 0
        tool_calls = 0
        answer = ""
        degraded = False
        exhausted_by: str | None = None

        for _ in range(self._max_steps):
            # 墙钟在每次模型调用**之前**判：判完还调用一次，等于最多超时一秒次调用。
            if self._clock() - started >= self._max_wall_seconds:
                exhausted_by = "时间"
                break
            result = self._client.chat(
                system=self._system, messages=messages, mode="thinking", tools=TOOL_SPECS
            )
            steps += 1
            usage = _add_usage(usage, result.usage)
            # reasoning_content 与 tool_calls 必须原样回传，否则下一轮请求会被服务端拒绝
            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or "",
                    "reasoning_content": result.reasoning,
                    "tool_calls": result.tool_calls,
                }
            )
            if not result.tool_calls:
                answer = (result.text or "").strip()
                break

            for tool_call in result.tool_calls:
                tool_calls += 1
                messages.append(self._execute_tool(tool_call, citations))

        if not answer:
            degraded = True
            answer = _fallback_answer(citations, reason=exhausted_by or "步数")

        return BaselineResult(
            system=SYSTEM,
            question=question,
            answer=answer,
            citations=list(citations.values()),
            steps=steps,
            cost_cny=usage.cost_cny(peak=True),  # 与目标系统同一价格表、同样按高峰价估
            tokens=usage.total_tokens,
            tool_calls=tool_calls,
            degraded=degraded,
            latency_ms=int((self._clock() - started) * 1000),
        )

    def _execute_tool(
        self, tool_call: dict[str, Any], citations: dict[str, Citation]
    ) -> dict[str, Any]:
        """执行一次工具调用，永远返回一条可直接塞回消息列表的 `tool` 消息。

        `AttributeError` 必须在捕获名单里：2026-09-25 主实验实测，模型偶发把
        `arguments` 写成 **JSON 数组**（如 `["阳光电源 净利润"]`）而不是对象，于是
        `args.get(...)` 抛 `AttributeError: 'list' object has no attribute 'get'`。
        它不在原名单里（只有 `ValueError/KeyError/TypeError/ToolResultError`），于是整个
        Baseline A 的那一次运行崩掉：30 次里 3 次（`core-mh-005`/`core-rf-001`/`core-rf-004`）
        落到 `state=error`、`system_cost_cny=0`、回答为空——**钱花了、次数扣了、证据全没了**。
        调用形状坏掉是可预期的模型行为，应当当成一次"工具报错"回给模型让它换路。
        """
        call_id = str(tool_call.get("id", ""))
        function = tool_call.get("function") or {}
        name = str(function.get("name", ""))
        try:
            payload = self._dispatch(name, function.get("arguments") or "{}")
        except (ToolResultError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # 工具不存在 / 参数不是 JSON：把错误当作工具结果回给模型，让它自己换路
            return {"role": "tool", "tool_call_id": call_id, "content": f"错误：{exc}"}
        for citation in collect_payload_citations(payload):
            citations.setdefault(citation.chunk_id, citation)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(payload, ensure_ascii=False),
        }

    def _dispatch(self, name: str, arguments: str) -> Any:
        if name not in TOOL_NAMES:
            raise ValueError(f"未知工具 {name!r}")
        args = json.loads(arguments or "{}")  # 非法 JSON 抛 ValueError，由上层转成 tool 消息
        if not isinstance(args, dict):
            # 模型写错参数形状（数组 / 字符串 / 数字）时给出可读的错误，而不是让它变成
            # `args.get(...)` 上的 AttributeError——后者会穿透上层捕获名单
            raise ValueError(
                f"工具参数必须是 JSON 对象，收到 {type(args).__name__}：{arguments!r}"
            )
        if name == "search_documents":
            return self._gateway.search_documents(
                args.get("query", ""),
                top_k=int(args.get("top_k", 8)),
                doc_filters=args.get("doc_filters"),
            )
        if name == "walk_graph":
            return self._gateway.walk_graph(
                args.get("entity", ""),
                hops=int(args.get("hops", 2)),
                relation_types=args.get("relation_types"),
            )
        if name == "resolve_entity":
            return self._gateway.resolve_entity(
                args.get("mention", ""), context=args.get("context", "")
            )
        if name == "fetch_chunk":
            return self._gateway.fetch_chunk(args["chunk_id"])
        return self._gateway.list_documents()
