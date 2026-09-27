"""MCP 工具的输入输出契约。字段名与 spec §6.1 严格对应，是 Agent 侧唯一的解析依据。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    score: float
    retriever: Literal["vector", "bm25", "both"]


class SearchResult(BaseModel):
    hits: list[Hit] = Field(default_factory=list)
    fusion: Literal["rrf"] = "rrf"
    retriever_stats: dict[str, int] = Field(default_factory=dict)


class EntityRefOut(BaseModel):
    entity_id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)


class RelRefOut(BaseModel):
    relation: str
    source_id: str
    target_id: str


class GraphPathOut(BaseModel):
    nodes: list[EntityRefOut] = Field(default_factory=list)
    relations: list[RelRefOut] = Field(default_factory=list)
    text: str = ""


class GraphResult(BaseModel):
    paths: list[GraphPathOut] = Field(default_factory=list)


class EntityCandidateOut(BaseModel):
    entity_id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    score: float


class EntityResult(BaseModel):
    candidates: list[EntityCandidateOut] = Field(default_factory=list)


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    text: str
    char_range: list[int] = Field(default_factory=list)


class DocumentItem(BaseModel):
    doc_id: str
    source: str
    doc_type: str
    chunk_count: int
    updated_at: str


class DocumentOverview(BaseModel):
    documents: list[DocumentItem] = Field(default_factory=list)


# MCP tool schema：Task 21 的 server 直接用这份声明注册工具，避免手写两份定义
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_documents",
        "description": "混合检索（向量 + BM25，RRF 融合），返回带 chunk_id 与 source 的片段。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 8},
                "doc_filters": {
                    "type": ["object", "null"],
                    "description": "目前仅支持 {'doc_ids': [...]}",
                },
                "pool_depth": {
                    "type": "integer",
                    # 与 `akr.mcp_server.tools.POOL_DEPTH_DEFAULT` 同步（这里不能 import：tools 反向
                    # 依赖本模块）；`tests/mcp_server/test_tools.py` 有一条等式把两处钉在一起。
                    "default": 50,
                    "minimum": 0,
                    "description": (
                        "第一段融合的候选池深；0 = 取 top_k 条（改动前的历史行为），"
                        "50 = 2026-09-26 第二次实测起的线上口径（与 rrf_k=20 一起上线的）"
                    ),
                },
                "rerank": {
                    "type": "string",
                    # 与 `akr.mcp_server.tools.RERANK_MODES` 同步（这里不能 import：tools 反向
                    # 依赖本模块）；`tests/mcp_server/test_tools.py` 有一条等式把两处钉在一起。
                    "enum": ["rrf", "cosine", "cross"],
                    "default": "cross",
                    "description": (
                        "融合后的排序模式；cross = cross-encoder 重排候选池尾部（2026-09-26 起的"
                        "线上默认，每次检索一次 /v1/rerank），rrf = 历史行为（不发网络请求）"
                    ),
                },
                "rerank_alpha": {
                    "type": "number",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "rerank=cosine 时余弦先验的权重（0 = 等价于 rrf，1 = 只看余弦）",
                },
                "rrf_k": {
                    "type": "integer",
                    # 与 `akr.mcp_server.tools.RRF_K_DEFAULT` 同步（这里不能 import：tools 反向
                    # 依赖本模块）；`tests/mcp_server/test_tools.py` 有一条等式把两处钉在一起。
                    "default": 20,
                    "minimum": 1,
                    "description": (
                        "第一段融合的 RRF 常数 k（1/(k+名次)）；60 = 改动前的历史行为，"
                        "20 = 2026-09-26 起的线上口径（池深 50 必须配它，见 tools.RRF_K_DEFAULT）"
                    ),
                },
                "rerank_lock": {
                    "type": "integer",
                    # 与 `akr.mcp_server.tools.RERANK_LOCK_DEFAULT` 同步（同上）。
                    "default": 4,
                    "minimum": 0,
                    "description": (
                        "rerank=cross 时由 RRF 独占的前 N 条（0 = 整池重排 = 已实测的负结果）；"
                        "只在 rerank='cross' 时可用，且不超过 top_k"
                    ),
                },
                "expand_neighbors": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": (
                        "同一文档内按 ordinal 向两侧扩邻块的宽度（0 = 关闭）；"
                        "只对返回列表前 3 条命中生效，扩完仍截回 top_k"
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "walk_graph",
        "description": "从实体出发做多跳关系遍历，返回路径与关系，用于多跳推理。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "hops": {"type": "integer", "default": 2, "minimum": 1, "maximum": 4},
                "relation_types": {
                    "type": ["array", "null"],
                    "items": {
                        "type": "string",
                        "enum": [
                            "SUBSIDIARY_OF",
                            "SUPPLIES",
                            "CUSTOMER_OF",
                            "COMPETES_WITH",
                            "MENTIONS",
                        ],
                    },
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "resolve_entity",
        "description": "把别名或简称解析成知识库中的规范实体。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mention": {"type": "string"},
                "context": {"type": "string", "default": ""},
            },
            "required": ["mention"],
        },
    },
    {
        "name": "fetch_chunk",
        "description": "按 chunk_id 取回原文片段与溯源信息。",
        "inputSchema": {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
    },
    {
        "name": "list_documents",
        "description": "列出知识库中的文档概览（doc_id / source / chunk 数 / 更新时间）。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]
