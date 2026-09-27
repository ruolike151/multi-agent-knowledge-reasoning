from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    """检索结果的唯一类型。

    score 只在同一路检索内部可比；跨路融合必须走 RRF（见 fusion.py）。
    retriever 取值：vector / bm25 / both。
    doc_id / source 是溯源元数据，MCP 工具与引用支撑率统计都依赖它们。
    """

    chunk_id: str
    doc_id: str
    source: str
    text: str
    score: float
    retriever: str
