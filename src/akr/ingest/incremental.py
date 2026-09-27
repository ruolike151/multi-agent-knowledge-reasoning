"""增量入库：spec §8.3 六项的实现。

核心不变式：`chunk_id` 由内容决定（Task 10），所以「内容没变的 chunk 重新解析后 id 不变」，
差量集合天然成立；本模块只负责把差量落到四个存储上，并把量化结果回传给评测层。

六项落点：指纹（content_hash 二次校验）、结构树（align_sections）、差量集合（ChunkDiff）、
增量写入（只碰 diff.added / diff.removed）、实体引用计数（删除顺序固定）、量化（IngestResult）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from akr.ingest.chunker import Chunk, chunk_pages
from akr.ingest.extract import extract_chunk, write_extraction
from akr.ingest.parser import ParsedPage
from akr.model.client import LLMClient
from akr.storage import sqlite_store
from akr.storage.artifact_store import Artifact
from akr.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkDiff:
    added: tuple[Chunk, ...]
    removed: tuple[Chunk, ...]
    unchanged: tuple[Chunk, ...]


@dataclass(frozen=True)
class SectionDiff:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    total_chunks: int
    added: int
    removed: int
    unchanged: int
    embedded: int
    extracted: int
    dropped_relations: int
    added_chunk_ids: tuple[str, ...]
    removed_chunk_ids: tuple[str, ...]
    sections: SectionDiff
    elapsed_ms: float

    @property
    def reembed_ratio(self) -> float:
        """重新向量化的比例；对照基准是文档级全量重建（= 1.0）。"""
        return 0.0 if not self.total_chunks else self.embedded / self.total_chunks

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "total_chunks": self.total_chunks,
            "added": self.added,
            "removed": self.removed,
            "unchanged": self.unchanged,
            "embedded": self.embedded,
            "extracted": self.extracted,
            "dropped_relations": self.dropped_relations,
            "reembed_ratio": self.reembed_ratio,
            "sections_added": list(self.sections.added),
            "sections_removed": list(self.sections.removed),
            "elapsed_ms": self.elapsed_ms,
        }


def diff_chunks(old: Sequence[Chunk], new: Sequence[Chunk]) -> ChunkDiff:
    """按内容寻址的 chunk_id 求差量，并用 content_hash 做二次校验（spec §8.3 第 1、3 项）。"""
    old_by_id = {c.chunk_id: c for c in old}
    new_by_id = {c.chunk_id: c for c in new}

    # id 相同但指纹不同 → 视为「换了内容的同一个位置」，走先删后加，绝不当作 unchanged
    mismatched = {
        chunk_id
        for chunk_id in old_by_id.keys() & new_by_id.keys()
        if old_by_id[chunk_id].content_hash != new_by_id[chunk_id].content_hash
    }
    if mismatched:
        logger.warning("chunk_id 相同但 content_hash 不同，按变更处理：%s", sorted(mismatched))

    added = tuple(
        new_by_id[chunk_id]
        for chunk_id in new_by_id
        if chunk_id not in old_by_id or chunk_id in mismatched
    )
    removed = tuple(
        old_by_id[chunk_id]
        for chunk_id in old_by_id
        if chunk_id not in new_by_id or chunk_id in mismatched
    )
    unchanged = tuple(
        new_by_id[chunk_id]
        for chunk_id in new_by_id
        if chunk_id in old_by_id and chunk_id not in mismatched
    )
    return ChunkDiff(added=added, removed=removed, unchanged=unchanged)


def align_sections(old: Sequence[Chunk], new: Sequence[Chunk]) -> SectionDiff:
    """文档结构树对齐（spec §8.3 第 2 项）：按 section_path 集合给出结构变更。"""
    old_paths = {c.section_path for c in old if c.section_path}
    new_paths = {c.section_path for c in new if c.section_path}
    return SectionDiff(
        added=tuple(sorted(new_paths - old_paths)),
        removed=tuple(sorted(old_paths - new_paths)),
        unchanged=tuple(sorted(new_paths & old_paths)),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ingest_document(
    *,
    doc_id: str,
    source: str,
    pages: Sequence[ParsedPage],
    conn,
    artifacts,
    vectors,
    bm25,
    graph: GraphStore,
    client: LLMClient | None = None,
    chunker: Callable[..., list[Chunk]] = chunk_pages,
) -> IngestResult:
    """把一份文档的解析结果增量写入四个存储，返回量化结果。

    顺序固定为「先删、再增、最后落分块表」：先删避免 removed chunk 的旧数据留在索引里被检索到；
    `client=None` 时跳过抽取（只重建索引的场景，例如改分块参数后回归）。
    """
    started = time.perf_counter()
    new_chunks = chunker(pages, doc_id=doc_id, source=source)
    old_chunks = sqlite_store.list_chunks(conn, doc_id)
    diff = diff_chunks(old_chunks, new_chunks)
    sections = align_sections(old_chunks, new_chunks)

    removed_ids = [c.chunk_id for c in diff.removed]
    if removed_ids:
        vectors.delete_ids(removed_ids)
        bm25.remove(removed_ids)
        for chunk in diff.removed:
            artifacts.delete(chunk.doc_id, chunk.chunk_id)
            # 顺序不可交换：先摘关系引用，再摘实体引用（Task 17 的引用计数不变式）
            graph.delete_relations_by_source(chunk_id=chunk.chunk_id)
            graph.delete_entity_sources(chunk_id=chunk.chunk_id)

    if diff.added:
        for chunk in diff.added:
            artifacts.put(
                Artifact(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    char_range=(chunk.char_start, chunk.char_end),
                    page=chunk.page_start,
                )
            )
        vectors.add(list(diff.added))
        bm25.upsert(list(diff.added))

    extracted = 0
    dropped_relations = 0
    if client is not None:
        for chunk in diff.added:
            extraction = extract_chunk(client, chunk)
            _, relations_written = write_extraction(
                graph, extraction, doc_id=doc_id, chunk_id=chunk.chunk_id
            )
            extracted += 1
            dropped_relations += len(extraction.relations) - relations_written

    # 文档行必须先于分块表写：chunks.doc_id 外键指向 documents.doc_id，
    # 首次入库（documents 里还没有这一行）时顺序反了会直接 IntegrityError。
    sqlite_store.upsert_document(
        conn,
        doc_id=doc_id,
        source=source,
        doc_type="annual_report",
        page_count=len(pages),
        updated_at=_now(),
    )
    # 分块表最后替换，且 replace_chunks 内部是单事务：中途失败时旧的分块表仍完整，
    # 重跑本函数即可恢复（不需要先清空再灌）。
    sqlite_store.replace_chunks(conn, doc_id, list(new_chunks))

    return IngestResult(
        doc_id=doc_id,
        total_chunks=len(new_chunks),
        added=len(diff.added),
        removed=len(diff.removed),
        unchanged=len(diff.unchanged),
        embedded=len(diff.added),
        extracted=extracted,
        dropped_relations=dropped_relations,
        added_chunk_ids=tuple(c.chunk_id for c in diff.added),
        removed_chunk_ids=tuple(removed_ids),
        sections=sections,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def rebuild_all(
    *, documents: Sequence[tuple[str, str, Sequence[ParsedPage]]], **kwargs
) -> list[IngestResult]:
    """文档级全量重建对照臂：同一批文档一次性灌库（spec §8.3 第 6 项的对照基准）。

    本身就是循环调用 ingest_document，不含额外逻辑；因为 chunk_id 内容寻址，
    对照臂的「重新向量化比例」天然恒为 1.0。
    """
    return [
        ingest_document(doc_id=doc_id, source=source, pages=pages, **kwargs)
        for doc_id, source, pages in documents
    ]
