"""chunk 级实体与关系抽取。

模型输出一律当作不可信输入：关系类型必须落在 ALLOWED_RELATIONS 白名单内，
端点名必须映射成 Task 17 定义的稳定实体 id；写库遵循「先实体、后关系」。
"""

from __future__ import annotations

import logging
from typing import Callable, cast

from pydantic import BaseModel, Field

from akr.ingest.chunker import Chunk
from akr.model.client import LLMClient
from akr.model.routing import mode_for
from akr.model.structured import parse_with_repair
from akr.storage.graph_store import (
    ALLOWED_RELATIONS,
    ExtractedEntity,
    GraphStore,
    make_entity_id,
    normalize_name,
)

logger = logging.getLogger(__name__)

# 稳定前缀放 system，动态内容（章节路径 + chunk 正文）放 user：前缀缓存友好（spec §7.5）
EXTRACT_SYSTEM = """你是企业年报的信息抽取器，只输出 JSON。

要求：
1. 只抽取正文中明确出现的事实，不要推断、不要补全。
2. relation 只能取以下五个值之一：
   SUBSIDIARY_OF（母子公司）、SUPPLIES（供应）、CUSTOMER_OF（客户）、
   COMPETES_WITH（竞争）、MENTIONS（同段共现但无明确关系）。
3. 每条 relation 必须给 evidence，evidence 是正文中支撑该关系的原文片段。
4. 输出结构：{"entities": [{"name","type","aliases"}],
             "relations": [{"source","relation","target","evidence"}]}
   type 取 company / product / person / regulation / other。找不到内容时输出空数组。"""


class EntityDraft(BaseModel):
    name: str
    type: str = "other"
    aliases: list[str] = Field(default_factory=list)


class RelationDraft(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str = ""


class ExtractionResult(BaseModel):
    entities: list[EntityDraft] = Field(default_factory=list)
    relations: list[RelationDraft] = Field(default_factory=list)


def extract_chunk(
    client: LLMClient,
    chunk: Chunk,
    *,
    repair: Callable[[str], str] | None = None,
) -> ExtractionResult:
    """抽取单个 chunk。节点名 `extract` 决定 thinking / non-thinking 模式。"""
    llm = client.chat(
        system=EXTRACT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"章节路径：{chunk.section_path}\n页码：{chunk.page_start}\n\n{chunk.text}",
            }
        ],
        mode=mode_for("extract"),
        json_object=True,
    )
    return cast(ExtractionResult, parse_with_repair(llm.text, ExtractionResult, repair=repair))


def resolve_endpoint(name: str, by_name: dict[str, str]) -> str:
    """把关系端点名映射成实体 id；未在实体列表中出现则回退 type='other' 并告警。"""
    key = normalize_name(name)
    if key in by_name:
        return by_name[key]
    logger.warning("关系端点 %r 未在实体列表中出现，回退 type='other'", name)
    return make_entity_id(name, "other")


def write_extraction(
    store: GraphStore,
    result: ExtractionResult,
    *,
    doc_id: str,
    chunk_id: str,
) -> tuple[int, int]:
    """写入实体与关系，返回 (entities_written, relations_written)。

    先实体后关系：upsert_relation 会 MERGE 端点节点，顺序反了会留下没有 sources 的裸节点。
    """
    by_name: dict[str, str] = {}
    entities_written = 0
    for draft in result.entities:
        if not draft.name.strip():
            logger.warning("跳过空名实体（chunk_id=%s）", chunk_id)
            continue
        entity = ExtractedEntity(
            name=draft.name,
            type=draft.type or "other",
            aliases=tuple(alias for alias in draft.aliases if alias.strip()),
        )
        entity_id = store.upsert_entity(entity, doc_id=doc_id, chunk_id=chunk_id)
        by_name[normalize_name(draft.name)] = entity_id
        # 别名也登记：主名优先（setdefault），保证「甲公司 / 甲」指到同一个 id
        for alias in entity.aliases:
            by_name.setdefault(normalize_name(alias), entity_id)
        entities_written += 1

    relations_written = 0
    for draft in result.relations:
        if draft.relation not in ALLOWED_RELATIONS:
            # 白名单外的关系类型必须留痕：Task 19 的量化对照要统计这类丢弃量
            logger.warning(
                "丢弃白名单外关系类型 %r（chunk_id=%s，%s -> %s）",
                draft.relation,
                chunk_id,
                draft.source,
                draft.target,
            )
            continue
        store.upsert_relation(
            resolve_endpoint(draft.source, by_name),
            draft.relation,
            resolve_endpoint(draft.target, by_name),
            doc_id=doc_id,
            chunk_id=chunk_id,
            evidence=draft.evidence,
        )
        relations_written += 1

    return entities_written, relations_written
