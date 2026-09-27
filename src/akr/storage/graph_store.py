"""Neo4j 图存储：参数化 Cypher、关系白名单、实体引用计数。

两条硬约束（对应计划全局约束与 spec §6.2 / §8.3）：
1. 实体名/id、证据文本、doc_id、chunk_id、hops、limit 全部走 Cypher 参数。
   MERGE 关系时关系类型无法参数化，这是唯一允许插值的位置，且必须先过
   _check_relation() 白名单；非法关系在发查询之前就抛 UnknownRelationError。
2. 实体与关系上用 sources 数组记录来源 chunk_id：删除某个 chunk 时只摘掉
   该 chunk_id 的引用，sources 变空（且实体已无剩余关系）才真正删除。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

SUBSIDIARY_OF = "SUBSIDIARY_OF"
SUPPLIES = "SUPPLIES"
CUSTOMER_OF = "CUSTOMER_OF"
COMPETES_WITH = "COMPETES_WITH"
MENTIONS = "MENTIONS"

ALLOWED_RELATIONS: frozenset[str] = frozenset(
    {SUBSIDIARY_OF, SUPPLIES, CUSTOMER_OF, COMPETES_WITH, MENTIONS}
)
MAX_HOPS = 4

# 归一化时抹掉的噪声：空白、半角/全角括号
_NOISE_CHARS = {ord(ch): None for ch in " \t\r\n\u3000()（）"}


class GraphStoreError(RuntimeError):
    """图存储层异常基类。"""


class UnknownRelationError(GraphStoreError, ValueError):
    """关系类型不在 ALLOWED_RELATIONS 白名单内。"""


class ResultLike(Protocol):
    def data(self) -> list[dict[str, Any]]: ...


class SessionLike(Protocol):
    def __enter__(self) -> "SessionLike": ...

    def __exit__(self, *exc: object) -> bool | None: ...

    def run(self, query: str, **params: Any) -> ResultLike: ...


class DriverLike(Protocol):
    def session(self, **kwargs: Any) -> SessionLike: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str = "other"
    aliases: tuple[str, ...] = ()

    def entity_id(self) -> str:
        return make_entity_id(self.name, self.type)


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    relation: str
    target: str
    evidence: str = ""


@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    name: str
    type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationRef:
    relation: str
    source_id: str
    target_id: str


@dataclass(frozen=True)
class GraphPath:
    nodes: tuple[EntityRef, ...]
    relations: tuple[RelationRef, ...]

    @property
    def text(self) -> str:
        """渲染成 `A 公司 -[SUPPLIES]-> B 公司` 的可读路径。

        派生视图而不是字段：存成字段就会与 nodes / relations 产生一致性风险。
        """
        if not self.nodes:
            return ""
        parts = [self.nodes[0].name]
        for relation, node in zip(self.relations, self.nodes[1:]):
            parts.append(f"-[{relation.relation}]-> {node.name}")
        return " ".join(parts)


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    name: str
    type: str
    aliases: tuple[str, ...]
    score: float


def normalize_name(name: str) -> str:
    """去掉空白与括号噪声后小写化，得到稳定可比较的实体名。"""
    if not isinstance(name, str):
        raise TypeError("name 必须是字符串")
    cleaned = name.translate(_NOISE_CHARS).strip().lower()
    if not cleaned:
        raise ValueError("实体名归一后为空")
    return cleaned


def make_entity_id(name: str, type: str) -> str:
    """生成 `type:normalized_name` 形态的稳定实体 id。"""
    return f"{normalize_name(type)}:{normalize_name(name)}"


def _check_relation(relation: str) -> str:
    """关系类型白名单校验：唯一允许插值进 Cypher 的字符串必须过这道闸。"""
    if not isinstance(relation, str) or relation not in ALLOWED_RELATIONS:
        raise UnknownRelationError(
            f"未知关系类型 {relation!r}；允许的关系为 {sorted(ALLOWED_RELATIONS)}"
        )
    return relation


def _deleted(rows: list[dict[str, Any]]) -> int:
    return int(rows[0]["deleted"]) if rows else 0


def _require_single_scope(doc_id: str | None, chunk_id: str | None) -> None:
    if (doc_id is None) == (chunk_id is None):
        raise ValueError("必须且只能指定 doc_id 或 chunk_id 之一")


class GraphStore:
    """Neo4j 封装。driver 由外部注入：单测传 fake，生产用 GraphStore.connect()。"""

    _SCHEMA_STATEMENTS: tuple[str, ...] = (
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_aliases_idx IF NOT EXISTS FOR (e:Entity) ON (e.aliases)",
        "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name, e.aliases]",
    )

    # sources / docs 用 reduce 去重累积：同一实体被同一 chunk 重复抽取也只记一份
    _UPSERT_ENTITY = (
        "MERGE (e:Entity {entity_id: $entity_id}) "
        "ON CREATE SET e.name = $name, e.type = $type, e.aliases = $aliases, "
        "e.sources = [$chunk_id], e.docs = [$doc_id] "
        "ON MATCH SET e.name = $name, e.type = $type, e.aliases = $aliases, "
        "e.sources = reduce(acc = [], s IN e.sources + [$chunk_id] | "
        "CASE WHEN s IN acc THEN acc ELSE acc + s END), "
        "e.docs = reduce(acc = [], d IN e.docs + [$doc_id] | "
        "CASE WHEN d IN acc THEN acc ELSE acc + d END) "
        "RETURN e.entity_id AS entity_id"
    )

    # `__RELATION_TYPE__` 是本模块唯一允许的字符串插值，调用前必须过 _check_relation()。
    # 用 str.replace 而非 str.format：Cypher 里的 map 字面量（如 {entity_id: $entity_id}）
    # 会被 format 当成占位符而抛 KeyError——计划文档里的 .format(relation=...) 写法实测踩了这个坑。
    _UPSERT_RELATION = (
        "MERGE (a:Entity {entity_id: $source_id}) "
        "MERGE (b:Entity {entity_id: $target_id}) "
        "MERGE (a)-[r:`__RELATION_TYPE__`]->(b) "
        "ON CREATE SET r.sources = [$chunk_id], r.docs = [$doc_id], r.evidence = $evidence "
        "ON MATCH SET r.sources = reduce(acc = [], s IN coalesce(r.sources, []) + [$chunk_id] | "
        "CASE WHEN s IN acc THEN acc ELSE acc + s END), "
        "r.docs = reduce(acc = [], d IN coalesce(r.docs, []) + [$doc_id] | "
        "CASE WHEN d IN acc THEN acc ELSE acc + d END), "
        "r.evidence = CASE WHEN coalesce(r.evidence, '') = '' THEN $evidence ELSE r.evidence END "
        "RETURN r.sources AS sources"
    )

    # 删除顺序：先删关系、再删实体（实体只在 sources 为空且无剩余关系时才删）
    _DELETE_ENTITY_SOURCES_BY_CHUNK = (
        "MATCH (e:Entity) WHERE $chunk_id IN coalesce(e.sources, []) "
        "SET e.sources = [s IN coalesce(e.sources, []) WHERE s <> $chunk_id] "
        "WITH e WHERE size(e.sources) = 0 "
        "OPTIONAL MATCH (e)-[r]-() "
        "WITH e, count(r) AS rel_count "
        "WHERE rel_count = 0 "
        "DETACH DELETE e "
        "RETURN count(*) AS deleted"
    )

    _DELETE_ENTITY_SOURCES_BY_DOC = (
        "MATCH (e:Entity) WHERE any(s IN coalesce(e.sources, []) WHERE s STARTS WITH $doc_prefix) "
        "SET e.sources = [s IN coalesce(e.sources, []) WHERE NOT s STARTS WITH $doc_prefix] "
        "WITH e WHERE size(e.sources) = 0 "
        "OPTIONAL MATCH (e)-[r]-() "
        "WITH e, count(r) AS rel_count "
        "WHERE rel_count = 0 "
        "DETACH DELETE e "
        "RETURN count(*) AS deleted"
    )

    _DELETE_RELATIONS_BY_CHUNK = (
        "MATCH ()-[r]->() WHERE $chunk_id IN coalesce(r.sources, []) "
        "SET r.sources = [s IN coalesce(r.sources, []) WHERE s <> $chunk_id] "
        "WITH r WHERE size(r.sources) = 0 "
        "DELETE r "
        "RETURN count(*) AS deleted"
    )

    _DELETE_RELATIONS_BY_DOC = (
        "MATCH ()-[r]->() WHERE any(s IN coalesce(r.sources, []) WHERE s STARTS WITH $doc_prefix) "
        "SET r.sources = [s IN coalesce(r.sources, []) WHERE NOT s STARTS WITH $doc_prefix] "
        "WITH r WHERE size(r.sources) = 0 "
        "DELETE r "
        "RETURN count(*) AS deleted"
    )

    # `*1..4` 里的 4 等于 MAX_HOPS，是源码常量而非用户输入；hops 仍作为参数参与长度过滤
    _WALK_QUERY = (
        "MATCH path = (start:Entity {entity_id: $entity_id})-[rels*1..4]-(other:Entity) "
        "WHERE size(rels) <= $hops AND all(r IN rels WHERE type(r) IN $relation_types) "
        "RETURN size(rels) AS distance, "
        "[n IN nodes(path) | {entity_id: n.entity_id, name: coalesce(n.name, ''), "
        "type: coalesce(n.type, 'other'), aliases: coalesce(n.aliases, [])}] AS nodes, "
        "[r IN relationships(path) | {relation: type(r), "
        "source_id: startNode(r).entity_id, target_id: endNode(r).entity_id}] AS relations "
        "ORDER BY distance "
        "LIMIT $limit"
    )

    # 绑定名不能叫 `$query`：driver 的 `Session.run(self, query, ...)` 首位形参就叫 `query`，
    # `session.run(cypher, query=...)` 会在签名绑定时 TypeError。同理 `$parameters` 也不能用。
    _RESOLVE_QUERY = (
        "CALL db.index.fulltext.queryNodes('entity_fulltext', $search_text) "
        "YIELD node, score "
        "RETURN node.entity_id AS entity_id, coalesce(node.name, '') AS name, "
        "coalesce(node.type, 'other') AS type, coalesce(node.aliases, []) AS aliases, score "
        "ORDER BY score DESC "
        "LIMIT $limit"
    )

    def __init__(self, driver: DriverLike, *, database: str = "neo4j") -> None:
        if not isinstance(database, str) or not database.strip():
            raise ValueError("database 不能为空")
        self._driver = driver
        self._database = database

    @classmethod
    def connect(
        cls, uri: str, user: str, password: str, *, database: str = "neo4j"
    ) -> "GraphStore":
        """延迟 import neo4j，让不依赖 Neo4j 的单测无需启动服务。"""
        try:
            import neo4j
        except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失路径
            raise GraphStoreError("缺少 neo4j 依赖：python -m pip install neo4j") from exc
        return cls(neo4j.GraphDatabase.driver(uri, auth=(user, password)), database=database)

    # --- 执行与 schema ---

    # driver 参数名是外部契约，这里替它挡一道：`query` / `parameters` 是 `Session.run` 的形参名，
    # 同名绑定必定 TypeError。位置参数也刻意不叫 `query`，避免调用方误用关键字形式。
    _RESERVED_PARAMS: frozenset[str] = frozenset({"query", "parameters"})

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        clash = self._RESERVED_PARAMS.intersection(params)
        if clash:
            raise GraphStoreError(
                f"绑定名与 driver 形参冲突：{sorted(clash)}（`Session.run(query, parameters=None, "
                "**kwargs)` 的形参名不能当绑定名用）"
            )
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, **params).data()

    def init_schema(self) -> None:
        """幂等建约束与索引；重复调用不报错。"""
        for statement in self._SCHEMA_STATEMENTS:
            self._run(statement)

    # --- 写入 ---

    def upsert_entity(self, entity: ExtractedEntity, *, doc_id: str, chunk_id: str) -> str:
        entity_id = entity.entity_id()
        rows = self._run(
            self._UPSERT_ENTITY,
            entity_id=entity_id,
            name=entity.name,
            type=entity.type,
            aliases=list(entity.aliases),
            doc_id=doc_id,
            chunk_id=chunk_id,
        )
        return str(rows[0]["entity_id"]) if rows else entity_id

    def upsert_relation(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        *,
        doc_id: str,
        chunk_id: str,
        evidence: str = "",
    ) -> None:
        checked = _check_relation(relation)  # 非法关系在这里就抛，一条查询都不发
        self._run(
            self._UPSERT_RELATION.replace("__RELATION_TYPE__", checked),
            source_id=source_id,
            target_id=target_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            evidence=evidence,
        )

    # --- 删除（引用计数） ---

    def delete_relations_by_source(
        self, *, doc_id: str | None = None, chunk_id: str | None = None
    ) -> int:
        """摘掉指定范围的 chunk 引用；sources 变空的关系才删除。"""
        _require_single_scope(doc_id, chunk_id)
        if chunk_id is not None:
            rows = self._run(self._DELETE_RELATIONS_BY_CHUNK, chunk_id=chunk_id)
        else:
            rows = self._run(self._DELETE_RELATIONS_BY_DOC, doc_prefix=f"{doc_id}#")
        return _deleted(rows)

    def delete_entity_sources(
        self, *, doc_id: str | None = None, chunk_id: str | None = None
    ) -> int:
        """摘掉指定范围的 chunk 引用；sources 变空且已无关系的实体才删除。"""
        _require_single_scope(doc_id, chunk_id)
        if chunk_id is not None:
            rows = self._run(self._DELETE_ENTITY_SOURCES_BY_CHUNK, chunk_id=chunk_id)
        else:
            rows = self._run(self._DELETE_ENTITY_SOURCES_BY_DOC, doc_prefix=f"{doc_id}#")
        return _deleted(rows)

    # --- 查询 ---

    def walk(
        self,
        entity_id: str,
        *,
        hops: int = 2,
        relation_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[GraphPath]:
        if not isinstance(hops, int) or not 1 <= hops <= MAX_HOPS:
            raise ValueError(f"hops 必须是 1..{MAX_HOPS} 之间的整数")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        checked = [_check_relation(r) for r in (relation_types or sorted(ALLOWED_RELATIONS))]
        rows = self._run(
            self._WALK_QUERY,
            entity_id=entity_id,
            hops=hops,
            relation_types=checked,
            limit=limit,
        )
        return [self._to_path(row) for row in rows]

    def resolve(self, mention: str, *, context: str = "", limit: int = 5) -> list[EntityCandidate]:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        search_text = normalize_name(mention)
        if context.strip():
            search_text = f"{search_text} {normalize_name(context)}"
        rows = self._run(self._RESOLVE_QUERY, search_text=search_text, limit=limit)
        return [
            EntityCandidate(
                entity_id=row["entity_id"],
                name=row.get("name", ""),
                type=row.get("type", "other"),
                aliases=tuple(row.get("aliases") or ()),
                score=float(row.get("score", 0.0)),
            )
            for row in rows
        ]

    def count_entities(self) -> int:
        rows = self._run("MATCH (e:Entity) RETURN count(e) AS total")
        return int(rows[0]["total"]) if rows else 0

    def count_relations(self) -> int:
        rows = self._run("MATCH ()-[r]->() RETURN count(r) AS total")
        return int(rows[0]["total"]) if rows else 0

    @staticmethod
    def _to_path(row: dict[str, Any]) -> GraphPath:
        nodes = tuple(
            EntityRef(
                entity_id=node["entity_id"],
                name=node.get("name", ""),
                type=node.get("type", "other"),
                aliases=tuple(node.get("aliases") or ()),
            )
            for node in row.get("nodes") or ()
        )
        relations = tuple(
            RelationRef(
                relation=rel["relation"],
                source_id=rel["source_id"],
                target_id=rel["target_id"],
            )
            for rel in row.get("relations") or ()
        )
        return GraphPath(nodes=nodes, relations=relations)

    def close(self) -> None:
        self._driver.close()
