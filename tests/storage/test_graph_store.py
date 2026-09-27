import inspect
import os

import pytest

from akr.storage.graph_store import (
    MAX_HOPS,
    ExtractedEntity,
    GraphStore,
    GraphStoreError,
    UnknownRelationError,
    make_entity_id,
    normalize_name,
)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return list(self._rows)


class FakeSession:
    """签名严格对齐真实 driver：`Session.run(self, query, parameters=None, **kwargs)`。

    旧 fake 写成 `run(self, query, **params)`，把 2026-09-25 那个存量缺陷放过去了：
    `_run(..., query=..., limit=...)` 在单测里不炸（fake 照收），在真 driver 上必炸
    （`query` 位置实参 + `query` 关键字实参 → TypeError: got multiple values）。fake 的
    签名跟契约不一致，单测就是在验证一个不存在的接口。
    """

    def __init__(self, driver):
        self._driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, parameters=None, **params):
        self._driver.calls.append(
            {"query": query, "params": params, "parameters": parameters}
        )
        rows = self._driver.responses.pop(0) if self._driver.responses else []
        return FakeResult(rows)


class FakeDriver:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])
        self.closed = False

    def session(self, **kwargs):
        return FakeSession(self)

    def close(self):
        self.closed = True


def make_store(*responses):
    driver = FakeDriver(responses)
    return GraphStore(driver), driver


def test_resolve_returns_candidates_and_feeds_fulltext_bindings():
    rows = [
        {
            "entity_id": "company:甲公司",
            "name": "甲公司",
            "type": "company",
            "aliases": ["甲"],
            "score": 1.5,
        }
    ]
    store, driver = make_store(rows)
    candidates = store.resolve("甲公司")
    assert [c.entity_id for c in candidates] == ["company:甲公司"]
    assert candidates[0].aliases == ("甲",)
    assert candidates[0].score == 1.5
    call = driver.calls[0]
    assert "db.index.fulltext.queryNodes('entity_fulltext', $search_text)" in call["query"]
    assert call["params"] == {"search_text": "甲公司", "limit": 5}
    assert call["parameters"] is None


def test_resolve_merges_context_and_validates_limit_before_any_query():
    store, driver = make_store([{"entity_id": "company:甲公司"}])
    store.resolve("甲公司", context="供应商章节", limit=3)
    assert driver.calls[0]["params"] == {"search_text": "甲公司 供应商章节", "limit": 3}
    with pytest.raises(ValueError):
        store.resolve("甲公司", limit=0)
    assert len(driver.calls) == 1


def test_run_refuses_bindings_that_collide_with_driver_keywords():
    store, driver = make_store([{"entity_id": "company:甲公司"}])
    with pytest.raises(GraphStoreError):
        store._run("RETURN 1", query="甲公司")
    with pytest.raises(GraphStoreError):
        store._run("RETURN 1", parameters={"search_text": "甲公司"})
    assert driver.calls == []


def test_every_binding_sent_by_resolve_binds_against_the_real_driver_signature():
    """不连库，只拿真 driver 的签名绑定一次：`query` 作绑定名必炸。"""
    neo4j = pytest.importorskip("neo4j")
    signature = inspect.signature(neo4j.Session.run)
    signature.bind(object(), "CYPHER RETURN 1", search_text="甲公司", limit=5)
    with pytest.raises(TypeError):
        signature.bind(object(), "CYPHER RETURN 1", query="甲公司", limit=5)


def test_normalize_and_entity_id_are_stable():
    assert normalize_name(" 中国平安 保险（集团） ") == "中国平安保险集团"
    assert make_entity_id("中国平安", "company") == "company:中国平安"


def test_unknown_relation_rejected_before_any_query():
    store, driver = make_store()
    with pytest.raises(UnknownRelationError):
        store.upsert_relation(
            "company:a", "DROP_DATABASE", "company:b", doc_id="doc1", chunk_id="doc1#c0000"
        )
    assert driver.calls == []


def test_upsert_relation_parameterizes_values_and_whitelists_type():
    store, driver = make_store([{"sources": ["doc1#c0000"]}])
    store.upsert_relation(
        "company:a",
        "SUPPLIES",
        "company:b",
        doc_id="doc1",
        chunk_id="doc1#c0000",
        evidence="年报披露",
    )
    call = driver.calls[0]
    assert "`SUPPLIES`" in call["query"]
    assert "$source_id" in call["query"]
    assert call["params"]["source_id"] == "company:a"
    assert "company:a" not in call["query"]
    assert "年报披露" not in call["query"]


def test_walk_rejects_hops_over_max_and_unknown_relation_type():
    store, driver = make_store()
    with pytest.raises(ValueError):
        store.walk("company:a", hops=MAX_HOPS + 1)
    with pytest.raises(UnknownRelationError):
        store.walk("company:a", hops=2, relation_types=["DROP"])
    assert driver.calls == []


def test_walk_builds_paths_from_records():
    rows = [
        {
            "nodes": [
                {"entity_id": "company:a", "name": "A 公司", "type": "company", "aliases": []},
                {"entity_id": "company:b", "name": "B 公司", "type": "company", "aliases": ["B"]},
            ],
            "relations": [
                {"relation": "SUPPLIES", "source_id": "company:a", "target_id": "company:b"}
            ],
        }
    ]
    store, _ = make_store(rows)
    paths = store.walk("company:a", hops=2)
    assert paths[0].text == "A 公司 -[SUPPLIES]-> B 公司"
    assert paths[0].relations[0].target_id == "company:b"


def test_upsert_entity_keeps_sources_accumulated():
    store, driver = make_store([{"entity_id": "company:a"}])
    entity = ExtractedEntity("甲公司", "company", aliases=("甲",))
    assert store.upsert_entity(entity, doc_id="doc1", chunk_id="doc1#c0000") == "company:a"
    assert driver.calls[0]["params"]["chunk_id"] == "doc1#c0000"
    assert "reduce(" in driver.calls[0]["query"]  # sources 去重累积


def test_delete_entity_sources_returns_deleted_count():
    store, driver = make_store([{"deleted": 2}])
    assert store.delete_entity_sources(chunk_id="doc1#c0000") == 2
    assert "$chunk_id" in driver.calls[0]["query"]


def test_delete_requires_exactly_one_scope():
    store, _ = make_store()
    with pytest.raises(ValueError):
        store.delete_entity_sources()
    with pytest.raises(ValueError):
        store.delete_entity_sources(doc_id="doc1", chunk_id="doc1#c0000")


def test_delete_by_doc_uses_prefix_scope():
    store, driver = make_store([{"deleted": 0}])
    assert store.delete_entity_sources(doc_id="doc1") == 0
    call = driver.calls[0]
    assert call["params"]["doc_prefix"] == "doc1#"
    assert "STARTS WITH $doc_prefix" in call["query"]


@pytest.mark.integration
def test_graph_store_roundtrip_against_local_neo4j():
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        pytest.skip("需要 NEO4J_URI（先 docker compose up -d neo4j）")
    store = GraphStore.connect(
        uri,
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "akr-local-pass"),
    )
    try:
        store.init_schema()
        store.upsert_entity(ExtractedEntity("甲公司", "company"), doc_id="d1", chunk_id="d1#c0001")
        store.upsert_entity(ExtractedEntity("乙公司", "company"), doc_id="d1", chunk_id="d1#c0001")
        store.upsert_relation(
            "company:甲公司", "SUPPLIES", "company:乙公司", doc_id="d1", chunk_id="d1#c0001"
        )
        paths = store.walk(make_entity_id("甲公司", "company"), hops=1)
        assert paths and paths[0].relations[0].relation == "SUPPLIES"
        candidates = store.resolve("甲公司")
        assert candidates and candidates[0].entity_id == make_entity_id("甲公司", "company")
        # 先删只被该 chunk 引用的关系，再删失去全部引用的实体（spec §8.3 第 5 项的顺序）
        assert store.delete_relations_by_source(chunk_id="d1#c0001") == 1
        assert store.delete_entity_sources(chunk_id="d1#c0001") == 2
    finally:
        store.close()
