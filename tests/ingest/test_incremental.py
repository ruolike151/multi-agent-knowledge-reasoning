import json
from dataclasses import replace

from akr.ingest.chunker import Chunk, content_hash
from akr.ingest.incremental import align_sections, diff_chunks, ingest_document
from akr.ingest.parser import ParsedPage
from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.storage import sqlite_store


def page(text, number=1):
    return ParsedPage(page=number, text=text, tables=[])


# 真实年报单页在 2000 字量级，远超 chunker 的 max_chars=800。用短文测「改一页只重嵌受影响
# chunk」会失效：两页会被合并成同一个 chunk，改一页等于改整篇（计划文档此处用了短文，实测如此）。
_PAD = "本段为测试填充内容，用于把两页拆到不同的 chunk 中。"


def long_page(marker, number=1, repeats=40):
    return page(marker + _PAD * repeats, number)


def chunk(text, *, doc_id="d1", section="", ordinal=0):
    digest = content_hash(text)
    return Chunk(
        chunk_id=f"{doc_id}#c{digest}",
        doc_id=doc_id,
        source="s.pdf",
        ordinal=ordinal,
        text=text,
        content_hash=digest,
        section_path=section,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


class FakeArtifacts:
    def __init__(self):
        self.puts = []
        self.deletes = []

    def put(self, artifact):
        self.puts.append(artifact.chunk_id)
        return f"{artifact.doc_id}/{artifact.chunk_id}.txt"

    def delete(self, doc_id, chunk_id):
        self.deletes.append(chunk_id)


class FakeVectors:
    def __init__(self):
        self.added = []
        self.deleted = []

    def add(self, chunks):
        self.added.extend(c.chunk_id for c in chunks)
        return len(chunks)

    def delete_ids(self, chunk_ids):
        self.deleted.extend(chunk_ids)
        return len(chunk_ids)


class FakeBm25:
    def __init__(self):
        self.upserts = []
        self.removed = []

    def upsert(self, chunks):
        self.upserts.extend(c.chunk_id for c in chunks)
        return len(chunks)

    def remove(self, chunk_ids):
        self.removed.extend(chunk_ids)
        return len(chunk_ids)


class FakeGraph:
    def __init__(self):
        self.order = []
        self.writes = []

    def delete_relations_by_source(self, *, doc_id=None, chunk_id=None):
        self.order.append(("relations", chunk_id))
        return 1

    def delete_entity_sources(self, *, doc_id=None, chunk_id=None):
        self.order.append(("entities", chunk_id))
        return 2

    def upsert_entity(self, entity, *, doc_id, chunk_id):
        self.writes.append(("entity", entity.entity_id(), chunk_id))
        return entity.entity_id()

    def upsert_relation(self, source_id, relation, target_id, *, doc_id, chunk_id, evidence=""):
        self.writes.append(("relation", relation, chunk_id))


class FakeClient:
    def __init__(self, payload):
        self._payload = payload

    def chat(self, **kwargs):
        return LLMResult(
            text=json.dumps(self._payload, ensure_ascii=False),
            reasoning="",
            usage=Usage(hit_tokens=0, miss_tokens=8, output_tokens=4),
            mode=kwargs["mode"],
        )


def make_stores():
    conn = sqlite_store.connect(":memory:")
    sqlite_store.init_schema(conn)
    artifacts, vectors, bm25, graph = FakeArtifacts(), FakeVectors(), FakeBm25(), FakeGraph()
    return conn, artifacts, vectors, bm25, graph


def ingest(pages, stores, *, doc_id="d1", client=None):
    conn, artifacts, vectors, bm25, graph = stores
    return ingest_document(
        doc_id=doc_id,
        source="s.pdf",
        pages=pages,
        conn=conn,
        artifacts=artifacts,
        vectors=vectors,
        bm25=bm25,
        graph=graph,
        client=client,
    )


def test_diff_chunks_splits_by_content_addressed_id():
    old = [chunk("A", ordinal=0), chunk("B", ordinal=1)]
    new = [chunk("B", ordinal=0), chunk("C", ordinal=1)]
    diff = diff_chunks(old, new)
    assert [c.text for c in diff.added] == ["C"]
    assert [c.text for c in diff.removed] == ["A"]
    assert [c.text for c in diff.unchanged] == ["B"]


def test_diff_chunks_flags_hash_mismatch_as_change():
    old = [chunk("A")]
    tampered = replace(old[0], content_hash="deadbeefdeadbeef")
    diff = diff_chunks(old, [tampered])
    assert len(diff.added) == 1 and len(diff.removed) == 1 and not diff.unchanged


def test_align_sections_reports_structure_change():
    old = [chunk("A", section="第四节 经营情况"), chunk("B", section="第五节 财务报告")]
    new = [
        chunk("A", section="第四节 经营情况"),
        chunk("C", section="第五节 财务报告 > 主要子公司"),
    ]
    sections = align_sections(old, new)
    assert sections.added == ("第五节 财务报告 > 主要子公司",)
    assert sections.removed == ("第五节 财务报告",)
    assert sections.unchanged == ("第四节 经营情况",)


def test_first_ingest_embeds_everything():
    stores = make_stores()
    result = ingest([page("甲公司注册资本 100 亿元。"), page("乙公司是最大客户。", 2)], stores)
    assert result.total_chunks > 0
    assert result.embedded == result.total_chunks == len(result.added_chunk_ids)
    assert result.reembed_ratio == 1.0


def test_reingest_same_pages_embeds_nothing():
    stores = make_stores()
    pages = [page("甲公司注册资本 100 亿元。"), page("乙公司是最大客户。", 2)]
    first = ingest(pages, stores)
    second = ingest(pages, stores)
    assert second.embedded == 0
    assert second.added == 0 and second.removed == 0
    assert second.unchanged == first.total_chunks
    assert second.reembed_ratio == 0.0


def test_page_edit_only_reembeds_affected_chunks():
    stores = make_stores()
    first = ingest([long_page("甲公司注册资本 100 亿元。", 1), long_page("乙公司是最大客户。", 2)], stores)
    edited = ingest([long_page("甲公司注册资本 100 亿元。", 1), long_page("丙公司是最大客户。", 2)], stores)
    assert 0 < edited.embedded < first.total_chunks
    assert edited.embedded == len(edited.added_chunk_ids)
    assert edited.embedded + edited.unchanged == edited.total_chunks


def test_removed_chunks_are_deleted_and_graph_deletes_relations_first():
    stores = make_stores()
    conn, artifacts, vectors, bm25, graph = stores
    ingest([page("甲公司注册资本 100 亿元。"), page("乙公司是最大客户。", 2)], stores)
    second = ingest([page("甲公司注册资本 100 亿元。")], stores)
    assert second.removed_chunk_ids
    assert set(vectors.deleted) == set(second.removed_chunk_ids)
    assert set(bm25.removed) == set(second.removed_chunk_ids)
    assert set(artifacts.deletes) == set(second.removed_chunk_ids)
    assert graph.order == [
        (kind, chunk_id)
        for chunk_id in second.removed_chunk_ids
        for kind in ("relations", "entities")
    ]


def test_added_chunks_are_extracted_when_client_provided():
    stores = make_stores()
    payload = {"entities": [{"name": "甲公司", "type": "company"}], "relations": []}
    result = ingest([page("甲公司注册资本 100 亿元。")], stores, client=FakeClient(payload))
    assert result.extracted == len(result.added_chunk_ids)
    assert any(write[0] == "entity" for write in stores[4].writes)


def test_report_exposes_quantified_reduction():
    stores = make_stores()
    ingest([long_page("甲公司注册资本 100 亿元。", 1), long_page("乙公司是最大客户。", 2)], stores)
    edited = ingest([long_page("甲公司注册资本 100 亿元。", 1), long_page("丙公司是最大客户。", 2)], stores)
    report = edited.to_dict()
    assert report["embedded"] < report["total_chunks"]
    assert 0.0 <= report["reembed_ratio"] < 1.0
    assert report["elapsed_ms"] >= 0.0
