import pytest

from akr.ingest.chunker import Chunk
from akr.storage import sqlite_store as store


def make_chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c{ordinal:04d}",
        doc_id=doc_id,
        source="a.pdf",
        ordinal=ordinal,
        text=text,
        content_hash=f"hash{ordinal:04d}",
        section_path="第一章",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def make_conn(tmp_path):
    conn = store.connect(tmp_path / "akr.sqlite")
    store.init_schema(conn)
    store.upsert_document(
        conn, doc_id="doc1", source="a.pdf", doc_type="annual_report", page_count=3,
        updated_at="2026-09-23T00:00:00",
    )
    return conn


def test_upsert_document_is_idempotent(tmp_path):
    conn = make_conn(tmp_path)
    store.upsert_document(
        conn, doc_id="doc1", source="a.pdf.gz", doc_type="annual_report", page_count=4,
        updated_at="2026-09-24T00:00:00",
    )
    rows = store.list_documents(conn)
    assert len(rows) == 1
    assert rows[0].source == "a.pdf.gz"
    assert rows[0].page_count == 4


def test_replace_chunks_drops_previous_rows(tmp_path):
    conn = make_conn(tmp_path)
    store.replace_chunks(conn, "doc1", [make_chunk("doc1", 0, "旧"), make_chunk("doc1", 1, "旧二")])
    store.replace_chunks(conn, "doc1", [make_chunk("doc1", 0, "新")])
    chunks = store.list_chunks(conn, "doc1")
    assert [c.text for c in chunks] == ["新"]


def test_list_chunks_is_ordered_by_ordinal_and_round_trips_fields(tmp_path):
    conn = make_conn(tmp_path)
    store.replace_chunks(conn, "doc1", [make_chunk("doc1", 1, "后"), make_chunk("doc1", 0, "前")])
    chunks = store.list_chunks(conn, "doc1")
    assert [c.ordinal for c in chunks] == [0, 1]
    assert chunks[0].page_start == 1
    assert chunks[0].section_path == "第一章"


def test_document_overview_counts_chunks(tmp_path):
    conn = make_conn(tmp_path)
    store.replace_chunks(conn, "doc1", [make_chunk("doc1", 0, "a"), make_chunk("doc1", 1, "b")])
    rows = store.list_documents(conn)
    assert rows[0].chunk_count == 2
    assert rows[0].doc_id == "doc1"


def test_costs_are_aggregated_per_run(tmp_path):
    conn = make_conn(tmp_path)
    for run_id, node, cost in (("r1", "plan", 0.2), ("r1", "verify", 0.1), ("r2", "plan", 1.0)):
        store.add_cost(
            conn, run_id=run_id, node=node, cost_cny=cost,
            hit_tokens=10, miss_tokens=20, output_tokens=30,
            created_at="2026-09-23T00:00:00",
        )
    assert store.total_cost_cny(conn, run_id="r1") == pytest.approx(0.3)
    assert store.total_cost_cny(conn) == pytest.approx(1.3)


def test_deleting_document_cascades_chunks(tmp_path):
    conn = make_conn(tmp_path)
    store.replace_chunks(conn, "doc1", [make_chunk("doc1", 0, "a")])
    conn.execute("DELETE FROM documents WHERE doc_id = ?", ("doc1",))
    conn.commit()
    assert store.list_chunks(conn, "doc1") == []
