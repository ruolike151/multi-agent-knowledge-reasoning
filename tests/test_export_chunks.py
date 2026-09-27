"""`scripts/export_chunks.py` 的单测。

关键契约不在本脚本自己身上，而在下游 `eval.taskset.build_candidates.read_chunks`：
导出的每一行必须能直接被它读进去，所以这里有一条跨模块的联调用例。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from akr.ingest.chunker import Chunk
from akr.storage import sqlite_store
from eval.taskset.build_candidates import read_chunks
from eval.taskset.schema import TasksetError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_chunks.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_chunks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_chunks = _load()


def _make_chunk(doc_id: str, ordinal: int, text: str = "正文") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{ordinal:04d}",
        doc_id=doc_id,
        source=f"data/corpus/annual/{doc_id}.pdf",
        ordinal=ordinal,
        text=text,
        content_hash=f"h{ordinal}",
        section_path="第三节 管理层讨论与分析",
        page_start=ordinal + 1,
        page_end=ordinal + 1,
        char_start=ordinal * 100,
        char_end=ordinal * 100 + len(text),
    )


@pytest.fixture()
def seeded_db(tmp_path):
    path = tmp_path / "akr.sqlite"
    conn = sqlite_store.connect(path)
    sqlite_store.init_schema(conn)
    for doc_id, count in (("annual-600438-2025", 3), ("ann-002129-ir", 2)):
        sqlite_store.upsert_document(
            conn,
            doc_id=doc_id,
            source=f"data/corpus/annual/{doc_id}.pdf",
            doc_type="annual_report",
            page_count=1,
            updated_at="2026-09-25T00:00:00+00:00",
        )
        sqlite_store.replace_chunks(
            conn,
            doc_id,
            [_make_chunk(doc_id, i, f"{doc_id} 第 {i} 段") for i in range(count)],
        )
    conn.close()
    return path


def test_export_is_readable_by_build_candidates(seeded_db, tmp_path, capsys):
    out = tmp_path / "chunks.jsonl"
    assert export_chunks.main(["--db", str(seeded_db), "--out", str(out)]) == 0
    rows = read_chunks(out)
    assert len(rows) == 5
    assert rows[0]["chunk_id"] == "ann-002129-ir#0000"
    assert rows[0]["text"].endswith("第 0 段")
    assert all({"chunk_id", "doc_id", "source", "text"} <= set(row) for row in rows)
    assert "共 5 条 / 2 份文档" in capsys.readouterr().out


def test_export_is_byte_for_byte_repeatable(seeded_db, tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert export_chunks.main(["--db", str(seeded_db), "--out", str(first)]) == 0
    assert export_chunks.main(["--db", str(seeded_db), "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()


def test_export_filters_by_doc_id_and_limit(seeded_db, tmp_path):
    out = tmp_path / "head.jsonl"
    assert (
        export_chunks.main(
            [
                "--db",
                str(seeded_db),
                "--out",
                str(out),
                "--doc-id",
                "annual-600438-2025",
                "--limit-per-doc",
                "2",
            ]
        )
        == 0
    )
    rows = read_chunks(out)
    assert [row["chunk_id"] for row in rows] == ["annual-600438-2025#0000", "annual-600438-2025#0001"]


def test_missing_db_fails_fast(tmp_path, capsys):
    assert export_chunks.main(["--db", str(tmp_path / "nope.sqlite")]) == 1
    assert "库不存在" in capsys.readouterr().err


def test_empty_db_fails_fast(tmp_path, capsys):
    path = tmp_path / "empty.sqlite"
    conn = sqlite_store.connect(path)
    sqlite_store.init_schema(conn)
    conn.close()
    assert export_chunks.main(["--db", str(path), "--out", str(tmp_path / "out.jsonl")]) == 1
    assert "没有任何分块" in capsys.readouterr().err


def test_unknown_doc_id_fails_fast(seeded_db, tmp_path, capsys):
    code = export_chunks.main(["--db", str(seeded_db), "--doc-id", "nope"])
    assert code == 1
    assert "库里没有这些 doc_id" in capsys.readouterr().err


def test_bad_limit_fails_fast(seeded_db, capsys):
    assert export_chunks.main(["--db", str(seeded_db), "--limit-per-doc", "0"]) == 1
    assert "必须 >= 1" in capsys.readouterr().err


def test_row_fields_match_the_declared_contract(seeded_db, tmp_path):
    """字段名与 `FIELDS` 常量必须同源：改 `Chunk` 字段名时这里会先炸。"""

    conn = sqlite_store.connect(seeded_db)
    rows = export_chunks.collect_rows(conn)
    conn.close()
    assert set(rows[0]) == set(export_chunks.FIELDS)


def test_malformed_export_is_rejected_by_the_consumer(tmp_path):
    """坏行必须由下游按行号报错（本脚本的产物不允许被静默吞掉）。"""

    path = tmp_path / "bad.jsonl"
    path.write_text('{"chunk_id": "x"}\n', encoding="utf-8")
    with pytest.raises(TasksetError, match="缺少字段"):
        read_chunks(path)
