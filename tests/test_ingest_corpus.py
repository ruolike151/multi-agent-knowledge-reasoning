"""`scripts/ingest_corpus.py` 的单测。

与 `tests/test_m1_smoke.py` / `tests/test_fetch_corpus.py` 保持一致：脚本不在 `src/` 下，
用 importlib 按路径加载。这里刻意不触发真实链路（不解析真 PDF、不连 Neo4j、不加载嵌入模型），
真链路由脚本自己跑。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from akr.ingest.parser import ParsedPage
from akr.model.cost import Usage
from akr.storage import sqlite_store

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ingest_corpus.py"
CORPUS_READY = (ROOT / "data" / "corpus" / "annual").is_dir()


def _load():
    spec = importlib.util.spec_from_file_location("ingest_corpus", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # 必须先注册进 sys.modules：本模块用了 `from __future__ import annotations`，
    # dataclass 解析字符串注解时要回头查 sys.modules[cls.__module__]。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest_corpus = _load()


@dataclass
class _FakeResult:
    text: str
    usage: Usage


def _record(**overrides):
    record = {
        "doc_id": "annual-600438-2025",
        "kind": "annual",
        "code": "600438",
        "period": "2025年度",
        "pages": 254,
        "file": "data/corpus/annual/600438_通威股份_2025年报.pdf",
    }
    record.update(overrides)
    return record


def _fake_pages(count: int = 3, *, chars: int = 900) -> list[ParsedPage]:
    return [ParsedPage(page=i + 1, text="光" * chars, tables=[]) for i in range(count)]


def test_manifest_lists_the_fifteen_corpus_documents():
    records = ingest_corpus.load_documents()
    assert len(records) == 15
    assert len({r["doc_id"] for r in records}) == 15
    for record in records:
        assert {"doc_id", "kind", "code", "period", "file"} <= set(record)


def test_load_documents_rejects_manifest_without_documents(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"documents": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        ingest_corpus.load_documents(path)


def test_select_documents_filters_then_limits():
    records = [_record(doc_id="a"), _record(doc_id="b"), _record(doc_id="c")]
    assert [r["doc_id"] for r in ingest_corpus.select_documents(records)] == ["a", "b", "c"]
    assert [
        r["doc_id"] for r in ingest_corpus.select_documents(records, doc_ids=("c", "a"))
    ] == ["a", "c"]
    assert [
        r["doc_id"] for r in ingest_corpus.select_documents(records, doc_ids=("c", "a"), limit=1)
    ] == ["a"]


def test_select_documents_rejects_unknown_doc_id_and_bad_limit():
    records = [_record(doc_id="a")]
    with pytest.raises(ValueError, match="没有这些 doc_id"):
        ingest_corpus.select_documents(records, doc_ids=("nope",))
    with pytest.raises(ValueError, match="--limit"):
        ingest_corpus.select_documents(records, limit=0)


def test_resolve_pdf_prefers_the_declared_file(tmp_path):
    pdf = tmp_path / "data" / "corpus" / "annual" / "600438_通威股份_2025年报.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    assert ingest_corpus.resolve_pdf(_record(), root=tmp_path) == pdf


def test_resolve_pdf_falls_back_when_manifest_file_is_wrong(tmp_path):
    """manifest 的 `file` 与实体文件名曾经不一致，必须能按公司代码 + 年份回退命中。"""

    kind_dir = tmp_path / "data" / "corpus" / "annual"
    kind_dir.mkdir(parents=True)
    (kind_dir / "002129_TCL中环_2024年报.pdf").write_bytes(b"%PDF-1.4")
    (kind_dir / "002129_TCL中环_2025年报.pdf").write_bytes(b"%PDF-1.4")
    record = _record(
        doc_id="annual-002129-2024",
        code="002129",
        period="2024年度",
        file="data/corpus/annual/002129_TCL中环_2024年年度报告.pdf",  # 不存在的名字
    )
    assert ingest_corpus.resolve_pdf(record, root=tmp_path).name == "002129_TCL中环_2024年报.pdf"


def test_resolve_pdf_refuses_to_guess_between_ambiguous_candidates(tmp_path):
    kind_dir = tmp_path / "data" / "corpus" / "annual"
    kind_dir.mkdir(parents=True)
    for year in ("2024", "2025"):
        (kind_dir / f"002129_TCL中环_{year}年报.pdf").write_bytes(b"%PDF-1.4")
    record = _record(doc_id="annual-002129", code="002129", period="", file="missing.pdf")
    with pytest.raises(FileNotFoundError, match="找不到 PDF"):
        ingest_corpus.resolve_pdf(record, root=tmp_path)


def test_preview_counts_chunks_without_touching_stores(monkeypatch):
    monkeypatch.setattr(ingest_corpus, "parse_document", lambda path, pages=None: _fake_pages(2))
    monkeypatch.setattr(ingest_corpus, "resolve_pdf", lambda record, root=None: Path("fake.pdf"))
    rows = ingest_corpus.preview([_record()])
    assert len(rows) == 1
    assert rows[0].pages == 2
    assert rows[0].chunks >= 1
    assert rows[0].chars_per_chunk > 0


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [_record()]}), encoding="utf-8")
    monkeypatch.setattr(ingest_corpus, "parse_document", lambda path, pages=None: _fake_pages(2))
    monkeypatch.setattr(ingest_corpus, "resolve_pdf", lambda record, root=None: Path("fake.pdf"))

    data_root = tmp_path / "data"
    code = ingest_corpus.main(
        ["--dry-run", "--manifest", str(manifest), "--data-root", str(data_root)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "[dry-run]" in out and "[parse]" in out and "合计" in out
    assert not data_root.exists()  # 预演不落任何盘


def test_flags_reject_a_dead_budget_gate(capsys):
    """不配 `--extract` 时闸门是死开关：必须报错而不是静默忽略。"""

    assert ingest_corpus.main(["--max-calls", "10"]) == 1
    assert "只在 --extract 下有意义" in capsys.readouterr().err
    assert ingest_corpus.main(["--extract", "--max-calls", "0"]) == 1
    assert "必须 >= 1" in capsys.readouterr().err


def test_call_meter_refuses_the_call_that_would_exceed_the_gate():
    meter = ingest_corpus.CallMeter(max_calls=1, peak=False)
    meter.check()
    meter.charge(Usage(miss_tokens=1_000_000))
    assert meter.calls == 1
    assert meter.cost_cny == pytest.approx(0.15 * 7.1, rel=1e-6)
    with pytest.raises(ingest_corpus.CallBudgetExceeded, match="--max-calls=1"):
        meter.check()


def test_call_meter_stops_on_cost_gate_too():
    meter = ingest_corpus.CallMeter(max_cost_cny=0.5, peak=True)
    meter.check()
    meter.charge(Usage(miss_tokens=1_000_000))
    with pytest.raises(ingest_corpus.CallBudgetExceeded, match="--max-cost-cny"):
        meter.check()


def test_metered_client_counts_calls_and_never_calls_the_inner_client_twice_over_budget():
    class _Inner:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            return _FakeResult(text="{}", usage=Usage(miss_tokens=100, output_tokens=50))

    inner = _Inner()
    meter = ingest_corpus.CallMeter(max_calls=2, peak=True)
    client = ingest_corpus.MeteredClient(inner, meter)
    client.chat(system="s", messages=[], mode="non-thinking")
    client.chat(system="s", messages=[], mode="non-thinking")
    with pytest.raises(ingest_corpus.CallBudgetExceeded):
        client.chat(system="s", messages=[], mode="non-thinking")
    assert inner.calls == 2
    assert meter.calls == 2
    assert meter.cost_cny > 0


def test_dump_bm25_rebuilds_from_sqlite(tmp_path):
    conn = sqlite_store.connect(tmp_path / "akr.sqlite")
    sqlite_store.init_schema(conn)
    sqlite_store.upsert_document(
        conn,
        doc_id="doc-1",
        source="data/corpus/annual/x.pdf",
        doc_type="annual_report",
        page_count=1,
        updated_at="2026-09-25T00:00:00+00:00",
    )
    from akr.ingest.chunker import Chunk

    sqlite_store.replace_chunks(
        conn,
        "doc-1",
        [
            Chunk(
                chunk_id="doc-1#0000",
                doc_id="doc-1",
                source="data/corpus/annual/x.pdf",
                ordinal=0,
                text="公司 2025 年营业收入为 100 亿元。",
                content_hash="h0",
                section_path="第三节 管理层讨论与分析",
                page_start=1,
                page_end=1,
                char_start=0,
                char_end=17,
            )
        ],
    )
    path = tmp_path / "bm25.json"
    assert ingest_corpus.dump_bm25(conn, path) == 1
    assert path.is_file()
    assert ingest_corpus.Bm25Index.load(path).count() == 1


@pytest.mark.skipif(not CORPUS_READY, reason="语料 PDF 未下载（不入 Git）")
def test_every_manifest_document_resolves_to_an_existing_pdf():
    records = ingest_corpus.load_documents()
    paths = [ingest_corpus.resolve_pdf(record) for record in records]
    assert all(path.is_file() for path in paths)
    assert len({path.name for path in paths}) == 15
    assert sum(record["pages"] for record in records) == 2725
