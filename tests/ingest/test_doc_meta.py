"""文档身份元数据：从 manifest 派生检索前缀（`akr.ingest.doc_meta`）。

这一层是「chunk 没有身份」这个问题的修法：前缀只进 BM25 的打分矩阵，`Chunk.text`
与 artifact 里的原文都不动（2026-09-25 只读 A/B，`scripts/retrieval_ab.py`）。
"""

import dataclasses
import json

import pytest

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import (
    DocMeta,
    doc_headers,
    index_text,
    load_doc_meta,
    load_doc_meta_or_empty,
    make_index_text_fn,
)


def make_chunk(text: str = "本节披露硅片出货量。", section_path: str = "第四节 董事会报告") -> Chunk:
    return Chunk(
        chunk_id="d1#c0000",
        doc_id="d1",
        source="d1.pdf",
        ordinal=0,
        text=text,
        content_hash="hash0",
        section_path=section_path,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def test_prefix_is_identity_then_report_then_section():
    meta = DocMeta(
        doc_id="d1",
        company="TCL中环",
        code="002129",
        period="2024年年度",
        title="2024年年度报告",
    )
    assert meta.prefix("第四节 董事会报告") == "TCL中环（002129） 2024年年度 2024年年度报告 第四节 董事会报告"


def test_prefix_falls_back_to_company_when_the_code_is_missing():
    """代码缺失时不能留下空括号——那会变成一个凭空的 `（）` token。"""
    meta = DocMeta(doc_id="d1", company="甲公司", period="2024年年度")
    assert meta.prefix("第一章") == "甲公司 2024年年度 第一章"


def test_prefix_without_section_is_document_level():
    meta = DocMeta(doc_id="d1", company="甲公司", code="002129")
    assert meta.prefix() == "甲公司（002129）"


def test_years_are_extracted_from_the_period():
    meta = DocMeta(doc_id="d1", period="2024年年度")
    assert meta.years == ("2024",)
    assert DocMeta(doc_id="d2", period="2026年半年度").years == ("2026",)
    assert DocMeta(doc_id="d3", period="2024-2025年").years == ("2024", "2025")
    assert DocMeta(doc_id="d4", period="").years == ()


def test_index_text_prefixes_the_chunk_without_touching_its_text():
    """返回给模型的 `text`（以及 artifact 里的原文）必须是同一份原始正文。"""
    chunk = make_chunk()
    meta = DocMeta(doc_id="d1", company="TCL中环", code="002129", period="2024年年度")

    text = index_text(chunk, meta)

    assert text == f"TCL中环（002129） 2024年年度 第四节 董事会报告\n{chunk.text}"
    assert chunk.text == "本节披露硅片出货量。"


def test_index_text_is_plain_text_without_metadata():
    chunk = make_chunk()
    assert index_text(chunk, None) == chunk.text


def test_section_path_alone_becomes_the_prefix():
    """章节路径是**块级**信息，没有公司身份时它仍然进索引文本；不会多出一个空行。"""
    chunk = make_chunk(section_path="第四节 董事会报告")
    text = index_text(chunk, DocMeta(doc_id="d1"))
    assert text == f"第四节 董事会报告\n{chunk.text}"


def test_metadata_without_identity_and_without_section_leaves_the_text_alone():
    chunk = make_chunk(section_path="")
    assert index_text(chunk, DocMeta(doc_id="d1")) == chunk.text


def test_make_index_text_fn_resolves_the_chunk_own_document():
    """`VectorStore(index_text_fn=...)` 拿到的是 chunk，必须按它自己的 doc_id 查身份。"""

    metas = {
        "d1": DocMeta(doc_id="d1", company="甲公司", code="002129", period="2024年年度"),
        "d2": DocMeta(doc_id="d2", company="乙公司", code="300001", period="2025年年度"),
    }
    render = make_index_text_fn(metas)

    assert render(make_chunk()).startswith("甲公司（002129） 2024年年度")
    other = make_chunk()
    assert render(dataclasses.replace(other, doc_id="d2", chunk_id="d2#c0000")).startswith(
        "乙公司（300001） 2025年年度"
    )


def test_make_index_text_fn_falls_back_to_raw_text_for_unknown_documents():
    """缺身份的 chunk 退化成原文（与 BM25 同口径），不拼一行空前缀出来。"""

    chunk = make_chunk()
    assert make_index_text_fn({})(chunk) == chunk.text
    assert make_index_text_fn({"d2": DocMeta(doc_id="d2", company="乙公司")})(chunk) == chunk.text


def write_manifest(tmp_path, documents):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_doc_meta_reads_the_identity_fields(tmp_path):
    path = write_manifest(
        tmp_path,
        [
            {
                "doc_id": "annual-002129-2024",
                "company": "TCL中环",
                "code": "002129",
                "period": "2024年年度",
                "announcement_title": "2024年年度报告",
            }
        ],
    )
    metas = load_doc_meta(path)
    assert metas["annual-002129-2024"].code == "002129"
    assert metas["annual-002129-2024"].prefix() == "TCL中环（002129） 2024年年度 2024年年度报告"


def test_load_doc_meta_rejects_records_without_doc_id(tmp_path):
    path = write_manifest(tmp_path, [{"company": "TCL中环"}])
    with pytest.raises(ValueError, match="doc_id"):
        load_doc_meta(path)


def test_load_doc_meta_rejects_a_manifest_without_documents(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"companies": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="documents"):
        load_doc_meta(path)


def test_load_doc_meta_or_empty_tolerates_missing_manifest(tmp_path):
    """缺 manifest 只是「没有身份前缀」，不该让工具进程起不来。"""
    assert load_doc_meta_or_empty(tmp_path / "不存在.json") == {}
    assert load_doc_meta_or_empty(None) == {}


def test_doc_headers_are_document_level_prefixes(tmp_path):
    path = write_manifest(
        tmp_path,
        [
            {"doc_id": "d1", "company": "甲公司", "code": "002129", "period": "2024年年度"},
            {"doc_id": "d2", "company": "", "code": "", "period": ""},
        ],
    )
    assert doc_headers(load_doc_meta(path)) == {"d1": "甲公司（002129） 2024年年度"}
