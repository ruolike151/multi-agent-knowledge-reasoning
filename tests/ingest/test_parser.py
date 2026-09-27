import pytest

from akr.ingest.parser import (
    ParseError,
    has_text_layer,
    pages_from_pages,
    parse_pdf,
)


class FakePage:
    def __init__(self, text, tables=None):
        self._text = text
        self._tables = tables or []

    def extract_text(self):
        return self._text

    def extract_tables(self):
        return self._tables


def test_page_numbers_start_at_one_and_whitespace_is_normalized():
    pages = pages_from_pages([FakePage("第一页   内容\r\n第二行\t尾部")])
    assert pages[0].page == 1
    assert pages[0].text == "第一页 内容\n第二行 尾部"


def test_none_text_becomes_empty_string():
    assert pages_from_pages([FakePage(None)])[0].text == ""


def test_start_page_offset_is_applied():
    pages = pages_from_pages([FakePage("x"), FakePage("y")], start_page=7)
    assert [p.page for p in pages] == [7, 8]


def test_tables_are_preserved_and_none_cells_become_empty():
    table = [["项目", None], ["营业收入", "100"]]
    pages = pages_from_pages([FakePage("x", [table])])
    assert pages[0].tables == [[["项目", ""], ["营业收入", "100"]]]


def test_parse_pdf_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(ParseError, match="不存在"):
        parse_pdf(tmp_path / "没有这个文件.pdf")


def test_has_text_layer_detects_scanned_documents():
    scanned = pages_from_pages([FakePage(""), FakePage(None)])
    assert has_text_layer(scanned) is False
    assert has_text_layer(pages_from_pages([FakePage("正文" * 200)])) is True
