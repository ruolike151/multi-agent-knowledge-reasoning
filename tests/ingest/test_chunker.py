from akr.ingest.chunker import chunk_pages, content_hash, detect_heading
from akr.ingest.parser import ParsedPage


def page(text, number=1):
    return ParsedPage(page=number, text=text, tables=[])


def test_headings_are_detected_with_levels():
    assert detect_heading("第一章 公司简介").level == 1
    assert detect_heading("一、主要业务").level == 2
    assert detect_heading("（三）员工情况").level == 3
    assert detect_heading("1.1 合并财务报表").level == 2
    assert detect_heading("1.1.1 营业收入构成").level == 3
    assert detect_heading("2024 年公司营业收入为 100 亿元。") is None


def test_chunk_ids_are_content_addressed_and_stable_across_reruns():
    pages = [page("第一章 公司简介\n本公司成立于 1998 年。")]
    first = chunk_pages(pages, doc_id="doc1", source="a.pdf")
    second = chunk_pages(pages, doc_id="doc1", source="a.pdf")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == f"doc1#c{content_hash(first[0].text)}"


def test_duplicate_text_inside_one_document_gets_suffix():
    pages = [page("单位：元\n单位：元")]
    chunks = chunk_pages(pages, doc_id="doc1", source="a.pdf", max_chars=10, overlap=0)
    assert chunks[0].chunk_id.endswith("#c" + content_hash("单位：元"))
    assert chunks[1].chunk_id.endswith("#2")


def test_section_path_is_nested_and_written_onto_chunks():
    pages = [page("第一章 公司简介\n公司概况正文。\n一、主要业务\n业务正文。")]
    chunks = chunk_pages(pages, doc_id="doc1", source="a.pdf", max_chars=12, overlap=0)
    paths = {c.text.split("\n")[0]: c.section_path for c in chunks}
    assert paths["公司概况正文。"] == "第一章 公司简介"
    assert paths["业务正文。"] == "第一章 公司简介 / 一、主要业务"


def test_overlap_carries_tail_text_into_next_chunk():
    pages = [page("A" * 10 + "\n" + "B" * 10)]
    chunks = chunk_pages(pages, doc_id="doc1", source="a.pdf", max_chars=12, overlap=5)
    assert len(chunks) == 2
    assert "A" * 5 in chunks[1].text


def test_long_single_line_is_hard_split_within_budget():
    pages = [page("长" * 25)]
    chunks = chunk_pages(pages, doc_id="doc1", source="a.pdf", max_chars=10, overlap=0)
    assert [len(c.text) for c in chunks] == [10, 10, 5]


def test_page_and_char_offsets_are_tracked():
    pages = [page("第一行。", number=3), page("第二行。", number=4)]
    chunks = chunk_pages(pages, doc_id="doc1", source="a.pdf", max_chars=5, overlap=0)
    assert chunks[0].page_start == 3
    assert chunks[-1].page_end == 4
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 4
