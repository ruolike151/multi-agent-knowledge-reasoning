"""`akr.ingest.unit_chunks` 的单测：夹具用真实 chunk 原文摘录，不造合成表格。"""

from __future__ import annotations

from akr.ingest.chunker import chunk_pages, content_hash
from akr.ingest.doc_meta import DocMeta
from akr.ingest.parser import ParsedPage
from akr.ingest.table_context import table_lead
from akr.ingest.unit_chunks import (
    EMBED_INPUT_CHARS,
    MIN_UNIT_CHARS,
    unit_chunks,
    unit_window,
)


def page(text: str, number: int = 1) -> ParsedPage:
    return ParsedPage(page=number, text=text, tables=[])


# `data/chunks.jsonl` → `annual-002129-2024#cf615a2c79af5ffa4` 的**逐字摘录**（截到第 6 行）。
# 贴字面量而不是读 `data/`：单测不依赖生产语料是否已经入库。
REAL_TABLE_EXCERPT = (
    "2024年 2023年 本年比上年增减 2022年\n"
    "营业收入（元） 28,418,504,235.63 59,146,463,192.95 -51.95% 67,010,157,024.98\n"
    "归属于上市公司股东\n"
    "-9,818,364,927.98 3,416,058,966.07 -387.42% 6,818,653,821.89\n"
    "的净利润（元）\n"
    "经营活动产生的现金\n"
    "2,839,341,936.85 5,181,162,667.85 -45.20% 5,056,839,128.83\n"
    "流量净额（元）"
)
PROSE = "公司主营业务为半导体硅片的研发、生产与销售，报告期内未发生重大变化。"
META = DocMeta(doc_id="doc1", company="甲公司", code="002129", period="2024年年度")


def build(text: str, *, meta: DocMeta | None = META, **kwargs):
    pages = [page(text)]
    return unit_chunks(pages, doc_id="doc1", source="a.pdf", meta=meta, **kwargs)


def test_the_budget_constant_is_the_one_the_embedder_enforces() -> None:
    """两份常量必须等值：ingest 层不许 import storage 层，只能用一条断言锁住口径。"""

    from akr.storage.embeddings import MAX_INPUT_CHARS

    assert EMBED_INPUT_CHARS == MAX_INPUT_CHARS


def test_unit_ids_are_content_addressed_and_stable_across_reruns() -> None:
    first = build(PROSE)
    second = build(PROSE)
    assert [u.chunk.chunk_id for u in first] == [u.chunk.chunk_id for u in second]
    assert first[0].chunk.chunk_id == f"doc1#u{content_hash(PROSE)}"


def test_unit_ids_live_in_their_own_namespace_so_they_cannot_shadow_parents() -> None:
    """`#u` / `#c` 两个前缀是「unit 与父块共处一个集合」的前提。"""

    parents = {c.chunk_id for c in chunk_pages([page(PROSE)], doc_id="doc1", source="a.pdf")}
    assert not parents & {u.chunk.chunk_id for u in build(PROSE)}
    assert all("#u" in u.chunk.chunk_id for u in build(PROSE))


def test_every_unit_maps_back_to_a_parent_chunk_id_the_production_chunker_produces() -> None:
    """unit 折回父块靠 `parent_id`：它必须逐字等于 `chunk_pages` 会给的那个 id。"""

    pages = [page(REAL_TABLE_EXCERPT + "\n" + PROSE)]
    parents = [c.chunk_id for c in chunk_pages(pages, doc_id="doc1", source="a.pdf")]
    units = unit_chunks(pages, doc_id="doc1", source="a.pdf", meta=META)

    assert {u.parent_id for u in units} == set(parents)
    # 顺序按父块顺序、父块内按行顺序（同一条父块的 unit 不许被别的父块插空）。
    order = [parents.index(u.parent_id) for u in units]
    assert order == sorted(order)


def test_units_of_a_parent_cover_its_text_without_gaps_or_duplicates() -> None:
    """unit 是父块行流上的连续切片：拼回去必须逐字等于父块原文（不丢行、不重复）。"""

    pages = [page(REAL_TABLE_EXCERPT + "\n" + PROSE)]
    units = unit_chunks(pages, doc_id="doc1", source="a.pdf", meta=META)
    parents = {c.chunk_id: c.text for c in chunk_pages(pages, doc_id="doc1", source="a.pdf")}

    by_parent: dict[str, list[str]] = {}
    for unit in units:
        by_parent.setdefault(unit.parent_id, []).append(unit.chunk.text)
    assert set(by_parent) == set(parents)
    for parent_id, texts in by_parent.items():
        assert "\n".join(texts) == parents[parent_id]


def test_table_block_is_split_into_row_aligned_units() -> None:
    """窗口容得下最长一行时，切点必须落在**行边界**上：不切在指标名与数值之间。"""

    rows = REAL_TABLE_EXCERPT.split("\n")
    lead = table_lead(REAL_TABLE_EXCERPT)
    window = unit_window(header="", lead=lead, target=200)
    assert max(len(row) for row in rows) < window  # 前提：没有一行需要硬切

    units = build(REAL_TABLE_EXCERPT, meta=None, embed_chars=300, table_chars=200)

    assert len(units) > 1
    for unit in units:
        assert all(line in rows for line in unit.chunk.text.split("\n"))
        assert len(unit.chunk.text) <= window


def test_a_row_longer_than_the_window_falls_back_to_a_hard_split() -> None:
    """单行超窗（长数值行）只能硬切——这是 `chunker._hard_split` 的兜底，不是行级切法失效。"""

    units = build(REAL_TABLE_EXCERPT, meta=None, embed_chars=140, min_chars=20, table_chars=60)
    window = unit_window(header="", lead=table_lead(REAL_TABLE_EXCERPT), target=60, embed_chars=140)
    assert max(len(u.chunk.text) for u in units) <= window


def test_each_unit_of_a_table_block_inherits_the_parent_lead_line() -> None:
    """导读行来自**父块**：单元自己可能整段不含期间行/数值行，线索不能只留在第一块里。"""

    units = build(REAL_TABLE_EXCERPT, embed_chars=120, min_chars=20, table_chars=40)
    leads = {u.index_text.split("\n")[1] for u in units}

    assert len(units) > 1
    assert len(leads) == 1
    assert leads.pop().startswith("2024年 2023年 本年比上年增减 2022年")


def test_prose_block_stays_in_one_unit_while_the_budget_allows() -> None:
    """正文块的目标＝父块大小：预算够时不许把连续叙述切碎。"""

    units = build(PROSE)
    assert len(units) == 1
    assert units[0].chunk.text == PROSE
    assert units[0].index_text.endswith(PROSE)


def test_index_text_fits_the_embedding_window() -> None:
    """切块的目的就是「一条 unit 的索引文本不会被嵌入截断」。"""

    units = build(REAL_TABLE_EXCERPT + "\n" + PROSE)
    assert units
    assert max(len(u.index_text) for u in units) <= EMBED_INPUT_CHARS


def test_window_is_the_prefix_and_lead_subtracted_budget() -> None:
    """窗口 = min(目标, 预算)：前缀与导读行占掉的字符必须从 unit 预算里扣掉。"""

    header = "甲公司（002129） 2024年年度"
    lead = "2024年 2023年 本年比上年增减 2022年"
    assert unit_window(header=header, lead=lead, target=400) == 400  # 预算 440 > 400
    long_header = "前缀" * 100
    assert unit_window(header=long_header, lead=lead, target=400) == EMBED_INPUT_CHARS - (
        len(long_header) + 1 + len(lead)
    )
    # 前缀自己就把窗口吃光时取下限：切得再碎也塞不进去，保住可读的下限更有用。
    assert unit_window(header="前缀" * 200, lead=lead, target=400) == MIN_UNIT_CHARS


def test_duplicate_unit_text_inside_one_document_gets_a_suffix() -> None:
    """与父块同一套去重写法：重复行块的 id 必须可区分，否则 Chroma 会互相覆盖。"""

    pages = [page("单位：元\n用友\n单位：元")]
    units = unit_chunks(
        pages, doc_id="doc1", source="a.pdf", meta=META, embed_chars=6, min_chars=4, prose_chars=6
    )
    ids = [u.chunk.chunk_id for u in units]
    assert len(ids) == len(set(ids))
    assert any(chunk_id.endswith("#2") for chunk_id in ids)


def test_without_meta_the_unit_index_text_is_the_raw_text() -> None:
    """缺身份时只丢前缀，导读行照旧生效（与块级 `table_index_text` 同一条降级口径）。"""

    pages = [page(REAL_TABLE_EXCERPT)]
    units = unit_chunks(
        pages, doc_id="doc1", source="a.pdf", meta=None, embed_chars=300, table_chars=200
    )
    assert units[0].index_text.startswith(
        "2024年 2023年 本年比上年增减 2022年 | 营业收入（元）"
    )
