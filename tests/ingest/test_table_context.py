"""`akr.ingest.table_context` 的单测：夹具用真实 chunk 原文，不造合成表格。"""

from __future__ import annotations

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import DocMeta, index_text
from akr.ingest.table_context import (
    TABLE_LEAD_LINES,
    make_table_index_text_fn,
    table_index_text,
    table_lead,
)

# `data/chunks.jsonl` → `annual-002129-2024#cf615a2c79af5ffa4` 的 `text` **逐字拷贝**
# （`ordinal` 18、`section_path` = 「第二节 公司简介和主要财务指标 / 六、主要会计数据和财务指标」）。
# 贴字面量而不是读 `data/`：单测不依赖生产语料是否已经入库。
REAL_TABLE_EXCERPT = (
    "幢10层1001-1至1001-26\n"
    "签字会计师姓名 张媛媛、熊能、熊欣\n"
    "公司聘请的报告期内履行持续督导职责的保荐机构\n"
    "□适用 \uf052不适用\n"
    "公司聘请的报告期内履行持续督导职责的财务顾问\n"
    "□适用 \uf052不适用\n"
    "六、主要会计数据和财务指标\n"
    "公司是否需追溯调整或重述以前年度会计数据\n"
    "□是 \uf052否\n"
    "2024年 2023年 本年比上年增减 2022年\n"
    "营业收入（元） 28,418,504,235.63 59,146,463,192.95 -51.95% 67,010,157,024.98\n"
    "归属于上市公司股东\n"
    "-9,818,364,927.98 3,416,058,966.07 -387.42% 6,818,653,821.89\n"
    "的净利润（元）\n"
    "归属于上市公司股东\n"
    "的扣除非经常性损益 -10,900,344,275.25 2,574,813,445.38 -523.35% 6,483,116,499.48\n"
    "的净利润（元）\n"
    "经营活动产生的现金\n"
    "2,839,341,936.85 5,181,162,667.85 -45.20% 5,056,839,128.83\n"
    "流量净额（元）\n"
    "基本每股收益（元/\n"
    "-2.4629 0.8518 -389.14% 1.6949\n"
    "股）\n"
    "稀释每股收益（元/\n"
    "-2.4629 0.8518 -389.14% 1.6947\n"
    "股）\n"
    "加权平均净资产收益\n"
    "-27.26% 8.83% -36.09% 19.74%\n"
    "率\n"
    "2024年末 2023年末 本年末比上年末增减 2022年末\n"
    "总资产（元） 125,597,525,162.66 125,063,043,317.18 0.43% 108,312,922,803.15\n"
    "归属于上市公司股东\n"
    "30,906,672,036.26 41,483,746,947.05 -25.50% 37,617,698,425.43\n"
    "的净资产（元）"
)

# 真实原文摘出的那 3 行导读（第 1 行是期间行、第 2/3 行是数值行，第 3 行的指标名由
# 「归属于上市公司股东」+「的净利润（元）」两行碎片拼回）。
REAL_TABLE_LEAD = (
    "2024年 2023年 本年比上年增减 2022年"
    " | 营业收入（元） 28,418,504,235.63 59,146,463,192.95 -51.95% 67,010,157,024.98"
    " | 归属于上市公司股东的净利润（元） -9,818,364,927.98 3,416,058,966.07"
    " -387.42% 6,818,653,821.89"
)


def make_chunk(*, text: str, section_path: str = "", doc_id: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c0000",
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        ordinal=0,
        text=text,
        content_hash="hash0000",
        section_path=section_path,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def test_table_lead_extracts_period_and_numeric_rows_from_a_real_table() -> None:
    """真实表格块：第一行是期间行，后两行是指标名 + 数值，顺序与原表一致。"""

    assert table_lead(REAL_TABLE_EXCERPT) == REAL_TABLE_LEAD


def test_table_lead_joins_the_name_fragments_around_a_numeric_row() -> None:
    """指标名被排版切成两行时必须拼回：`归属于上市公司股东` + `的净利润（元）`。"""

    rows = table_lead(REAL_TABLE_EXCERPT).split(" | ")
    assert rows[2].startswith("归属于上市公司股东的净利润（元） ")
    # 被吃掉的那两个碎片行不再单独成行——否则 max_lines=3 会被碎片占满。
    assert len(rows) == TABLE_LEAD_LINES


def test_numeric_row_with_cjk_does_not_steal_the_next_line_name() -> None:
    """行内已经有汉字的数值行，指标名就在本行上，不能再把下一行的名字接过来。"""

    lead = table_lead("营业收入（元） 1 2 3\n的净利润（元）\n1 2 3")
    assert lead == "营业收入（元） 1 2 3 | 的净利润（元） 1 2 3"


def test_table_lead_returns_empty_for_a_prose_chunk() -> None:
    """没有表格行的块必须返回空串——这是「正文块恒等变换」的前提。"""

    assert table_lead("公司主营业务为半导体硅片的研发、生产与销售，报告期内未发生重大变化。") == ""


def test_table_lead_ignores_a_prose_sentence_that_contains_numbers() -> None:
    """正文句子按空格拆开也会有好几个纯数字格子，不能被当成数值行。"""

    assert table_lead("根据中国证券监督管理委员会于 2021 年 11 月 9 日签发的证监许可") == ""


def test_table_lead_rejects_a_section_heading_as_a_name_fragment() -> None:
    """13 字的章节标题超出碎片长度上界，不能作为下一行的指标名。"""

    assert table_lead("六、主要会计数据和财务指标\n营业收入 1 2 3") == "营业收入 1 2 3"


def test_table_lead_respects_max_lines() -> None:
    """`max_lines` 是硬上界：它决定 prompt 与嵌入预算被占多少。"""

    assert table_lead(REAL_TABLE_EXCERPT, max_lines=1) == "2024年 2023年 本年比上年增减 2022年"


def test_table_index_text_is_identical_to_index_text_without_a_table() -> None:
    """正文块上必须**逐字恒等**于身份前缀口径，否则「默认值 = 历史行为」不成立。"""

    chunk = make_chunk(text="公司主营业务为半导体硅片的研发、生产与销售。", section_path="第一节")
    meta = DocMeta(doc_id="doc1", company="甲公司", code="002129", period="2024年年度")
    assert table_index_text(chunk, meta) == index_text(chunk, meta)
    assert table_index_text(chunk, None) == chunk.text


def test_table_index_text_prepends_the_lead_before_the_original_text() -> None:
    """索引文本 = 身份前缀 → 导读行 → 原文；原文一个字都不改。"""

    chunk = make_chunk(text=REAL_TABLE_EXCERPT, section_path="六、主要会计数据和财务指标")
    meta = DocMeta(doc_id="doc1", company="甲公司", code="002129", period="2024年年度")
    lines = table_index_text(chunk, meta).split("\n")
    assert lines[0] == "甲公司（002129） 2024年年度 六、主要会计数据和财务指标"
    assert lines[1] == REAL_TABLE_LEAD
    assert table_index_text(chunk, meta).endswith(chunk.text)


def test_make_table_index_text_fn_only_drops_the_identity_prefix_without_meta() -> None:
    """缺 meta 的块只丢身份前缀，导读行照旧生效（与 `index_text` 的降级口径一致）。"""

    chunk = make_chunk(text=REAL_TABLE_EXCERPT, section_path="六、主要会计数据和财务指标")
    metas = {"doc1": DocMeta(doc_id="doc1", company="甲公司", code="002129", period="2024年年度")}
    render = make_table_index_text_fn(metas)
    assert render(chunk).split("\n")[:2] == [
        "甲公司（002129） 2024年年度 六、主要会计数据和财务指标",
        REAL_TABLE_LEAD,
    ]
    assert render(make_chunk(text=REAL_TABLE_EXCERPT, doc_id="doc9")) == (
        f"{REAL_TABLE_LEAD}\n{REAL_TABLE_EXCERPT}"
    )
