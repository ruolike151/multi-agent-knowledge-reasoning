"""查询侧收窄：从问句里抽代码 / 年份 / 公司名（`akr.storage.query_hints`，纯规则、零模型）。

两条必须被测试钉住的性质：**抽不到就不收窄**（宁可全库检索也不能抽错把 gold 过滤掉）、
**代码优先于公司名，年份只做二次过滤**（过滤后为空要撤销年份，而不是清空候选）。
"""

from akr.ingest.doc_meta import DocMeta
from akr.storage.query_hints import QueryHints, extract_hints, narrow_doc_ids


def metas():
    """一家公司两份年报 + 一份公告，外加一家同业公司——够覆盖全部收窄分支。"""

    return {
        "annual-002129-2024": DocMeta(
            doc_id="annual-002129-2024", company="TCL中环", code="002129", period="2024年年度"
        ),
        "annual-002129-2025": DocMeta(
            doc_id="annual-002129-2025", company="TCL中环", code="002129", period="2025年年度"
        ),
        "ann-002129-ir": DocMeta(
            doc_id="ann-002129-ir", company="TCL中环", code="002129", period="2026年半年度"
        ),
        "annual-601012-2024": DocMeta(
            doc_id="annual-601012-2024", company="隆基绿能", code="601012", period="2024年年度"
        ),
    }


def test_extracts_code_year_and_company():
    hints = extract_hints("TCL中环（002129）2024年年报的营业收入是多少？", metas())
    assert hints.codes == ("002129",)
    assert hints.years == ("2024",)
    assert hints.companies == ("TCL中环",)


def test_six_digit_numbers_not_in_the_corpus_are_not_hints():
    """`842123 万元` 这种金额也长得像证券代码；语料里没有它就不该拿去收窄。"""
    hints = extract_hints("2024 年成交金额 842123 万元，同比变化多少？", metas())
    assert hints.codes == ()
    assert hints.companies == ()
    assert not hints


def test_seven_digit_numbers_do_not_produce_a_code():
    hints = extract_hints("编号 0021299 的记录说明了什么？", metas())
    assert hints.codes == ()


def test_known_codes_can_be_restricted_by_the_caller():
    hints = extract_hints("002129 的营业收入", metas(), known_codes=frozenset({"601012"}))
    assert hints.codes == ()


def test_years_are_deduplicated_and_sorted():
    hints = extract_hints("2025 年与 2024 年相比，2024 年的毛利率如何？", metas())
    assert hints.years == ("2024", "2025")


def test_narrow_by_code_prefers_the_unique_key_over_the_company_name():
    """问句同时点出两家公司时，代码是唯一键，公司名会连带出同业文档。"""
    hints = extract_hints("TCL中环与隆基绿能 2024 年的毛利率，TCL中环（002129）是多少？", metas())
    assert hints.companies == ("TCL中环", "隆基绿能")
    assert narrow_doc_ids(hints, metas()) == ("annual-002129-2024",)


def test_narrow_by_company_keeps_every_document_of_that_company_when_no_year():
    hints = extract_hints("TCL中环的研发投入占营业收入的比例是多少？", metas())
    assert narrow_doc_ids(hints, metas()) == (
        "ann-002129-ir",
        "annual-002129-2024",
        "annual-002129-2025",
    )


def test_year_filters_within_the_company_hit():
    hints = extract_hints("隆基绿能（601012）2024 年的经营活动现金流", metas())
    assert narrow_doc_ids(hints, metas()) == ("annual-601012-2024",)


def test_year_is_dropped_when_it_would_empty_the_candidate_set():
    """问句说 2023 年、语料只有 2024/2025 —— 正确行为是回到代码命中的文档，不是清空。"""
    hints = extract_hints("TCL中环（002129）2023 年的净利润是多少？", metas())
    assert hints.years == ("2023",)
    assert narrow_doc_ids(hints, metas()) == (
        "ann-002129-ir",
        "annual-002129-2024",
        "annual-002129-2025",
    )


def test_no_hints_means_no_narrowing():
    hints = extract_hints("硅片出货量同比变化多少？", metas())
    assert hints == QueryHints()
    assert narrow_doc_ids(hints, metas()) == ()


def test_query_hints_are_falsy_without_code_or_company_even_with_a_year():
    """只有年份不足以确定文档，所以年份不参与「有没有线索」的真假判定。"""
    assert not QueryHints(years=("2024",))
    assert QueryHints(companies=("TCL中环",))
    assert QueryHints(codes=("002129",))
