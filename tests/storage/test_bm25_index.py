import json

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import DocMeta
from akr.storage.bm25_index import Bm25Index, tokenize_zh


def make_chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c{ordinal:04d}",
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        ordinal=ordinal,
        text=text,
        content_hash=f"hash{ordinal:04d}",
        section_path="第一章",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def test_tokenize_zh_keeps_terms_and_numbers():
    tokens = tokenize_zh("2023 年营业收入 100.5 亿元，同比增长 12%")
    assert "营业收入" in tokens
    assert "2023" in tokens
    assert "" not in tokens
    assert all(token == token.strip().lower() for token in tokens)


def test_tokenize_zh_drops_numeric_fragments_but_keeps_years_and_ratios():
    """千分位把一个金额切成的碎片 token 必须丢；年份与带小数点/百分号的数必须留。

    判据不是拍的：`tmp/probe_tokenizer_variants.py` 的 A/B 显示，只丢这一类（写法
    `nofrag-int`）BM25 单路 top-8 从 14/25 提到 16/25、MRR 0.3413 → 0.3980，而
    recall@50 一条不动；再激进一档指标逐格相同，所以保守取「纯整数且 ≤3 位」。
    """

    tokens = tokenize_zh("2024年营业收入 9,818,364,927.98 元，同比增长 12%，占比 0.35%")
    assert "9" not in tokens
    assert "818" not in tokens
    assert "364" not in tokens
    assert "2024" in tokens  # 4 位年份留着
    assert "927.98" in tokens  # 带小数点的金额留着
    assert "0.35%" in tokens  # 带百分号的比率留着
    assert "营业收入" in tokens  # 领域短语不受影响


def test_tokenize_zh_applies_the_fragment_rule_to_the_query_side_too():
    """查询侧同一条规则：`3`（「第 3 季度」）丢，公司代码 `002129` 与年份留着。"""

    tokens = tokenize_zh("TCL中环（002129）2024年年报中，硅片出货量为 9,818 万片")
    assert "002129" in tokens
    assert "2024" in tokens
    assert "9" not in tokens
    assert "818" not in tokens
    assert "3" not in tokenize_zh("第 3 季度经营活动产生的现金流量净额")


def test_search_ranks_query_term_match_first():
    index = Bm25Index()
    index.build([
        make_chunk("doc1", 0, "公司营业收入为 100 亿元"),
        make_chunk("doc1", 1, "公司员工人数为 5000 人"),
    ])
    hits = index.search("营业收入", top_k=2)
    assert hits[0].chunk_id == "doc1#c0000"
    assert hits[0].retriever == "bm25"
    assert hits[0].doc_id == "doc1"
    assert hits[0].source == "doc1.pdf"


def test_search_ignores_chunks_with_zero_score():
    index = Bm25Index()
    index.build([make_chunk("doc1", 0, "营业收入"), make_chunk("doc1", 1, "员工人数")])
    assert [hit.chunk_id for hit in index.search("员工", top_k=5)] == ["doc1#c0001"]


def test_upsert_is_idempotent_by_chunk_id():
    index = Bm25Index()
    chunk = make_chunk("doc1", 0, "营业收入")
    assert index.upsert([chunk]) == 1
    assert index.upsert([chunk]) == 1
    assert index.count() == 1


def test_remove_drops_chunk_from_results():
    index = Bm25Index()
    index.build([make_chunk("doc1", 0, "营业收入"), make_chunk("doc2", 0, "营业收入")])
    assert index.remove(["doc1#c0000"]) == 1
    assert [hit.chunk_id for hit in index.search("营业收入", top_k=5)] == ["doc2#c0000"]
    assert index.remove(["doc1#c0000"]) == 0


def test_search_filters_by_doc_ids():
    index = Bm25Index()
    index.build([make_chunk("doc1", 0, "营业收入"), make_chunk("doc2", 0, "营业收入")])
    hits = index.search("营业收入", top_k=5, doc_ids=["doc2"])
    assert [hit.chunk_id for hit in hits] == ["doc2#c0000"]


def test_save_and_load_roundtrip(tmp_path):
    index = Bm25Index()
    index.build([make_chunk("doc1", 0, "公司营业收入为 100 亿元")])
    path = tmp_path / "bm25.json"
    index.save(path)
    reloaded = Bm25Index.load(path)
    assert reloaded.count() == 1
    assert reloaded.search("营业收入", top_k=1)[0].chunk_id == "doc1#c0000"


# ---- 文档身份前缀（2026-09-25 新增，见 `akr.ingest.doc_meta` 的动机说明）----
# 语料里 6176 个 chunk，正文含证券代码的只有 36 个，而问句几乎都点名公司 + 代码——
# 前缀就是为了让「问句里最有区分度的 token」在语料里真实存在，且**只影响打分文本**。


def metas_for_doc2():
    return {"doc2": DocMeta(doc_id="doc2", company="乙公司", code="300274", period="2024年年度")}


def test_index_text_of_prefixes_with_document_identity():
    index = Bm25Index(doc_metas=metas_for_doc2())
    index.build([make_chunk("doc2", 0, "营业收入为 100 亿元")])
    header, body = index.index_text_of(make_chunk("doc2", 0, "营业收入为 100 亿元")).split("\n")
    assert header == "乙公司（300274） 2024年年度 第一章"
    assert body == "营业收入为 100 亿元"


def test_doc_metas_break_ties_between_identical_chunks_from_different_documents():
    """两家的年报里措辞雷同的段落靠身份前缀才分得开，不靠 chunk_id 兜底。"""
    plain = Bm25Index()
    plain.build([make_chunk("doc1", 0, "营业收入为 100 亿元"), make_chunk("doc2", 0, "营业收入为 100 亿元")])
    assert plain.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc1"

    indexed = Bm25Index(doc_metas=metas_for_doc2())
    indexed.build([make_chunk("doc1", 0, "营业收入为 100 亿元"), make_chunk("doc2", 0, "营业收入为 100 亿元")])
    assert indexed.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc2"


def test_returned_text_stays_the_original_chunk_text():
    """命中里的 `text` 会被回给模型、也被引用层当原文：前缀绝不能污染它。"""
    index = Bm25Index(doc_metas=metas_for_doc2())
    index.build([make_chunk("doc2", 0, "营业收入为 100 亿元")])
    hit = index.search("乙公司（300274）的营业收入", top_k=1)[0]
    assert hit.text == "营业收入为 100 亿元"
    assert "300274" not in hit.text


def test_set_doc_metas_rebuilds_the_scoring_matrix():
    index = Bm25Index()
    index.build([make_chunk("doc1", 0, "营业收入为 100 亿元"), make_chunk("doc2", 0, "营业收入为 100 亿元")])
    assert index.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc1"

    index.set_doc_metas(metas_for_doc2())

    assert index.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc2"


def test_load_applies_doc_metas_only_when_they_are_passed_in(tmp_path):
    """身份不落盘：`bm25.json` 里只有原文，装配时才把 manifest 派生的前缀接上。"""
    index = Bm25Index()
    index.build([make_chunk("doc1", 0, "营业收入为 100 亿元"), make_chunk("doc2", 0, "营业收入为 100 亿元")])
    path = tmp_path / "bm25.json"
    index.save(path)
    assert "doc_metas" not in path.read_text(encoding="utf-8")

    with_meta = Bm25Index.load(path, doc_metas=metas_for_doc2())
    without_meta = Bm25Index.load(path)

    assert with_meta.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc2"
    assert without_meta.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc1"


# ---- 索引文本口径的装配层旋钮（2026-09-25 新增，见 `akr.ingest.table_context`）----
# `index_text_fn` 优先于 `doc_metas` 派生的前缀；它是装配层旋钮，不进 `TOOL_SCHEMAS`，
# 也**不落盘**——同一份库用两种索引文本打分会让结果无法归因。


def test_index_text_fn_overrides_the_doc_metas_derivation():
    """两个都传时只走 `index_text_fn` 一条路：前缀绝不叠加。"""

    index = Bm25Index(
        doc_metas=metas_for_doc2(),
        index_text_fn=lambda chunk: f"自定义口径\n{chunk.text}",
    )
    chunk = make_chunk("doc2", 0, "营业收入为 100 亿元")
    assert index.index_text_of(chunk) == "自定义口径\n营业收入为 100 亿元"


def test_index_text_fn_is_accepted_by_load_and_still_returns_the_original_text(tmp_path):
    """`load` 也要接住这个旋钮；命中里的 `text` 仍是原文（前缀只进打分矩阵）。"""

    index = Bm25Index()
    index.build([make_chunk("doc2", 0, "营业收入为 100 亿元")])
    path = tmp_path / "bm25.json"
    index.save(path)

    reloaded = Bm25Index.load(
        path,
        doc_metas=metas_for_doc2(),
        index_text_fn=lambda chunk: f"乙公司（300274） 2024年年度\n{chunk.text}",
    )
    assert reloaded.index_text_of(make_chunk("doc2", 0, "营业收入为 100 亿元")).split("\n")[0] == (
        "乙公司（300274） 2024年年度"
    )
    hit = reloaded.search("乙公司（300274）的营业收入", top_k=1)[0]
    assert hit.text == "营业收入为 100 亿元"


def test_index_text_fn_is_not_persisted_in_the_payload(tmp_path):
    """索引文本口径与 `doc_metas` 一样不落盘：`bm25.json` 里永远只有 `k1` / `b` / `chunks`。"""

    index = Bm25Index(index_text_fn=lambda chunk: f"自定义口径\n{chunk.text}")
    index.build([make_chunk("doc1", 0, "营业收入为 100 亿元")])
    path = tmp_path / "bm25.json"
    index.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(payload) == ["b", "chunks", "k1"]
