import math

import pytest

from akr.ingest.chunker import Chunk
from akr.storage.vector_store import VectorStore


def fake_embed(texts: list[str]) -> list[list[float]]:
    """确定性假嵌入：按关键词出现次数构造 3 维向量。"""
    return [[float(t.count("营收")), float(t.count("员工")), 1.0] for t in texts]


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


def make_store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", fake_embed)


def test_index_text_fn_changes_embedding_input_but_not_stored_documents(tmp_path):
    """身份前缀只进嵌入文本：`documents` 存原文，否则回给模型的证据会多一行前缀。"""

    seen: list[list[str]] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return fake_embed(texts)

    store = VectorStore(
        tmp_path / "chroma",
        recording_embed,
        index_text_fn=lambda chunk: f"{chunk.doc_id} 身份\n{chunk.text}",
    )
    store.add([make_chunk("doc1", 0, "公司营收 100 亿元")])

    assert seen == [["doc1 身份\n公司营收 100 亿元"]]
    # 库内的 documents 是原文：命中回给模型的 text 不该带前缀
    assert store.query("营收", top_k=1)[0].text == "公司营收 100 亿元"


def test_default_index_text_fn_is_the_raw_chunk_text(tmp_path):
    """不开开关时口径必须与本仓历史索引完全一致（嵌的就是原文）。"""

    seen: list[list[str]] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return fake_embed(texts)

    VectorStore(tmp_path / "chroma", recording_embed).add([make_chunk("doc1", 0, "公司营收")])
    assert seen == [["公司营收"]]


def test_add_then_query_returns_nearest_chunk(tmp_path):
    store = make_store(tmp_path)
    store.add([
        make_chunk("doc1", 0, "公司营收 100 亿元"),
        make_chunk("doc1", 1, "员工人数 5000 人"),
    ])
    hits = store.query("营收", top_k=2)
    assert hits[0].chunk_id == "doc1#c0000"
    assert hits[0].retriever == "vector"
    assert hits[0].score > hits[1].score


def test_hit_carries_provenance_metadata(tmp_path):
    """溯源字段必须随命中返回：没有 doc_id / source 就无法做引用支撑率统计（spec §6.2 第 2 条）。"""
    store = make_store(tmp_path)
    store.add([make_chunk("doc1", 0, "公司营收 100 亿元")])
    hit = store.query("营收", top_k=1)[0]
    assert hit.doc_id == "doc1"
    assert hit.source == "doc1.pdf"


def test_query_filters_by_doc_ids(tmp_path):
    store = make_store(tmp_path)
    store.add([make_chunk("doc1", 0, "营收 A"), make_chunk("doc2", 0, "营收 B")])
    hits = store.query("营收", top_k=5, doc_ids=["doc2"])
    assert [h.chunk_id for h in hits] == ["doc2#c0000"]


def test_add_is_upsert_not_duplicate(tmp_path):
    store = make_store(tmp_path)
    chunk = make_chunk("doc1", 0, "营收 100")
    store.add([chunk])
    store.add([chunk])
    assert store.count() == 1


def test_delete_ids_and_delete_by_doc(tmp_path):
    store = make_store(tmp_path)
    store.add([make_chunk("doc1", 0, "a"), make_chunk("doc1", 1, "b"), make_chunk("doc2", 0, "c")])
    assert store.delete_ids(["doc1#c0000"]) == 1
    assert store.delete_by_doc("doc1") == 1
    assert store.count() == 1


def test_empty_query_on_empty_collection_returns_empty(tmp_path):
    assert make_store(tmp_path).query("营收") == []


# ---------------------------------------------------------------------------
# id_map：细粒度索引（`data/chroma-v3` 的 unit）折回父块 id
#
# 这组用例盯的是「unit 级库能不能和块级库放同一个口径下比」：返回的 `chunk_id` 必须是
# 父块 id（否则评测的 gold、RRF 的分支身份都要重造），且同一个父块只许占一个名次。
# ---------------------------------------------------------------------------


def angle(degrees: float) -> tuple[float, float]:
    """单位圆上的一点：固定查询向量 `[1, 0]` 时，名次 = 角度升序（余弦距离单调）。"""

    rad = math.radians(degrees)
    return (math.cos(rad), math.sin(rad))


def text_embed(spec: dict[str, tuple[float, float]]):
    """按**文本**查表的确定性嵌入：名次完全由用例声明，不依赖任何模型。"""

    def embed(texts: list[str]) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in (spec[text] for text in texts)]

    return embed


def make_raw_chunk(chunk_id: str, text: str, ordinal: int = 0) -> Chunk:
    """只关心 `chunk_id` / `text` 的块：id 与父块/unit 命名空间无关，由用例自己给。"""

    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        source="doc1.pdf",
        ordinal=ordinal,
        text=text,
        content_hash=f"hash{chunk_id}",
        section_path="第一章",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def test_id_map_folds_units_back_to_their_parent_chunk(tmp_path):
    """同归属的多条 unit 只留名次最好那条，且 `chunk_id` 换成父块 id。"""

    spec = {"查询": angle(0), "营收 A": angle(5), "营收 A 续": angle(10), "营收 B": angle(30)}
    store = VectorStore(
        tmp_path / "chroma-v3",
        text_embed(spec),
        id_map={"doc1#u0001": "doc1#c0001", "doc1#u0002": "doc1#c0001", "doc2#u0001": "doc2#c0001"},
    )
    store.add([
        make_raw_chunk("doc1#u0001", "营收 A"),
        make_raw_chunk("doc1#u0002", "营收 A 续", 1),
        make_raw_chunk("doc2#u0001", "营收 B"),
    ])

    hits = store.query("查询", top_k=2)
    assert [hit.chunk_id for hit in hits] == ["doc1#c0001", "doc2#c0001"]
    # 留下的是同一归属里名次最好的那条（angle(5) 的 unit），分数与 text 都跟着它走。
    assert hits[0].text == "营收 A"
    assert hits[0].score > hits[1].score


def test_id_map_deepens_the_fetch_to_fill_top_k_distinct_owners(tmp_path):
    """前排被同一父块的 unit 挤满时，必须往后多取才凑得够 top_k 个不同归属。"""

    spec = {
        "查询": angle(0),
        "a1": angle(2),
        "a2": angle(4),
        "a3": angle(6),
        "b1": angle(40),
        "c1": angle(60),
    }
    store = VectorStore(
        tmp_path / "chroma-v3",
        text_embed(spec),
        id_map={"a1": "P1", "a2": "P1", "a3": "P1", "b1": "P2", "c1": "P3"},
    )
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "a3", "b1", "c1")])

    # 只取 2 条原始行的话两条都是 P1 的 unit，折叠后只剩 1 个归属——必须取到 4 条。
    assert [hit.chunk_id for hit in store.query("查询", top_k=2)] == ["P1", "P2"]
    assert [hit.chunk_id for hit in store.query("查询", top_k=3)] == ["P1", "P2", "P3"]
    # 库里的行数不够时就是取完了，不做任何补偿（也不许报错）。
    assert [hit.chunk_id for hit in store.query("查询", top_k=9)] == ["P1", "P2", "P3"]


def test_without_id_map_the_row_ids_pass_through_unchanged(tmp_path):
    """不开 `id_map` 时口径必须与本仓历史索引逐字一致：索引 id 就是命中 id。"""

    spec = {"查询": angle(0), "a1": angle(2), "a2": angle(4), "b1": angle(40)}
    store = VectorStore(tmp_path / "chroma", text_embed(spec))
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "b1")])

    assert [hit.chunk_id for hit in store.query("查询", top_k=3)] == ["a1", "a2", "b1"]


# ---------------------------------------------------------------------------
# 折叠口径：`fold="rrf_sum"`（父块聚合）与默认的 `"best"`（历史行为）
#
# 这组用例盯的是「同一个父块被多条 unit 命中时，该不该只按它最好的那条 unit 站队」。
# 默认必须是逐字的历史行为；聚合口径要能把「多条 unit 都在头部」的父块顶上去，同时
# 不许改 `score` 的语义（余弦相似度），也不许把返回条数撑大。
# ---------------------------------------------------------------------------


def test_fold_default_is_the_historical_best_and_unknown_modes_are_rejected(tmp_path):
    """口径是白名单：拼错一个字母必须报错，不能静默换一条检索路径。"""

    from akr.storage import vector_store as module

    assert module.FOLD_DEFAULT == "best"
    assert module.FOLD_MODES == frozenset({"best", "rrf_sum"})
    with pytest.raises(ValueError, match="折叠口径"):
        VectorStore(tmp_path / "chroma", fake_embed, fold="magic")
    with pytest.raises(ValueError, match="fold_window"):
        VectorStore(tmp_path / "chroma", fake_embed, fold_window=0)


def test_best_fold_ranks_a_parent_by_its_single_best_unit(tmp_path):
    """`best` 口径：P1 的三条 unit 全在第 2~4 名，却仍排在只有一条 unit 的 P2 后面。"""

    spec = {
        "查询": angle(0),
        "a1": angle(10),
        "a2": angle(12),
        "a3": angle(14),
        "b1": angle(5),
    }
    store = VectorStore(
        tmp_path / "chroma-v3",
        text_embed(spec),
        id_map={"a1": "P1", "a2": "P1", "a3": "P1", "b1": "P2"},
    )
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "a3", "b1")])

    assert [hit.chunk_id for hit in store.query("查询", top_k=2)] == ["P2", "P1"]


def test_aggregate_fold_lifts_the_parent_whose_units_all_land_in_the_head(tmp_path):
    """`rrf_sum` 口径：同一个父块的三条 unit 累加分超过「只有一条更靠前的 unit」的 P2。"""

    spec = {
        "查询": angle(0),
        "a1": angle(10),
        "a2": angle(12),
        "a3": angle(14),
        "b1": angle(5),
    }
    store = VectorStore(
        tmp_path / "chroma-v3",
        text_embed(spec),
        id_map={"a1": "P1", "a2": "P1", "a3": "P1", "b1": "P2"},
        fold="rrf_sum",
    )
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "a3", "b1")])

    hits = store.query("查询", top_k=2)

    assert [hit.chunk_id for hit in hits] == ["P1", "P2"]
    # 证据行与 `score` 仍跟着该归属**最好**的那条 unit：score 的语义没说变就变。
    assert hits[0].text == "a1"
    # Chroma 以 float32 落库，比较容差按 float32 精度给。
    assert hits[0].score == pytest.approx(math.cos(math.radians(10)), abs=1e-6)
    assert hits[0].doc_id == "doc1" and hits[0].source == "doc1.pdf"
    assert hits[0].retriever == "vector"


def test_aggregate_fold_without_id_map_keeps_the_raw_row_order(tmp_path):
    """聚合只在「一个归属有多条行」时才有区别；没有 `id_map` 时每个索引行各自成归属。"""

    spec = {"查询": angle(0), "a1": angle(2), "a2": angle(4), "b1": angle(40)}
    store = VectorStore(tmp_path / "chroma", text_embed(spec), fold="rrf_sum")
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "b1")])

    assert [hit.chunk_id for hit in store.query("查询", top_k=3)] == ["a1", "a2", "b1"]


def test_aggregate_fold_never_pads_the_list_to_top_k(tmp_path):
    """只有 2 个归属就必须只返回 2 条：凑不满 `top_k` 时不做任何补偿。"""

    spec = {"查询": angle(0), "a1": angle(10), "a2": angle(12), "b1": angle(20)}
    store = VectorStore(
        tmp_path / "chroma-v3",
        text_embed(spec),
        id_map={"a1": "P1", "a2": "P1", "b1": "P2"},
        fold="rrf_sum",
    )
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "b1")])

    assert len(store.query("查询", top_k=9)) == 2


def test_fold_window_caps_which_rows_take_part_in_the_aggregate(tmp_path):
    """`fold_window` 是参与聚合的原始行数上限，不是「至少取这么多」。

    窗口只有 2 时，第 3 名那条 P2 的 unit 根本没进过打分表——它不会以「分低」的形状出现，
    而是以「看不见」的形状出现，所以这里断言的是**条数**。反过来，`top_k` 比窗口大时窗口
    要跟着放大，否则返回条数会不够（第二条断言）。
    """

    spec = {"查询": angle(0), "a1": angle(10), "a2": angle(12), "b1": angle(20)}
    ids = {"a1": "P1", "a2": "P1", "b1": "P2"}
    store = VectorStore(
        tmp_path / "chroma-v3",
        text_embed(spec),
        id_map=ids,
        fold="rrf_sum",
        fold_window=2,
    )
    store.add([make_raw_chunk(name, name) for name in ("a1", "a2", "b1")])

    assert [hit.chunk_id for hit in store.query("查询", top_k=2)] == ["P1"]
    assert [hit.chunk_id for hit in store.query("查询", top_k=5)] == ["P1", "P2"]
