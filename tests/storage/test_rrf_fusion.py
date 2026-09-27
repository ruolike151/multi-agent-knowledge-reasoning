import pytest

from akr.storage.fusion import DEFAULT_ALPHA, cosine_rerank, rrf_fuse, score_rerank
from akr.storage.hits import RetrievalHit


def hit(chunk_id: str, score: float, retriever: str) -> RetrievalHit:
    doc_id = chunk_id.split("#")[0]
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        text=f"text of {chunk_id}",
        score=score,
        retriever=retriever,
    )


def test_single_list_uses_reciprocal_rank_scores():
    fused = rrf_fuse([[hit("doc1#c0000", 0.9, "vector"), hit("doc1#c0001", 0.5, "vector")]])
    assert [item.chunk_id for item in fused] == ["doc1#c0000", "doc1#c0001"]
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)
    assert fused[0].retriever == "vector"


def test_chunk_in_both_lists_is_marked_both_and_ranks_first():
    vector = [hit("doc1#c0000", 0.9, "vector"), hit("doc1#c0001", 0.8, "vector")]
    bm25 = [hit("doc1#c0001", 12.0, "bm25")]
    fused = rrf_fuse([vector, bm25])
    assert fused[0].chunk_id == "doc1#c0001"
    assert fused[0].retriever == "both"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].retriever == "vector"


def test_ties_are_broken_by_chunk_id_ascending():
    fused = rrf_fuse([[hit("doc2#c0000", 1.0, "vector")], [hit("doc1#c0000", 5.0, "bm25")]])
    assert [item.chunk_id for item in fused] == ["doc1#c0000", "doc2#c0000"]


def test_top_k_truncates_after_sorting():
    vector = [hit(f"doc1#c{i:04d}", 1.0 - i / 10, "vector") for i in range(5)]
    assert [item.chunk_id for item in rrf_fuse([vector], top_k=2)] == ["doc1#c0000", "doc1#c0001"]


def test_empty_input_returns_empty_list():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def fused_pair() -> list[RetrievalHit]:
    """两条按 RRF 排好序的命中：`doc1#c0000` 名次 1、`doc1#c0001` 名次 2。

    分数是 1/61 与 1/62（`DEFAULT_K=60` 的固定后果），下面几条断言直接按这个事实写。
    """

    return rrf_fuse([[hit("doc1#c0000", 0.9, "vector"), hit("doc1#c0001", 0.5, "vector")]])


def test_cosine_rerank_with_alpha_one_orders_by_cosine_alone():
    """alpha=1 是完全交给余弦：RRF 只决定谁在候选里，不决定谁在前面。"""

    cosine = [hit("doc1#c0001", 0.9, "vector"), hit("doc1#c0000", 0.1, "vector")]
    ranked = cosine_rerank(fused_pair(), cosine, alpha=1.0)
    assert [item.chunk_id for item in ranked] == ["doc1#c0001", "doc1#c0000"]


def test_cosine_rerank_with_alpha_zero_keeps_the_fused_order():
    """alpha=0 是恒等：这条守住「开重排但把权重调到 0」等于没开。"""

    cosine = [hit("doc1#c0001", 0.9, "vector"), hit("doc1#c0000", 0.1, "vector")]
    ranked = cosine_rerank(fused_pair(), cosine, alpha=0.0)
    assert [item.chunk_id for item in ranked] == ["doc1#c0000", "doc1#c0001"]


def test_cosine_rerank_blends_the_normalised_rrf_score_with_bounded_cosine():
    """混合公式写死在这里，防止以后有人「顺手」改成量纲不同的加权和。"""

    cosine = [hit("doc1#c0001", 1.0, "vector"), hit("doc1#c0000", 0.0, "vector")]
    ranked = cosine_rerank(fused_pair(), cosine, alpha=0.5)
    # c0001：(1-0.5)*(1/62 ÷ 1/61) + 0.5*1 = 0.5*(61/62) + 0.5
    # c0000：(1-0.5)*1 + 0.5*0 = 0.5
    assert ranked[0].chunk_id == "doc1#c0001"
    assert ranked[0].score == pytest.approx(0.5 * (61 / 62) + 0.5, abs=1e-9)
    assert ranked[1].score == pytest.approx(0.5, abs=1e-9)


def test_negative_cosine_is_clamped_to_zero_not_penalised():
    """负相似度当作「没有信号」，不当成负分：反向语义不代表这句更该排在后面。"""

    cosine = [hit("doc1#c0001", -0.9, "vector"), hit("doc1#c0000", 0.0, "vector")]
    ranked = cosine_rerank(fused_pair(), cosine, alpha=0.5)
    assert [item.chunk_id for item in ranked] == ["doc1#c0000", "doc1#c0001"]
    assert ranked[1].score == pytest.approx(0.5 * (61 / 62), abs=1e-9)


def test_hits_absent_from_cosine_keep_their_relative_order_and_rank_last():
    """缺余弦分的命中不给凑出来的 0 分：它们排在所有有信号的命中之后，内部序不变。"""

    fused = rrf_fuse(
        [
            [
                hit("doc1#c0000", 0.9, "vector"),
                hit("doc1#c0001", 0.8, "vector"),
                hit("doc1#c0002", 0.7, "vector"),
            ]
        ]
    )
    ranked = cosine_rerank(fused, [hit("doc1#c0002", 0.4, "vector")], alpha=0.5)
    assert [item.chunk_id for item in ranked] == ["doc1#c0002", "doc1#c0000", "doc1#c0001"]


def test_empty_cosine_returns_the_fused_order_and_scores_untouched():
    fused = fused_pair()
    ranked = cosine_rerank(fused, [])
    assert [item.chunk_id for item in ranked] == [item.chunk_id for item in fused]
    assert [item.score for item in ranked] == [item.score for item in fused]


def test_empty_fused_returns_empty_list():
    assert cosine_rerank([], [hit("doc1#c0000", 0.9, "vector")]) == []


def test_top_k_truncates_after_reranking():
    cosine = [hit("doc1#c0001", 0.9, "vector"), hit("doc1#c0000", 0.1, "vector")]
    ranked = cosine_rerank(fused_pair(), cosine, alpha=1.0, top_k=1)
    assert [item.chunk_id for item in ranked] == ["doc1#c0001"]


def test_provenance_and_retriever_come_from_the_fused_hit():
    """重排只改分数与顺序：`text` / `doc_id` / `source` / `retriever` 一律沿用融合结果。

    `retriever` 尤其不能改：`akr.mcp_server.schemas.Hit.retriever` 是
    `Literal["vector","bm25","both"]`，重排给自己编一个新取值会直接让 MCP 层校验失败。
    """

    fused = [hit("doc1#c0000", 1 / 61, "both")]
    ranked = cosine_rerank(fused, [hit("doc1#c0000", 0.7, "vector")], alpha=1.0)
    assert ranked[0].doc_id == "doc1"
    assert ranked[0].source == "doc1.pdf"
    assert ranked[0].text == "text of doc1#c0000"
    assert ranked[0].retriever == "both"


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_alpha_outside_the_unit_interval_is_rejected(alpha):
    """权重越界必须报错，不能靠 `min/max` 悄悄夹回去——那样调参结果无法解释。"""

    with pytest.raises(ValueError):
        cosine_rerank(fused_pair(), [], alpha=alpha)


def test_duplicate_chunk_ids_in_fused_are_rejected():
    """`fused` 里同一个 chunk 出现两次时，「按 chunk_id 查余弦分」是二义的：宁可报错。"""

    fused = [hit("doc1#c0000", 1 / 61, "vector"), hit("doc1#c0000", 1 / 62, "bm25")]
    with pytest.raises(ValueError):
        cosine_rerank(fused, [hit("doc1#c0000", 0.5, "vector")])


def test_default_alpha_is_half():
    assert DEFAULT_ALPHA == 0.5


# --- 外部打分重排（cross-encoder，`rerank="cross"`） -------------------------


def test_score_rerank_orders_by_the_external_score_alone():
    """顺序**完全**由重排分决定：RRF 名次一个数都不参与，这是与 cosine 重排的本质差别。

    同一个候选集：RRF 说 c0000 在前，重排分说 c0001 在前——只看这两条，就知道顺序换了。
    """

    fused = fused_pair()
    assert [item.chunk_id for item in fused] == ["doc1#c0000", "doc1#c0001"]

    ranked = score_rerank(fused, [0.10, 0.95])

    assert [item.chunk_id for item in ranked] == ["doc1#c0001", "doc1#c0000"]
    # score 写的是**重排分**（与 `cosine_rerank` 写混合分同一口径），不是 RRF 分。
    assert [item.score for item in ranked] == [0.95, 0.10]


def test_a_tie_keeps_the_fused_order_not_the_chunk_id_order():
    """同分时保持 `fused` 里的先后（那是 RRF 的结论），不是按 chunk_id 排。

    `rrf_fuse` 的同分兜底恰好是按 chunk_id 升序，所以这里刻意构造「fused 序与 chunk_id 序
    相反」的两条：只有真的拿原下标当兜底才过得了。
    """

    fused = [hit("doc2#c0000", 1 / 61, "vector"), hit("doc1#c0000", 1 / 61, "vector")]
    ranked = score_rerank(fused, [0.5, 0.5])
    assert [item.chunk_id for item in ranked] == ["doc2#c0000", "doc1#c0000"]


def test_scores_must_line_up_one_to_one_with_the_candidates():
    """条数不等直接报错：少一条只可能是调用方把两份列表错配了，不能猜。"""

    with pytest.raises(ValueError, match="一一对应"):
        score_rerank(fused_pair(), [0.5])
    with pytest.raises(ValueError, match="一一对应"):
        score_rerank(fused_pair(), [0.5, 0.4, 0.3])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_scores_are_rejected(bad):
    """NaN 进排序键会让名次取决于比较次数——那是不可复现的返回，必须报错。"""

    with pytest.raises(ValueError, match="有限数"):
        score_rerank(fused_pair(), [bad, 0.5])


def test_score_rerank_truncates_after_ordering():
    ranked = score_rerank(fused_pair(), [0.1, 0.9], top_k=1)
    assert [item.chunk_id for item in ranked] == ["doc1#c0001"]


def test_score_rerank_keeps_provenance_and_retriever():
    """与 `cosine_rerank` 同一条纪律：只改分与序，`retriever` 不能自造取值。"""

    fused = [hit("doc1#c0000", 1 / 61, "both")]
    ranked = score_rerank(fused, [0.5])
    assert ranked[0].doc_id == "doc1"
    assert ranked[0].source == "doc1.pdf"
    assert ranked[0].text == "text of doc1#c0000"
    assert ranked[0].retriever == "both"


def test_score_rerank_of_an_empty_pool_is_empty():
    assert score_rerank([], []) == []
