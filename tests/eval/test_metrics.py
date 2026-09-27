import pytest

from eval.metrics import (
    DEFAULT_K,
    MetricInput,
    aggregate,
    citation_support_rate,
    cohen_kappa,
    entity_link_accuracy,
    hallucination_rate,
    mrr,
    per_call_recall_at_k,
    percentile,
    recall_at_k,
    refusal_metrics,
    tool_call_success_rate,
)


def test_recall_at_k_and_top_k_slicing():
    retrieved = ["a", "b", "c", "d"]
    gold = ["b", "d"]

    assert recall_at_k(retrieved, gold) == 1.0
    assert recall_at_k(retrieved, gold, k=2) == 0.5   # 只看前 2 条，只命中 b
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert DEFAULT_K == 8


def test_recall_at_k_is_vacuous_for_empty_gold():
    """拒答题没有 gold chunk；它的检索指标由 aggregate 过滤，函数本身不抛异常。"""
    assert recall_at_k(["a"], [], k=3) == 1.0


def test_per_call_recall_is_unmeasurable_when_no_call_happened():
    """一次检索都没发生 → None（不可测），不是 0.0：0 在这套指标里是真值。"""
    assert per_call_recall_at_k((), ["a1"]) is None
    # 有调用时，空 gold 仍走 recall_at_k 的 vacuous 口径，保证与主表同源
    assert per_call_recall_at_k([["a1"]], []) == 1.0
    with pytest.raises(ValueError):
        per_call_recall_at_k([["a1"]], ["a1"], k=0)


def test_per_call_recall_counts_each_call_separately_not_the_truncated_union():
    """逐次口径要看得见"后面那轮检索补回的 gold"——主表前 K 截断会把它抹掉。"""
    calls = (("x1", "x2"), ("a1",))
    union = ("x1", "x2", "a1")

    assert recall_at_k(union, ["a1"], k=2) == 0.0        # 主表口径：前 2 条没有 gold
    assert per_call_recall_at_k(calls, ["a1"], k=2) == pytest.approx(0.5)   # (0 + 1) / 2


def test_aggregate_reports_per_call_recall_without_touching_the_main_column():
    """主表 `recall@K` 一字不动；逐次口径作为**并列新键**追加，且不可测时不加键。"""
    rows = [
        MetricInput(
            task_id="t1",
            type="single_hop",
            answerable=True,
            retrieved_ids=("x1", "x2", "a1"),
            retrieved_per_call=(("x1", "x2"), ("a1",)),
            gold_ids=("a1",),
        ),
        MetricInput(
            task_id="t2",
            type="single_hop",
            answerable=True,
            retrieved_ids=("x3", "x4", "a2"),
            retrieved_per_call=(("x3", "x4"), ("a2",)),
            gold_ids=("a2",),
        ),
    ]

    summary = aggregate(rows, k=2)

    assert summary["recall@2"] == 0.0                    # 主表口径：两题的前 2 条都没命中
    assert summary["per_call_recall@2"] == pytest.approx(0.5)

    no_per_call = aggregate(
        [
            MetricInput(
                task_id="t3",
                type="single_hop",
                answerable=True,
                retrieved_ids=("a3",),
                gold_ids=("a3",),
            )
        ],
        k=2,
    )
    assert no_per_call["recall@2"] == 1.0
    assert "per_call_recall@2" not in no_per_call       # 没有逐次数据 → 不加键，不填 0.0


def test_mrr_returns_reciprocal_rank_of_first_hit():
    assert mrr(["x", "a", "b"], ["a"]) == pytest.approx(1 / 2)
    assert mrr(["a"], ["a"]) == 1.0
    assert mrr(["x", "y"], ["a"]) == 0.0
    assert mrr(["a"], []) == 0.0


def test_citation_support_rate_and_hallucination_are_complementary():
    assert citation_support_rate([True, False, True]) == pytest.approx(2 / 3)
    assert hallucination_rate([True, False, True]) == pytest.approx(1 / 3)
    # 空答案：没有断言也就没有幻觉断言，但这条"空答案刷分"的路由成对拒答指标堵住
    assert citation_support_rate([]) == 1.0
    assert hallucination_rate([]) == 0.0


def test_refusal_metrics_returns_both_rates_together():
    expected_refusal = [True, True, False, False, False]
    answered = [False, True, False, True, True]   # 第 1 条正确拒答；第 3 条是误拒

    metrics = refusal_metrics(expected_refusal, answered)

    assert metrics.total_expected_refusal == 2
    assert metrics.correct_refusal == 1
    assert metrics.correct_refusal_rate == pytest.approx(0.5)
    assert metrics.total_answerable == 3
    assert metrics.over_refusal == 1
    assert metrics.over_refusal_rate == pytest.approx(1 / 3)


def test_refusal_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        refusal_metrics([True], [False, True])


def test_cohen_kappa_on_perfect_and_chance_level_agreement():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert cohen_kappa([True] * 4, [True] * 4) == 1.0   # 常量标注：pe=1 且完全一致
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([7], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_component_rates():
    assert tool_call_success_rate([True, True, False, True]) == pytest.approx(0.75)
    assert tool_call_success_rate([]) == 0.0
    assert entity_link_accuracy(["甲", "乙"], ["甲", "丙"]) == pytest.approx(0.5)
