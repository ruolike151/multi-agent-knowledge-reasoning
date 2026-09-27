"""评测指标层：纯函数、零 import、零 IO。

为什么单独一层：三组系统（Baseline A / Baseline B / 目标系统）跑的是同一批题，
必须穿过同一套度量函数，对照表才是"同一个尺子量三次"。任何 runner 自己算一次
「差不多」的 recall，都会让对比退化成两次不同测量的拼凑。

本模块**刻意不 import 项目内任何模块**：指标只认最朴素的输入（id 列表、布尔列表、
浮点数）。这样它是稳定的——数据层、模型层、编排层怎么重构都不会影响它。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# 检索指标的默认截断位：与工具层 search_documents 的默认 top_k 保持一致（spec §6.1）
DEFAULT_K = 8


def _check_same_length(*groups: Sequence[object]) -> None:
    """长度不一致直接抛错：指标算错比算不出来危险得多（错了没人会怀疑）。"""
    lengths = {len(g) for g in groups}
    if len(lengths) > 1:
        raise ValueError(f"输入序列长度不一致：{sorted(lengths)}")


def recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    *,
    k: int | None = None,
) -> float:
    """标准 Recall@K：命中数 / gold 总数。

    gold 为空时返回 1.0（vacuous truth）。拒答题本来就没有 gold chunk，
    它不该参与检索指标——过滤职责在 aggregate()，函数本身保持数学上的自洽。
    """
    top_k = DEFAULT_K if k is None else k
    if top_k <= 0:
        raise ValueError(f"k 必须是正整数，收到 {top_k}")
    if not gold_ids:
        return 1.0
    gold = set(gold_ids)
    hits = sum(1 for doc_id in retrieved_ids[:top_k] if doc_id in gold)
    return hits / len(gold)


def mrr(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """MRR：第一条命中结果的排名倒数，全不命中记 0.0。

    gold 为空返回 0.0 而不是 1.0：空 gold 的题在检索上"没有可命中的目标"，
    给 1.0 会让拒答题把 MRR 拉高。
    """
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def per_call_recall_at_k(
    calls: Sequence[Sequence[str]],
    gold_ids: Sequence[str],
    *,
    k: int | None = None,
) -> float | None:
    """「逐次检索调用」口径的 Recall@K：先算每次调用各自的 recall@K，再对调用取均值。

    为什么单开一个口径：主表 `recall_at_k` 吃的是「一次运行内所有调用去重保序后
    的前 K 条」，这在首查就把 K 个名额填满时，会把 verify 轮 / 图遍历轮再查到的
    结果全部截掉——多轮检索的收益在指标上看不见。本函数把「每次调用」当成一次
    独立的检索测量，于是「第二次检索补回了第一轮漏掉的 gold」会体现在数字上。

    返回 `None` 而不是 0.0 表示**不可测**（这道题一次检索都没发生，例如拒答题
    或纯图查询）。0.0 在这里是有意义的值（真的每次都全错），不能被"没数据"冒用。
    """
    top_k = DEFAULT_K if k is None else k
    if top_k <= 0:
        raise ValueError(f"k 必须是正整数，收到 {top_k}")
    if not calls:
        return None
    per_call = [recall_at_k(call, gold_ids, k=top_k) for call in calls]
    return sum(per_call) / len(per_call)


def tool_call_success_rate(successes: Sequence[bool]) -> float:
    """工具调用成功率：一次都没调用记 0.0（不是 1.0——空集在这里没有可辩护的乐观口径）。"""
    if not successes:
        return 0.0
    return sum(1 for ok in successes if ok) / len(successes)


def entity_link_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """实体链接准确率：逐位比对预测实体名与标准实体名。"""
    if not predicted and not gold:
        return 0.0
    _check_same_length(predicted, gold)
    if not gold:
        return 0.0
    return sum(1 for p, g in zip(predicted, gold, strict=True) if p == g) / len(gold)


def citation_support_rate(supports: Sequence[bool]) -> float:
    """引用支撑率：被原文支撑的断言 / 全部断言。

    断言切分口径唯一来源是 Verifier 的 split_claims（Task 30）；本函数只接收
    每条断言的蕴含判定结果。空断言列表记 1.0——空答案没有幻觉，"靠不写答案刷分"
    这条路由成对拒答指标堵住。
    """
    if not supports:
        return 1.0
    return sum(1 for ok in supports if ok) / len(supports)


def hallucination_rate(supports: Sequence[bool]) -> float:
    """幻觉率：与引用支撑率严格互补，避免两处各算一次导致加不到 1。"""
    return 1.0 - citation_support_rate(supports)


@dataclass(frozen=True)
class RefusalMetrics:
    """拒答指标：两个率必须成对返回。

    只报"正确拒答率"，系统就能靠"一律拒答"刷分；只报"过度拒答率"，系统就能靠
    "有问必答"刷分。把四个原始计数和两个率绑在一个冻结对象里，调用方拿不到
    "只有一半"的指标，想解释任何一个率都必须同时面对它的代价。
    """

    total_expected_refusal: int
    correct_refusal: int
    total_answerable: int
    over_refusal: int
    correct_refusal_rate: float
    over_refusal_rate: float


def refusal_metrics(
    expected_refusal: Sequence[bool],
    answered: Sequence[bool],
) -> RefusalMetrics:
    """统计拒答行为。

    expected_refusal[i] 为 True 表示第 i 题**标准答案就是拒答**（语料里查不到）；
    answered[i] 为 True 表示系统给出了答案。
    """
    _check_same_length(expected_refusal, answered)

    total_expected_refusal = sum(1 for flag in expected_refusal if flag)
    total_answerable = len(expected_refusal) - total_expected_refusal
    correct_refusal = sum(
        1
        for should_refuse, did_answer in zip(expected_refusal, answered, strict=True)
        if should_refuse and not did_answer
    )
    over_refusal = sum(
        1
        for should_refuse, did_answer in zip(expected_refusal, answered, strict=True)
        if not should_refuse and not did_answer
    )

    return RefusalMetrics(
        total_expected_refusal=total_expected_refusal,
        correct_refusal=correct_refusal,
        total_answerable=total_answerable,
        over_refusal=over_refusal,
        # 分母为 0 时记 0.0：任务集里没有该类题，这个率就不该有值
        correct_refusal_rate=(
            correct_refusal / total_expected_refusal if total_expected_refusal else 0.0
        ),
        over_refusal_rate=over_refusal / total_answerable if total_answerable else 0.0,
    )


def cohen_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    """Cohen's κ：两名标注者（或 judge 与人工）的一致性，已扣除随机一致的部分。

    用途是 judge 校准（spec §9.5）：judge 与人工的一致率会虚高（都倾向打"通过"），
    κ 把"随机也会一致"的那部分减掉，才看得出 judge 是否真的能替代人工。

    pe == 1 表示双方标注分布退化成单一类别：此时 po 必然是 1（否则长度或分布矛盾），
    直接返回 1.0，避免 0/0。
    """
    _check_same_length(labels_a, labels_b)
    if not labels_a:
        raise ValueError("标签订单为空，无法计算 κ")

    total = len(labels_a)
    po = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / total

    categories = set(labels_a) | set(labels_b)
    pe = 0.0
    for category in categories:
        p_a = sum(1 for label in labels_a if label == category) / total
        p_b = sum(1 for label in labels_b if label == category) / total
        pe += p_a * p_b

    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1.0 - pe)


def percentile(values: Sequence[float], q: float) -> float:
    """分位数，线性插值，口径与 numpy.percentile 默认值一致。

    口径对齐是为了"用 numpy 复核能对上"：P95 在 12 条样本上本来就粗糙
    （spec §9.4 已声明方差来自子集外推），如果连插值方式都不一样，
    报告里的复现步骤就会自相矛盾。
    """
    if not values:
        raise ValueError("空序列没有分位数")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q 必须在 [0, 100] 内，收到 {q}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass(frozen=True)
class MetricInput:
    """一行实验记录里指标层关心的所有字段。

    这是 metrics 层自己的类型，**不是 Task 35 的 `RunRecord`**：依赖方向必须是
    "runner → metrics"。如果这里 import 实验层，纯函数就长出了上游，三组系统
    复用同一套口径的前提也就没了。
    """

    task_id: str
    type: str
    answerable: bool
    retrieved_ids: tuple[str, ...] = ()
    # 逐次 search_documents 调用各自的 chunk_id（去重保序的坑在 retrieved_ids 上，
    # 这一份保留"每次调用看到了什么"）。缺省空元组 = 这道题没有逐次数据（旧 JSONL 回读）。
    retrieved_per_call: tuple[tuple[str, ...], ...] = ()
    gold_ids: tuple[str, ...] = ()
    supports: tuple[bool, ...] = ()
    answered: bool = True
    judge_score: float | None = None
    latency_ms: float = 0.0
    cost_cny: float = 0.0
    steps: int = 0
    illegal_transitions: int = 0


def _mean(values: Sequence[float]) -> float:
    """空集合记 0.0，并让调用方知道"这一组没有样本"是通过 0 而不是 1.0 表达的。"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate(rows: Sequence[MetricInput], *, k: int | None = None) -> dict[str, float]:
    """把一行行实验记录汇总成对照表要用的标量。

    两条关键口径：
    1. **检索指标只在 `answerable` 的行上算**——拒答题没有 gold chunk，混进来会
       凭空抬高或压低 recall；
    2. **没有任何行直接抛 `ValueError`**——不允许生成一份"全 0"的假表，
       因为 0 在这套指标里是有意义的值（真的全错），不能被"没数据"冒用。
    """
    if not rows:
        raise ValueError("没有任何实验记录，拒绝生成全 0 的假对照表")

    top_k = DEFAULT_K if k is None else k
    answerable_rows = [row for row in rows if row.answerable]

    recalls = [
        recall_at_k(row.retrieved_ids, row.gold_ids, k=top_k) for row in answerable_rows
    ]
    reciprocal_ranks = [mrr(row.retrieved_ids, row.gold_ids) for row in answerable_rows]

    # 逐次调用口径（与主表 `recall@K` 并列，不改主表口径）：把"这一题每次检索各自
    # 的 recall@K"先按调用求均值，再按题求均值。只有在题目确实带逐次数据时才产出
    # 这个键——一道都没有就**不加键**，而不是填 0.0 假装测过（0 在这里是真值）。
    per_call_values = [
        value
        for value in (
            per_call_recall_at_k(row.retrieved_per_call, row.gold_ids, k=top_k)
            for row in answerable_rows
        )
        if value is not None
    ]

    # 断言级指标把所有行的断言摊平：越长的答案在支撑率上权重越大，
    # 这是想要的——它衡量的是"整个系统输出的字里有多少是编的"
    all_supports = [flag for row in rows for flag in row.supports]
    refusal = refusal_metrics(
        [not row.answerable for row in rows],
        [row.answered for row in rows],
    )

    judge_scores = [row.judge_score for row in rows if row.judge_score is not None]
    latencies = [row.latency_ms for row in rows]
    costs = [row.cost_cny for row in rows]

    summary: dict[str, float] = {
        f"recall@{top_k}": _mean(recalls),
        "mrr": _mean(reciprocal_ranks),
        "citation_support_rate": citation_support_rate(all_supports),
        "hallucination_rate": hallucination_rate(all_supports),
        "correct_refusal_rate": refusal.correct_refusal_rate,
        "over_refusal_rate": refusal.over_refusal_rate,
        "judge_score_mean": _mean(judge_scores),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "cost_cny_mean": _mean(costs),
        "cost_cny_total": float(sum(costs)),
        "steps_mean": _mean([float(row.steps) for row in rows]),
        "illegal_transitions_total": float(
            sum(row.illegal_transitions for row in rows)
        ),
    }
    if per_call_values:
        summary[f"per_call_recall@{top_k}"] = _mean(per_call_values)
    return summary
