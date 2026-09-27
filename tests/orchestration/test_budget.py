import dataclasses
import json

import pytest

from akr.orchestration.budget import (
    EXHAUSTION_ORDER,
    Budget,
    BudgetState,
)


def fake_clock(values):
    """按顺序吐值的假时钟：单测里不做真实 sleep，墙钟维度才能被确定性地测。

    最后一个值会被反复返回，这样 `elapsed_seconds()` 多次调用结果稳定。
    """

    queue = list(values)

    def clock() -> float:
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    return clock


def test_budget_defaults_are_the_2026_09_26_retune():
    """默认上限按 15 格实测重调过一次（偏差日志第四十二节）。

    spec §5.6 的原值是 `max_tokens=120_000` / `max_cost_cny=0.5`；15 格端到端实测
    单题用量中位数就压在这个盒子上沿（最贵一题 123,852 tokens / ¥0.6166），触顶的
    2 条全是 `cost_cny`，被掐掉的题直接退化成 0 分——所以抬盒子本身是一条质量修复，
    不是「放宽了就能多跑」的稀释。`max_wall_seconds` 刻意不动（用户口令：先不管墙钟）。
    """

    budget = Budget()
    assert budget.max_steps == 24
    assert budget.max_llm_calls == 40
    assert budget.max_tokens == 200_000
    assert budget.max_cost_cny == 1.0
    assert budget.max_wall_seconds == 120


def test_budget_limits_are_immutable():
    budget = Budget()
    with pytest.raises(dataclasses.FrozenInstanceError):
        budget.max_steps = 999  # type: ignore[misc]


def test_fresh_state_is_not_exhausted_and_all_dimensions_are_zero():
    state = BudgetState()
    assert state.exhausted() is None
    for dimension in EXHAUSTION_ORDER:
        assert state.used_of(dimension) == 0
    assert state.elapsed_seconds() >= 0


def test_charge_accumulates_across_calls():
    state = BudgetState()
    state.charge(steps=2, llm_calls=3, tokens=1_000, cost_cny=0.01)
    state.charge(steps=5, llm_calls=1, tokens=500, cost_cny=0.02)
    assert state.used_of("steps") == 7
    assert state.used_of("llm_calls") == 4
    assert state.used_of("tokens") == 1_500
    assert state.used_of("cost_cny") == pytest.approx(0.03)
    assert state.exhausted() is None


AT_LIMIT = [
    ("steps", {"steps": 24}),
    ("llm_calls", {"llm_calls": 40}),
    ("tokens", {"tokens": 200_000}),
    ("cost_cny", {"cost_cny": 1.0}),
]


@pytest.mark.parametrize("dimension,charge_kwargs", AT_LIMIT)
def test_exactly_at_limit_is_not_exhaustion(dimension, charge_kwargs):
    state = BudgetState()
    state.charge(**charge_kwargs)
    assert state.used_of(dimension) == state.limit_of(dimension)
    assert state.exhausted() is None


OVER_LIMIT = [
    ("steps", {"steps": 25}),
    ("llm_calls", {"llm_calls": 41}),
    ("tokens", {"tokens": 200_001}),
    ("cost_cny", {"cost_cny": 1.01}),
]


@pytest.mark.parametrize("dimension,charge_kwargs", OVER_LIMIT)
def test_each_dimension_reports_its_own_name(dimension, charge_kwargs):
    state = BudgetState()
    state.charge(**charge_kwargs)
    assert state.exhausted() == dimension


def test_wall_seconds_uses_injected_clock():
    state = BudgetState(clock=fake_clock([100.0, 220.0]))
    assert state.elapsed_seconds() == pytest.approx(120.0)
    assert state.exhausted() is None  # 恰好 120 秒不算超

    state = BudgetState(clock=fake_clock([100.0, 220.1]))
    assert state.exhausted() == "wall_seconds"


def test_reporting_priority_is_fixed_and_documented():
    assert EXHAUSTION_ORDER == ("steps", "llm_calls", "tokens", "cost_cny", "wall_seconds")
    state = BudgetState()
    state.charge(steps=25, llm_calls=41, tokens=200_001, cost_cny=2.0)
    assert state.exhausted() == "steps"


def test_lower_priority_dimension_is_reported_when_higher_ones_are_clean():
    state = BudgetState()
    state.charge(llm_calls=41, tokens=200_001, cost_cny=2.0)
    assert state.exhausted() == "llm_calls"

    cost_only = BudgetState()
    cost_only.charge(cost_cny=1.01)
    assert cost_only.exhausted() == "cost_cny"


def test_exhaustion_reason_never_flips_back_to_none():
    state = BudgetState()
    state.charge(steps=25)
    assert state.exhausted() == "steps"
    for _ in range(3):
        state.charge(tokens=10)
        assert state.exhausted() == "steps"


@pytest.mark.parametrize(
    "charge_kwargs",
    [{"steps": -1}, {"llm_calls": -1}, {"tokens": -1}, {"cost_cny": -0.01}],
)
def test_negative_charge_is_rejected(charge_kwargs):
    with pytest.raises(ValueError, match="预算只能被消耗"):
        BudgetState().charge(**charge_kwargs)


def test_unknown_dimension_is_rejected():
    with pytest.raises(KeyError):
        BudgetState().used_of("latency_ms")
    with pytest.raises(KeyError):
        BudgetState().limit_of("latency_ms")


def test_snapshot_is_json_serialisable_and_restores_elapsed_time():
    original = BudgetState(clock=fake_clock([1_000.0, 1_030.0]))
    original.charge(steps=7, llm_calls=9, tokens=1_234, cost_cny=0.12)
    snapshot = json.loads(json.dumps(original.to_dict()))

    restored = BudgetState.from_dict(snapshot, clock=fake_clock([50_000.0, 50_020.0]))
    assert restored.used_of("steps") == 7
    assert restored.used_of("cost_cny") == pytest.approx(0.12)
    assert restored.limit_of("tokens") == 200_000
    # 30 秒历史墙钟 + 20 秒新增，而不是从 0 重新开始
    assert restored.elapsed_seconds() == pytest.approx(50.0)
    assert restored.exhausted() is None


def test_restored_state_still_reports_exhaustion():
    state = BudgetState()
    state.charge(cost_cny=1.25)
    restored = BudgetState.from_dict(state.to_dict())
    assert restored.exhausted() == "cost_cny"
