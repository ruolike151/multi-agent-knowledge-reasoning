"""`scripts/half_sample_compare.py` 的单测。

与 `tests/test_retrieval_pool_depth.py` 同一套做法：importlib 按路径加载脚本，**不触网、
不读真实 runs**（脚本本身只读 `eval/runs/*.jsonl`，真实表由它自己跑出来）。这里只钉住三件
容易悄悄坏掉的事：口径的键与短表头对得上、空切片不假装有数据、指标格式化不被宽字符带偏。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from eval.runners.run_experiment import RunRecord

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "half_sample_compare.py"


def _load():
    spec = importlib.util.spec_from_file_location("half_sample_compare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load()


def _record(system: str, task_id: str, **overrides) -> RunRecord:
    """最小可用记录：只填 `build_subsets` 要用的字段，其余交给默认值。"""

    payload = {
        "run_id": f"{system}-{task_id}",
        "system": system,
        "phase": "main",
        "repeat": 1,
        "task_id": task_id,
        "type": "single_hop",
        "answerable": True,
        "answer": "a",
        "answered": True,
        "latency_ms": 1000.0,
        "cost_cny": 0.1,
        "steps": 1,
        "state": "DONE",
        "model": "deepseek-flash",
        "started_at": "2026-09-25T20:00:00+08:00",
    }
    payload.update(overrides)
    return RunRecord.model_validate(payload)


def test_subsets_split_by_system_and_restrict_to_the_half_sample() -> None:
    """`same 15` 两列必须真的被切成 15 条；否则就是拿 15 题去比 30 题。"""

    half = [_record("ours", f"t{i}") for i in range(15)]
    main = half + [_record("ours", f"u{i}") for i in range(15)]
    main += [_record("baseline_a", f"t{i}") for i in range(15)]
    main += [_record("baseline_a", f"u{i}") for i in range(15)]

    subsets = compare.build_subsets(half, main)

    assert len(subsets["ours half15 (rerun)"]) == 15
    assert len(subsets["ours 30 grid"]) == 30
    assert len(subsets["ours 30 grid (same 15)"]) == 15
    assert len(subsets["baseline_a 30 grid"]) == 30
    assert len(subsets["baseline_a (same 15)"]) == 15


def test_every_subset_has_a_short_header() -> None:
    """口径键与短表头是两处手写的字符串，漏一个就会在表里露出全名、把列挤串。"""

    half = [_record("ours", "t0")]
    main = half + [_record("baseline_a", "t0")]
    assert set(compare.build_subsets(half, main)) <= set(compare.SHORT_LABELS)


def test_empty_subset_is_reported_as_zero_runs_not_as_a_missing_row() -> None:
    assert compare.describe([]) == {"runs": 0}


def test_describe_counts_degraded_and_gold_in_pool_on_top_of_the_shared_metrics() -> None:
    rows = [
        _record("ours", "t0", gold_ids=("g",), retrieved_ids=("x", "g"), judge_score=2.0),
        _record("ours", "t1", gold_ids=("g",), retrieved_ids=("x",), judge_score=0.0, degraded=True),
        # 拒答题没有 gold：不进 gold-in-pool 的分母。
        _record(
            "ours", "t2", answerable=False, answered=False, gold_ids=(), retrieved_ids=("x",)
        ),
    ]
    summary = compare.describe(rows)

    assert summary["runs"] == 3
    assert summary["degraded_runs"] == 1
    assert summary["answerable_runs"] == 2
    assert summary["gold_in_pool_runs"] == 1
    assert summary["latency_max_s"] == pytest.approx(1.0)
    # 指标本身来自共享实现，不在这里重算——只确认它确实被带出来了。
    assert set(summary) > {"runs", "recall@8", "mrr", "judge_score_mean", "cost_cny_total"}


def test_metric_format_keeps_millisecond_latency_in_seconds() -> None:
    """延迟在记录里是毫秒、在表里是秒——这里正是最容易差 1000 倍的地方。"""

    assert compare.format_metric("latency_p95_ms", 47844.0) == "47.8"
    assert compare.format_metric("latency_p50_ms", 0.0) == "0.0"
    assert compare.format_metric("recall@8", 0.2692307692307692) == "0.2692"
    assert compare.format_metric("steps_mean", 8) == "8"
    assert compare.format_metric("judge_score_mean", None) == "n/a"


def test_padding_uses_terminal_columns_for_cjk_labels() -> None:
    assert compare.display_width("幻觉率") == 6
    assert compare.display_width("recall@8") == 8
    assert compare.pad_right("幻觉率", 24) == "幻觉率" + " " * 18
    assert compare.pad_left("幻觉率", 18) == " " * 12 + "幻觉率"
    # 已经超宽时不再截断（宁可挤列，也不丢字）。
    assert compare.pad_left("baseline_a same15", 10) == "baseline_a same15"
