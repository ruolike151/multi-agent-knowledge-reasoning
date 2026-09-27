"""运行次数闸门的单测（2026-09-25 用户决策：整场实验总共 60 次运行）。

闸门的价值全在三个地方，用例也围着它们转：

1. **落盘**：进程退了、机器重启了，次数还在 → 必须能读回来；
2. **先记账后调用**：`claim()` 返回时盘上就已经 +1，中途中断只会少跑不会超跑；
3. **文件坏了要报错**：绝不静默从 0 重新计——静默重置等于把闸门拆掉。
"""

from __future__ import annotations

import json

import pytest

from eval.runners.run_ledger import (
    DEFAULT_MAX_RUNS,
    RunCountExceeded,
    RunLedger,
    RunLedgerError,
)


def _claim(ledger: RunLedger, *, phase: str = "main", system: str = "ours", task_id: str = "t1"):
    return ledger.claim(phase=phase, system=system, task_id=task_id)


def test_default_cap_is_sixty_and_nothing_is_written_before_the_first_run(tmp_path):
    path = tmp_path / "run-ledger.json"

    ledger = RunLedger(path)

    assert ledger.max_runs == DEFAULT_MAX_RUNS == 60
    assert (ledger.used, ledger.remaining) == (0, 60)
    assert ledger.cap_changed is False
    assert not path.exists()          # 没跑过就不落盘，不制造一个"看起来像账"的空文件


def test_claim_records_the_count_before_the_caller_spends_anything(tmp_path):
    path = tmp_path / "run-ledger.json"
    ledger = RunLedger(path, max_runs=3)

    used = _claim(ledger)

    assert used == 1
    # 先记账、后调用：调用方拿到返回值时，盘上已经是 1 了
    assert json.loads(path.read_text(encoding="utf-8"))["used"] == 1


def test_the_cap_is_a_hard_stop_and_the_blocked_claim_does_not_count(tmp_path):
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=2)
    _claim(ledger, system="baseline_a")
    _claim(ledger, task_id="t2")

    with pytest.raises(RunCountExceeded, match="2/2"):
        _claim(ledger, task_id="t3")

    assert ledger.used == 2           # 被拦下的那一次没有落账
    assert ledger.remaining == 0


def test_the_count_survives_a_new_process(tmp_path):
    path = tmp_path / "run-ledger.json"
    first = RunLedger(path, max_runs=10)
    for index in range(4):
        _claim(first, task_id=f"t{index}")

    second = RunLedger(path)          # 不带 --max-runs：沿用账本里记的上限

    assert (second.max_runs, second.used, second.remaining) == (10, 4, 6)
    assert second.cap_changed is False


def test_an_explicit_cap_overrides_the_recorded_one_and_says_so(tmp_path):
    path = tmp_path / "run-ledger.json"
    RunLedger(path, max_runs=60).claim(phase="main", system="ours", task_id="t1")

    raised = RunLedger(path, max_runs=120)
    assert (raised.max_runs, raised.used, raised.cap_changed) == (120, 1, True)

    tightened = RunLedger(path, max_runs=1)
    # 上限收到低于已用次数是允许的：意思是"这场实验到此为止"，不是报错
    assert (tightened.max_runs, tightened.used, tightened.remaining) == (1, 1, 0)
    assert tightened.cap_changed is True


def test_a_broken_ledger_is_refused_instead_of_restarted_from_zero(tmp_path):
    path = tmp_path / "run-ledger.json"

    path.write_text('{"max_runs": 60, "used": ', encoding="utf-8")
    with pytest.raises(RunLedgerError, match="无法解析"):
        RunLedger(path)

    path.write_text(json.dumps({"max_runs": 60, "used": -1}), encoding="utf-8")
    with pytest.raises(RunLedgerError, match="used"):
        RunLedger(path)

    path.write_text(json.dumps({"max_runs": 0, "used": 0}), encoding="utf-8")
    with pytest.raises(RunLedgerError, match="max_runs"):
        RunLedger(path)

    path.write_text(json.dumps({"used": 3}), encoding="utf-8")
    with pytest.raises(RunLedgerError, match="max_runs"):
        RunLedger(path)


def test_invalid_caps_and_counts_are_rejected(tmp_path):
    with pytest.raises(RunLedgerError, match="上限"):
        RunLedger(tmp_path / "l.json", max_runs=0)

    ledger = RunLedger(tmp_path / "l.json", max_runs=5)
    with pytest.raises(RunLedgerError, match="count"):
        ledger.claim(phase="main", system="ours", task_id="t1", count=0)


def test_history_keeps_the_label_of_every_claim(tmp_path):
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=5)

    _claim(ledger, phase="main", system="ours", task_id="single_hop-001")
    _claim(ledger, phase="variance", system="baseline_a", task_id="multi_hop-003")

    assert [
        (entry["phase"], entry["system"], entry["task_id"], entry["used"])
        for entry in ledger.state.history
    ] == [
        ("main", "ours", "single_hop-001", 1),
        ("variance", "baseline_a", "multi_hop-003", 2),
    ]


def test_two_runners_share_one_book(tmp_path):
    """两个 runner 默认共用一本账：60 次是整场实验的总数，不是每个脚本各 60 次。"""
    path = tmp_path / "run-ledger.json"
    experiment = RunLedger(path, max_runs=3)
    fault = RunLedger(path)

    _claim(experiment, task_id="t1")
    _claim(fault, phase="fault-error-0.5", task_id="t1")

    assert fault.used == 2            # 读到的是对方刚写下的数
    assert experiment.claim(phase="main", system="ours", task_id="t2") == 3
    with pytest.raises(RunCountExceeded):
        _claim(experiment, task_id="t4")


def test_claims_re_read_the_file_so_a_stale_handle_cannot_double_spend(tmp_path):
    path = tmp_path / "run-ledger.json"
    first = RunLedger(path, max_runs=2)
    second = RunLedger(path)

    _claim(first)
    assert second.used == 1           # 属性读盘：手里的旧对象不会给出过期数字

    _claim(second, task_id="t2")
    with pytest.raises(RunCountExceeded, match="2/2"):
        _claim(first, task_id="t3")


def test_writing_the_ledger_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "run-ledger.json"
    ledger = RunLedger(path, max_runs=2)
    _claim(ledger)
    _claim(ledger, task_id="t2")

    assert sorted(item.name for item in tmp_path.iterdir()) == ["run-ledger.json"]


def test_state_is_a_readable_snapshot(tmp_path):
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=60)
    _claim(ledger)

    state = ledger.state

    assert state.path == tmp_path / "run-ledger.json"
    assert (state.max_runs, state.used, state.remaining) == (60, 1, 59)
    assert state.to_json_dict()["used"] == 1
