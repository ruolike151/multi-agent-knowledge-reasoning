"""Task 37 的测试：数字只能来自 JSONL，文档与数据必须能互相校验。

全部用例都在内存与 tmp_path 里现造记录，不读真实 `eval/runs/`、不调用任何模型——
渲染器本身不允许有模型依赖（校准结果从 JSON 文件进），否则 `--check` 会变成一次 API 调用。
"""

import subprocess
import sys
from pathlib import Path

import pytest

from eval.judges.judge import CalibrationReport, Divergence
from eval.runners import render_report
from eval.runners.fault_injection import FaultInjectionLog, FaultRunLog
from eval.runners.render_report import (
    BEGIN_MARKER,
    EXIT_DRIFT,
    EXIT_NO_DATA,
    NOT_MEASURABLE,
    ReportError,
    build_generated_block,
    check_document,
    cost_table,
    fault_table,
    main,
    main_table,
    refusal_table,
    render_calibration_table,
    render_result_bullets,
    splice_generated,
    variance_section,
)
from eval.runners.run_experiment import (
    VARIANCE_REPEATS,
    VARIANCE_SEED,
    VARIANCE_SUBSET_SIZE,
    RunRecord,
)


def make_record(**overrides: object) -> RunRecord:
    """造一条 `RunRecord`：默认是"目标系统答对、检索命中、成本已计量"的健康记录。"""
    fields: dict[str, object] = dict(
        system="ours",
        phase="main",
        repeat=1,
        task_id="single_hop-001",
        type="single_hop",
        answerable=True,
        answer="2024 年营业收入为 100 亿元。",
        citations=("d1#c1",),
        retrieved_ids=("d1#c1", "d1#c2"),
        gold_ids=("d1#c1",),
        supports=(True, True),
        answered=True,
        judge_score=2.0,
        latency_ms=1000.0,
        cost_cny=0.01,
        system_cost_cny=0.008,
        judge_cost_cny=0.002,
        cost_source="metered",
        steps=4,
        illegal_transitions=0,
        tool_call_success=1.0,
        degraded=False,
        state="DONE",
        started_at="2026-09-23T21:00:00+08:00",
    )
    fields.update(overrides)
    # run_id 沿用 Task 35 的拼法，保证测试里的记录与真实跑出来的记录同构
    run_id = (
        f"{fields['phase']}-{fields['system']}-{fields['task_id']}-r{fields['repeat']}"
    )
    return RunRecord(run_id=run_id, **fields)  # type: ignore[arg-type]


def records_for_main() -> list[RunRecord]:
    """三组各 2 条：目标系统全通过、Baseline A 一半、Baseline B 检索不可测。"""
    records = [
        make_record(system="ours", task_id="single_hop-001"),
        make_record(system="ours", task_id="multi_hop-001", type="multi_hop"),
        make_record(
            system="baseline_a",
            task_id="single_hop-001",
            judge_score=1.0,
            latency_ms=1500.0,
            cost_cny=0.02,
            system_cost_cny=0.015,
            judge_cost_cny=0.005,
        ),
        make_record(
            system="baseline_a",
            task_id="multi_hop-001",
            type="multi_hop",
            judge_score=0.0,
            latency_ms=1500.0,
            cost_cny=0.02,
            system_cost_cny=0.015,
            judge_cost_cny=0.005,
        ),
    ]
    for task_id, task_type in (
        ("single_hop-001", "single_hop"),
        ("multi_hop-001", "multi_hop"),
    ):
        records.append(
            make_record(
                system="baseline_b",
                task_id=task_id,
                type=task_type,
                retrieved_ids=(),          # 独立进程，网关插不进去（Task 35 设计 2/3）
                tool_call_success=None,    # None = 不可测，不是"全失败"
                cost_source="reported",
                cost_cny=0.03,
                judge_score=0.0,
                latency_ms=1500.0,
            )
        )
    return records


def row_of(table: str, system: str) -> list[str]:
    """从 markdown 表里取某个系统的整行，避免用 `in` 做脆弱断言。"""
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0].strip("`") == system:
            return cells
    raise AssertionError(f"表里没有系统 {system}：\n{table}")


def rows_of(table: str) -> list[list[str]]:
    """取表的所有数据行（丢掉表头与分隔行）。"""
    lines = [line for line in table.splitlines() if line.startswith("|")]
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines[2:]
    ]


def fault_fixture() -> tuple[list[RunRecord], list[FaultRunLog]]:
    """两个模式 × 三个故障率 × 2 题，记录与日志用 `run_id` 一对一（Task 36 设计 7）。"""
    records: list[RunRecord] = []
    logs: list[FaultRunLog] = []
    for mode in ("error", "timeout"):
        for rate in (0.0, 0.2, 0.5):
            for task_id in ("single_hop-001", "multi_hop-001"):
                record = make_record(
                    system="ours",
                    phase=f"fault-{mode}-{rate:.1f}",
                    task_id=task_id,
                    state="DEGRADED",
                    degraded=True,
                    judge_score=0.0,
                )
                records.append(record)
                logs.append(
                    FaultRunLog(
                        mode=mode,
                        rate=rate,
                        repeat=1,
                        task_id=task_id,
                        run_id=record.run_id,
                        log=FaultInjectionLog(
                            total_calls=3,
                            injected=1 if rate > 0 else 0,
                            injected_rate=(1 / 3) if rate > 0 else 0.0,
                        ),
                    )
                )
    return records, logs


def test_main_table_covers_all_systems_in_fixed_order():
    """行序固定为 `SYSTEMS`，列名带截断位：表格形状本身也是结论的一部分。"""
    table = main_table(records_for_main())
    header_cells = [cell.strip() for cell in table.splitlines()[0].strip("|").split("|")]
    assert header_cells[0] == "系统"
    assert "recall@8" in header_cells
    assert [row[0].strip("`") for row in rows_of(table)] == [
        "ours",
        "baseline_a",
        "baseline_b",
    ]
    assert row_of(table, "ours")[4] == "100.0%"        # judge 正确率 2/2
    assert row_of(table, "baseline_a")[4] == "50.0%"   # 1/2
    assert row_of(table, "baseline_b")[4] == "0.0%"


def test_main_table_marks_unmeasurable_retrieval_and_tool_success():
    """Baseline B 的检索与工具指标一律「不可测」，并在表下写明原因（设计 5）。"""
    table = main_table(records_for_main())
    baseline_b = row_of(table, "baseline_b")
    assert baseline_b[2] == NOT_MEASURABLE   # recall@8
    assert baseline_b[3] == NOT_MEASURABLE   # MRR
    assert baseline_b[7] == NOT_MEASURABLE   # 工具成功率
    assert "检索指标不可测" in table
    assert row_of(table, "ours")[2] != NOT_MEASURABLE


def test_cost_table_separates_system_and_judge_cost_with_source():
    """成本拆系统/judge 两列并标来源：回答"这张表花了多少钱、几成花在尺子上"。"""
    table = cost_table(records_for_main())
    ours = row_of(table, "ours")
    assert ours[2] == "0.0160"    # 2 × 0.008 系统成本
    assert ours[3] == "0.0040"    # 2 × 0.002 judge 成本
    assert ours[4] == "0.0200"    # 合计
    assert ours[5] == "20.0%"     # judge 占比
    assert ours[6] == "0.0100"    # 平均 ¥/题
    assert ours[7] == "metered"
    assert row_of(table, "baseline_b")[7] == "reported"


def test_refusal_table_reports_correct_and_over_refusal():
    """正确拒答率与过度拒答率必须同框（spec §9.2）。"""
    records = [
        make_record(
            task_id="refusal-001",
            type="refusal",
            answerable=False,
            answered=False,
            gold_ids=(),
            retrieved_ids=(),
            supports=(),
        ),
        make_record(
            task_id="refusal-002",
            type="refusal",
            answerable=False,
            answered=True,
            gold_ids=(),
            retrieved_ids=(),
            supports=(),
            judge_score=0.0,
        ),
        make_record(task_id="single_hop-003", answered=False, judge_score=0.0),
        make_record(task_id="single_hop-004", judge_score=1.0),
    ]
    row = row_of(refusal_table(records), "ours")
    assert row[1] == "2"       # 应拒答题数
    assert row[2] == "1"       # 正确拒答
    assert row[3] == "50.0%"
    assert row[4] == "2"       # 可答题数
    assert row[5] == "1"       # 过度拒答
    assert row[6] == "50.0%"


def test_fault_table_groups_by_mode_and_rate_ascending():
    """故障表由 `summarize_faults` 重算：0% 档的恢复率是"不存在"（n/a），不是 0。"""
    records, logs = fault_fixture()
    cells = rows_of(fault_table(records, logs))
    assert [(cell[0], cell[1]) for cell in cells] == [
        ("error", "0.0%"),
        ("timeout", "0.0%"),
        ("error", "20.0%"),
        ("timeout", "20.0%"),
        ("error", "50.0%"),
        ("timeout", "50.0%"),
    ]
    assert cells[0][5] == "n/a"                  # 0% 档没有注入，恢复率不可定义
    assert cells[2][5] == "100.0%（2/2）"         # 20% 档两次注入运行都拿到了分数
    assert cells[2][6] == "0.0%"                 # 最终成功率：没崩 ≠ 答对
    assert cells[2][7] == "100.0%"               # 降级率


def test_fault_table_says_not_run_without_data():
    assert "未跑" in fault_table([], [])


def test_calibration_table_prints_kappa_and_divergences():
    report = CalibrationReport(
        total=15,
        agreement_rate=0.8666,
        kappa=0.61,
        judge_pass_rate=0.8,
        human_pass_rate=0.7333,
        divergences=(
            Divergence(
                task_id="multi_hop-004",
                human_passed=False,
                judge_passed=True,
                score=1,
            ),
        ),
        cost_cny=0.02,
    )
    table = render_calibration_table(report)
    assert "0.610" in table        # κ 三位小数，与 Task 34 报告同精度
    assert "86.7%" in table
    assert "multi_hop-004" in table


def test_calibration_table_says_zero_divergence_when_none():
    report = CalibrationReport(
        total=15,
        agreement_rate=1.0,
        kappa=1.0,
        judge_pass_rate=0.8,
        human_pass_rate=0.8,
        divergences=(),
        cost_cny=0.02,
    )
    assert "零分歧" in render_calibration_table(report)


def test_variance_section_states_sampling_assumption():
    """方差段必须把"子集外推"这个假设写在脸上（spec §9.4）。"""
    variance = [
        make_record(
            system=system,
            phase="variance",
            repeat=repeat,
            judge_score=2.0 if repeat < 3 else 0.0,
        )
        for system in ("ours", "baseline_a")
        for repeat in (1, 2, 3)
    ]
    section = variance_section(records_for_main(), variance)
    assert f"{VARIANCE_SUBSET_SIZE} 题" in section
    assert f"{VARIANCE_REPEATS} 次" in section
    assert f"种子 {VARIANCE_SEED}" in section
    assert "外推" in section
    assert "极差" in section
    assert "100.0" in section   # 第 3 次整轮未通过 ⇒ 极差 100.0 pp


def test_variance_section_says_not_run_without_data():
    assert "未跑" in variance_section(records_for_main(), [])


def test_result_bullets_use_headline_deltas():
    """成果条目里的 `X% → Y%` 直接来自记录；缺数据时明确写"勿对外引用"。"""
    bullets = render_result_bullets(records_for_main())
    assert "50.0% → 100.0%" in bullets   # judge 正确率：baseline_a → ours
    assert "数据缺失" not in bullets
    assert "{{" not in bullets


def test_result_bullets_report_missing_data():
    only_ours = [record for record in records_for_main() if record.system == "ours"]
    bullets = render_result_bullets(only_ours)
    assert "数据缺失" in bullets


def test_result_bullets_flag_regressed_metrics():
    """退步的条目必须自己说「退步」：进步与退步用同一种写法，等于把退步当卖点。

    2026-09-25 主实验真跑后，第 7 节一度渲染成「judge 正确率 56.7% → 27.6%」这种句子——
    数字没错，格式却在暗示「改进」。方向由渲染器说，不能指望读的人自己比大小。
    """

    records = [
        make_record(
            system="baseline_a",
            task_id="single_hop-001",
            judge_score=2.0,
            latency_ms=1000.0,
            cost_cny=0.01,
            system_cost_cny=0.008,
            judge_cost_cny=0.002,
        ),
        make_record(
            system="ours",
            task_id="single_hop-001",
            judge_score=0.0,
            latency_ms=9000.0,
            cost_cny=0.05,
            system_cost_cny=0.04,
            judge_cost_cny=0.01,
        ),
    ]
    bullets = render_result_bullets(records)
    assert "100.0% → 0.0%（退步，勿对外当作亮点）" in bullets
    assert "1000.0 ms → 9000.0 ms（退步，勿对外当作亮点）" in bullets
    assert "¥0.0100 → ¥0.0500（退步，勿对外当作亮点）" in bullets
    assert "方向为负" in bullets


def test_result_bullets_mark_flat_metrics():
    """两边一样时必须写「持平」：把 0.0 pp 的差渲染成 `X → Y`，会被读成「变化」。"""

    records = [
        make_record(system="baseline_a", task_id="single_hop-001"),
        make_record(system="ours", task_id="single_hop-001"),
    ]
    bullets = render_result_bullets(records)
    assert "（持平）" in bullets
    assert "（退步" not in bullets
    assert "方向为负" not in bullets


def test_empty_records_raise_report_error():
    with pytest.raises(ReportError):
        main_table([])


def test_splice_generated_rejects_half_marker_document():
    """标记只出现一个时不许猜：猜错会覆盖掉手写内容且不可恢复。"""
    with pytest.raises(ReportError):
        splice_generated(f"# 说明\n{BEGIN_MARKER}\n", "生成内容")


def test_check_document_detects_hand_edited_numbers(tmp_path: Path):
    generated = build_generated_block(
        main_records=records_for_main(),
        variance_records=[],
        fault_records=[],
        fault_logs=[],
        calibration=None,
    )
    doc = tmp_path / "results.md"
    doc.write_text(
        splice_generated("# 实验结果\n\n结论解读写在这里。\n", generated),
        encoding="utf-8",
    )
    assert check_document(doc, generated) is None

    tampered = doc.read_text(encoding="utf-8").replace("100.0%", "88.8%", 1)
    doc.write_text(tampered, encoding="utf-8")
    problem = check_document(doc, generated)
    assert problem is not None
    assert "不一致" in problem


def test_cli_writes_document_then_detects_drift(tmp_path: Path):
    runs = tmp_path / "main.jsonl"
    runs.write_text(
        "".join(record.to_jsonl_line() + "\n" for record in records_for_main()),
        encoding="utf-8",
    )
    doc = tmp_path / "results.md"

    assert main(["--runs", str(runs), "--out", str(doc)]) == 0
    text = doc.read_text(encoding="utf-8")
    assert BEGIN_MARKER in text
    assert "### 1. 主表" in text

    assert main(["--runs", str(runs), "--out", str(doc), "--check"]) == 0

    doc.write_text(text.replace("100.0%", "88.8%", 1), encoding="utf-8")
    assert main(["--runs", str(runs), "--out", str(doc), "--check"]) == EXIT_DRIFT


def test_cli_exits_when_runs_missing(tmp_path: Path):
    doc = tmp_path / "results.md"
    code = main(["--runs", str(tmp_path / "absent.jsonl"), "--out", str(doc)])
    assert code == EXIT_NO_DATA
    assert not doc.exists()


def test_render_module_has_no_model_dependency():
    """护栏：渲染器不许依赖模型客户端，否则 `--check` 会变成一次 API 调用、烧掉预算。"""
    source = Path(render_report.__file__).read_text(encoding="utf-8")
    assert "akr.model" not in source
    assert "LLMClient" not in source


def test_cli_help_runs_without_pytest_pythonpath():
    """同 `run_experiment.py`：脚本直跑时 `eval/` 不在 sys.path 里，`--help` 必须能起来。"""

    root = Path(__file__).resolve().parents[2]
    script = root / "eval" / "runners" / "render_report.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(root),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "渲染成对外结果文档" in result.stdout
