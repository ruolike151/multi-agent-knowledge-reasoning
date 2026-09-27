"""实验结果渲染器：JSONL → 对外结果文档（spec §15）。

为什么要有这一层：对外写的每个数字都必须能追到一条 `eval/runs/*.jsonl` 记录。
手工填表在改动一次实验之后必然过期，而"过期的数字"最经不起追问——
别人只要问一句"这个 78% 是怎么算出来的"就露了。所以：

1. 表格由脚本从记录生成，函数签名里根本没有"手填指标"这个入口；
2. 提供 `--check`：文档里的数字与 JSONL 不一致就**非零退出**，把"文档与数据一致"
   变成一条可执行的护栏，而不是靠人的自觉。

渲染器不做任何模型调用：judge 校准结果由 Task 34 的 `main` 落成 JSON 文件后传进来
（`--calibration`），因此 `--check` 永远不花 API 预算。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# 同 `run_experiment.py`：`eval/` 不在可编辑安装的 `src/` 里，直接执行脚本时
# sys.path[0] 是 `eval/runners/`，`import eval.*` 会 ModuleNotFoundError。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judges.judge import JUDGE_PASS_THRESHOLD, CalibrationReport, Divergence
from eval.metrics import refusal_metrics
from eval.runners.fault_injection import (
    DEFAULT_FAULT_LOGS,
    DEFAULT_FAULT_OUT,
    FaultRunLog,
    load_fault_logs,
    summarize_faults,
)
from eval.runners.run_experiment import (
    MAIN_REPEATS,
    RUNS_DIR,
    VARIANCE_REPEATS,
    VARIANCE_SEED,
    VARIANCE_SUBSET_SIZE,
    RunRecord,
    load_records,
    summarize_records,
)

DEFAULT_MAIN = RUNS_DIR / "main.jsonl"
DEFAULT_VARIANCE = RUNS_DIR / "variance.jsonl"
DEFAULT_CALIBRATION = Path("eval/judges/calibration.json")
DEFAULT_OUT = Path("docs/experiments/2026-09-23-results.md")

# 生成段边界：渲染器只接管这两个标记之间的内容，标记之外是手写区（结论、话术、局限）
BEGIN_MARKER = "<!-- BEGIN GENERATED: 由 render_report 生成，勿手改 -->"
END_MARKER = "<!-- END GENERATED -->"

NOT_MEASURABLE = "不可测"   # 拿不到的数就写"不可测"，绝不填 0（Task 35 设计 3）
NA = "n/a"                  # 条件不成立导致"这个比率不存在"（例如 0% 档的恢复率）
OUR_SYSTEM = "ours"
BASELINE_A = "baseline_a"
BASELINE_B = "baseline_b"

# 报告的行序在这里单独定义，不借用 Task 35 的 `SYSTEMS`：那个常量的用途是
# 「跑哪些系统」（顺序是 `baseline_a` 在前，见偏差日志第二十节的三组→两组收窄），
# 而表格行序是「先让读者看到谁」——目标系统必须是第一行。数据里没有的系统不出行
# （Baseline B 已放弃，见第二十节），表下会写明本表只剩两组。
ROW_ORDER: tuple[str, ...] = (OUR_SYSTEM, BASELINE_A, BASELINE_B)

EXIT_OK = 0
EXIT_NO_DATA = 1   # 还没跑实验：不是数据错，是没有数据
EXIT_DRIFT = 2     # 文档与 JSONL 对不上：这才需要人立刻处理

DOC_HEADER = """# 实验结果（2026-09-23）

> 数据源：`eval/runs/main.jsonl`（主表）、`eval/runs/variance.jsonl`（重复实验）、
> `eval/runs/fault.jsonl` + `eval/runs/fault-logs.jsonl`（故障注入）、
> `eval/judges/calibration.json`（judge 校准）。
>
> `BEGIN GENERATED` / `END GENERATED` 之间的内容由
> `python -m eval.runners.render_report` 生成，**不要手改**；
> 标记之外是手写区（结论解读、已知局限），重渲染不会覆盖。
> 校验：`python -m eval.runners.render_report --check`，不一致即退出码 2。
"""


class ReportError(RuntimeError):
    """报告无法生成：没有数据、标记错位，或者有人想手工填数字。"""


def format_pct(value: float | None) -> str:
    """百分比统一一位小数：报告里出现两种精度，就会被当成两个口径。"""
    return NA if value is None else f"{value * 100:.1f}%"


def data_stamp(records: Sequence[RunRecord]) -> str:
    """取数据自身的最大 `started_at`，而不是 `datetime.now()`。

    为什么较真：`--check` 的前提是"同输入必出同输出"。渲染时刻一变，生成段就变，
    每次校验都会报漂移，护栏自己先失效——人三天之内就会开始"失败了也照常提交"。
    """
    stamps = [record.started_at for record in records if record.started_at]
    return max(stamps) if stamps else "未标注"


def group_by_system(
    records: Sequence[RunRecord],
    *,
    phase: str | None = "main",
) -> dict[str, list[RunRecord]]:
    """按系统分组，行序固定为 `ROW_ORDER`：表格行顺序不许随文件里的行序漂移。"""
    selected = [record for record in records if phase is None or record.phase == phase]
    if not selected:
        raise ReportError(
            f"没有 phase={phase!r} 的记录：报告只允许由真实运行数据生成（spec §15）。"
            "先跑 `python -m eval.runners.run_experiment`。"
        )
    grouped: dict[str, list[RunRecord]] = {}
    for record in selected:
        grouped.setdefault(record.system, []).append(record)
    ordered = {system: grouped[system] for system in ROW_ORDER if system in grouped}
    for system in sorted(set(grouped) - set(ordered)):
        ordered[system] = grouped[system]
    return ordered


def judge_pass_rate(records: Sequence[RunRecord]) -> float | None:
    """judge 判"通过"的比例；阈值复用 `JUDGE_PASS_THRESHOLD`（设计 6）。

    为什么不在报告层再定一个阈值（比如"≥1.5 才算对"）：Task 34 的校准报告算的是
    同一个阈值下的 κ，两个"通过"一旦不是一回事，一致率与 κ 就都成了自证。
    """
    scored = [record for record in records if record.judge_score is not None]
    if not scored:
        return None
    passed = sum(
        1 for record in scored if record.judge_score >= JUDGE_PASS_THRESHOLD
    )
    return passed / len(scored)


def tool_success_rate(records: Sequence[RunRecord]) -> float | None:
    """工具调用成功率；全为 `None` 就是"不可测"，**不等于 0**。"""
    values = [
        record.tool_call_success
        for record in records
        if record.tool_call_success is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def retrieval_measurable(records: Sequence[RunRecord]) -> bool:
    """只要有一次检索结果非空，这组的检索指标就是可测的（独立进程基线恒为空）。"""
    return any(record.retrieved_ids for record in records)


MAIN_TABLE_HEADER = (
    "| 系统 | 题数 | {recall} | MRR | judge 正确率 | 引用支撑率 | 幻觉率 | "
    "工具成功率 | P50 ms | P95 ms | 步数均值 | 非法转移 | 成本均值 ¥ |"
)
MAIN_TABLE_SEP = "|" + "---|" * 13


def main_table(records: Sequence[RunRecord], *, phase: str | None = "main") -> str:
    """系统对照主表（spec §9.3）。所有指标来自 Task 33 的 `aggregate`，这里只做排版。"""
    grouped = group_by_system(records, phase=phase)
    summaries = {
        system: summarize_records(rows)[system] for system, rows in grouped.items()
    }
    # 截断位从 aggregate 的键名里取（Task 33 设计 7）：报告层不许硬编码 @8
    recall_key = next(
        key
        for summary in summaries.values()
        for key in summary
        if key.startswith("recall@")
    )
    lines = [MAIN_TABLE_HEADER.format(recall=recall_key), MAIN_TABLE_SEP]
    unmeasurable: list[str] = []
    for system, rows in grouped.items():
        summary = summaries[system]
        measurable = retrieval_measurable(rows)
        if not measurable:
            unmeasurable.append(system)
        tools = tool_success_rate(rows)
        recall = format_pct(summary[recall_key]) if measurable else NOT_MEASURABLE
        mrr = f"{summary['mrr']:.3f}" if measurable else NOT_MEASURABLE
        tool_rate = format_pct(tools) if tools is not None else NOT_MEASURABLE
        lines.append(
            f"| `{system}` | {summary['runs']} | {recall} | {mrr} | "
            f"{format_pct(judge_pass_rate(rows))} | "
            f"{format_pct(summary['citation_support_rate'])} | "
            f"{format_pct(summary['hallucination_rate'])} | {tool_rate} | "
            f"{summary['latency_p50_ms']:.1f} | {summary['latency_p95_ms']:.1f} | "
            f"{summary['steps_mean']:.2f} | "
            f"{int(summary['illegal_transitions_total'])} | "
            f"{summary['cost_cny_mean']:.4f} |"
        )
    if unmeasurable:
        lines.append("")
        lines.append(
            "> 注："
            + "、".join(f"`{system}`" for system in unmeasurable)
            + " 是独立进程（spec §9.3 要求原样运行），检索指标不可测"
            "（`retrieved_ids` 全空、`tool_call_success is None`），表中以「不可测」标注，"
            "**不伪造数值**。"
        )
    if BASELINE_B not in grouped:
        lines.append("")
        lines.append(
            "> 本表为两组对照：`baseline_b`（参考项目固定 DAG，原样运行）已于 "
            "2026-09-24 放弃（理由见内部偏差日志第二十节），"
            "因此本表只能回答「单 Agent 加工具不够用吗」，"
            "不能回答「我们比参考项目强多少」。"
        )
    return "\n".join(lines)


COST_TABLE_HEADER = (
    "| 系统 | 运行数 | 系统成本 ¥ | judge 成本 ¥ | 合计 ¥ | judge 占比 | "
    "平均 ¥/题 | 成本来源 |"
)
COST_TABLE_SEP = "|" + "---|" * 8


def cost_table(records: Sequence[RunRecord], *, phase: str | None = "main") -> str:
    """成本表：拆系统成本与 judge 成本，并标注每个数的来源（Task 35 设计 5）。"""
    grouped = group_by_system(records, phase=phase)
    lines = [COST_TABLE_HEADER, COST_TABLE_SEP]
    for system, rows in grouped.items():
        system_cost = sum(record.system_cost_cny for record in rows)
        judge_cost = sum(record.judge_cost_cny for record in rows)
        total = sum(record.cost_cny for record in rows)
        share = (judge_cost / total) if total > 0 else None
        sources = "/".join(sorted({record.cost_source for record in rows}))
        lines.append(
            f"| `{system}` | {len(rows)} | {system_cost:.4f} | {judge_cost:.4f} | "
            f"{total:.4f} | {format_pct(share)} | {total / len(rows):.4f} | {sources} |"
        )
    lines.append("")
    lines.append(
        "> 成本来源口径（Task 35 设计 5）：`metered` = 我们自己的 `Usage` 计量；"
        "`reported` = 外部进程自报；`unmetered` = 拿不到。三类混在同一列，来源必须一眼可见。"
    )
    lines.append(
        "> 系统成本与 judge 成本拆开报，是为了回答「这张对照表一共花了多少钱、"
        "其中几成花在尺子（judge）上」——只有一个合计数就无法解释"
        "「为什么我们只跑了 30 题 × 3 组」。"
    )
    return "\n".join(lines)


REFUSAL_TABLE_HEADER = (
    "| 系统 | 应拒答题数 | 正确拒答 | 正确拒答率 | 可答题数 | 过度拒答 | 过度拒答率 |"
)
REFUSAL_TABLE_SEP = "|" + "---|" * 7


def refusal_table(records: Sequence[RunRecord], *, phase: str | None = "main") -> str:
    """拒答指标：两个率必须同框，否则任一个都能被"策略性刷分"（spec §9.2）。"""
    grouped = group_by_system(records, phase=phase)
    lines = [REFUSAL_TABLE_HEADER, REFUSAL_TABLE_SEP]
    for system, rows in grouped.items():
        metrics = refusal_metrics(
            [not record.answerable for record in rows],
            [record.answered for record in rows],
        )
        lines.append(
            f"| `{system}` | {metrics.total_expected_refusal} | "
            f"{metrics.correct_refusal} | "
            f"{format_pct(metrics.correct_refusal_rate)} | "
            f"{metrics.total_answerable} | {metrics.over_refusal} | "
            f"{format_pct(metrics.over_refusal_rate)} |"
        )
    lines.append("")
    lines.append(
        "> 只报正确拒答率，系统可以靠「多拒答」刷高；只报过度拒答率，"
        "又可以靠「从不拒答」刷低。拒答判定对所有系统统一用 `is_refusal_text`"
        "（Task 35 设计 4：目标系统有 REFUSE 状态也不许拿它当判据，否则各组不可比）。"
    )
    return "\n".join(lines)


FAULT_TABLE_HEADER = (
    "| 故障模式 | 故障率 | 运行数 | 注入调用 | 注入运行 | 恢复率 | 最终成功率 | 降级率 |"
)
FAULT_TABLE_SEP = "|" + "---|" * 8


def fault_table(records: Sequence[RunRecord], logs: Sequence[FaultRunLog]) -> str:
    """故障注入表（spec §10）。数字由 Task 36 的 `summarize_faults` 重算，不手抄。"""
    if not records:
        return (
            "_本期未跑故障注入实验（缺 `eval/runs/fault.jsonl`）："
            "先跑 `python -m eval.runners.fault_injection` 再重新渲染。_"
        )
    lines = [FAULT_TABLE_HEADER, FAULT_TABLE_SEP]
    for report in summarize_faults(records, logs):
        if report.recovery_rate is None:
            recovery = NA
        else:
            recovery = (
                f"{format_pct(report.recovery_rate)}"
                f"（{report.recovered}/{report.injected_runs}）"
            )
        lines.append(
            f"| {report.mode} | {format_pct(report.fault_rate)} | {report.runs} | "
            f"{report.injected_calls} | {report.injected_runs} | {recovery} | "
            f"{format_pct(report.final_success_rate)} | "
            f"{format_pct(report.degraded_rate)} |"
        )
    lines.append("")
    lines.append(
        "> 恢复率 = 注入了故障且仍拿到 judge 分数的运行数 / 注入了故障的运行数；"
        "最终成功率按 `JUDGE_PASS_THRESHOLD` 判定。两个数必须分开读："
        "降级路径保证「不崩」，保证不了「答对」（Task 36 设计 6）。"
    )
    return "\n".join(lines)


VARIANCE_TABLE_SEP = "|" + "---|" * 6


def variance_section(
    main_records: Sequence[RunRecord],
    variance_records: Sequence[RunRecord],
    *,
    main_phase: str = "main",
    phase: str = "variance",
) -> str:
    """重复实验与方差：把"子集外推"这个假设写在脸上，而不是让人自己去猜（spec §9.4）。"""
    main_count = len([r for r in main_records if r.phase == main_phase])
    lines = [
        f"主表：核心集 {main_count} 条记录 × {MAIN_REPEATS} 次；"
        f"方差：按题型分层抽 {VARIANCE_SUBSET_SIZE} 题 × {VARIANCE_REPEATS} 次"
        f"（种子 {VARIANCE_SEED}，`select_variance_subset`）。",
        "",
        f"**已知假设**：以子集 {VARIANCE_REPEATS} 次重复的波动**外推**主表的稳定性；"
        "受 ¥5 预算约束，没有对全部 30 题重复采样（spec §9.4 / §7.7）。"
        "这句话必须写在文档里，否则「极差 0.0 pp」会被读成「系统完全确定」。",
        "",
    ]
    selected = [record for record in variance_records if record.phase == phase]
    if not selected:
        lines.append(
            f"_本期未跑重复实验（缺 `{DEFAULT_VARIANCE}`）："
            "先跑 `python -m eval.runners.run_experiment --phase variance`。_"
        )
        return "\n".join(lines)

    grouped = group_by_system(variance_records, phase=phase)
    repeats = sorted({record.repeat for record in selected})
    lines.append(
        "| 系统 | "
        + " | ".join(f"第 {index} 次 judge 正确率" for index in repeats)
        + " | 极差 (pp) |"
    )
    lines.append(VARIANCE_TABLE_SEP)
    for system, rows in grouped.items():
        rates = [
            judge_pass_rate([record for record in rows if record.repeat == index])
            for index in repeats
        ]
        known = [rate for rate in rates if rate is not None]
        spread = (max(known) - min(known)) * 100 if len(known) >= 2 else 0.0
        cells = " | ".join(format_pct(rate) for rate in rates)
        lines.append(f"| `{system}` | {cells} | {spread:.1f} |")
    lines.append("")
    lines.append(
        "> 极差按 judge 正确率（百分点）算，是这张表里最容易被忽略的一列："
        "它回答的是「换个随机种子、换个题序，结论会不会翻」——"
        "而单次运行的漂亮数字回答不了这个问题。"
    )
    return "\n".join(lines)


CALIBRATION_TABLE_HEADER = (
    "| 校准样本 | 一致率 | Cohen's κ | judge 通过率 | 人工通过率 | 校准成本 ¥ |"
)
CALIBRATION_TABLE_SEP = "|" + "---|" * 6
DIVERGENCE_TABLE_SEP = "|---|---|---|---|"


def render_calibration_table(report: CalibrationReport) -> str:
    """judge 校准表（spec §9.5）：没有这一节，主表里"judge 正确率"那一列不可引用。"""
    lines = [
        CALIBRATION_TABLE_HEADER,
        CALIBRATION_TABLE_SEP,
        f"| {report.total} | {format_pct(report.agreement_rate)} | "
        f"{report.kappa:.3f} | {format_pct(report.judge_pass_rate)} | "
        f"{format_pct(report.human_pass_rate)} | {report.cost_cny:.4f} |",
        "",
    ]
    if not report.divergences:
        lines.append(
            "本期零分歧：judge 与人工标签完全一致"
            "（人工判据 = gold 答案 + gold 引用，不看系统输出，spec §9.5）。"
        )
        return "\n".join(lines)
    lines.extend(
        [
            f"**分歧样本 {len(report.divergences)} 条**"
            "（人工判据 = gold 答案 + gold 引用，标注在跑实验之前完成）",
            "",
            "| task_id | 人工通过 | judge 通过 | judge 分数 |",
            DIVERGENCE_TABLE_SEP,
        ]
    )
    for item in report.divergences:
        lines.append(
            f"| `{item.task_id}` | "
            f"{'是' if item.human_passed else '否'} | "
            f"{'是' if item.judge_passed else '否'} | {item.score} |"
        )
    return "\n".join(lines)


FLAT_RATIO = 0.005


def metric_direction(
    before: float | None,
    after: float | None,
    *,
    higher_is_better: bool,
) -> str:
    """把一对数字判成「改善 / 退步 / 持平 / 缺对照」。

    为什么要有这个函数：`56.7% → 27.6%` 与 `11.7% → 19.5%` 在排版上长得一模一样，读者得自己比大小
    才知道哪个是进步。2026-09-25 主实验跑完后，第 7 节整段都是这种句子，照抄进对外材料就是把退步当卖点。
    方向由渲染器说，不指望读的人替我们比。

    差幅在 0.5% 以内算**持平**：四舍五入的抖动不该被读成「改善」。
    """

    if before is None or after is None:
        return "缺对照"
    scale = max(abs(before), abs(after))
    if abs(after - before) <= FLAT_RATIO * scale:
        return "持平"
    improved = after > before if higher_is_better else after < before
    return "改善" if improved else "退步"


def render_result_bullets(
    main_records: Sequence[RunRecord],
    *,
    fault_records: Sequence[RunRecord] = (),
    fault_logs: Sequence[FaultRunLog] = (),
    phase: str | None = "main",
) -> str:
    """成果条目（spec §15.1）：每个 `X% → Y%` 都由记录算出，缺数据就明说缺数据（设计 7）。"""
    grouped = group_by_system(main_records, phase=phase)
    ours = grouped.get(OUR_SYSTEM)
    baseline_a = grouped.get(BASELINE_A)
    if ours is None:
        raise ReportError(
            f"主表里没有 `{OUR_SYSTEM}` 的记录：成果条目必须来自目标系统的真实数据"
        )
    summaries = {
        system: summarize_records(rows)[system] for system, rows in grouped.items()
    }
    a_summary = summaries.get(BASELINE_A)
    o_summary = summaries[OUR_SYSTEM]

    def before(key: str) -> float | None:
        return None if a_summary is None else float(a_summary[key])

    def after(key: str) -> float:
        return float(o_summary[key])

    regressions: list[str] = []

    def delta(
        label: str,
        before_value: float | None,
        after_value: float | None,
        render,
        *,
        higher_is_better: bool,
    ) -> str:
        """`X → Y` 后面必须跟着方向：退步的那几条会在结尾统一点名。"""

        if before_value is None or after_value is None:
            return "数据缺失（缺对照记录，勿对外引用）"
        verdict = metric_direction(
            before_value, after_value, higher_is_better=higher_is_better
        )
        span = f"{render(before_value)} → {render(after_value)}"
        if verdict == "退步":
            regressions.append(label)
            return f"{span}（退步，勿对外当作亮点）"
        return f"{span}（{verdict}）"

    def as_ms(value: float) -> str:
        return f"{value:.1f} ms"

    def as_cny(value: float) -> str:
        return f"¥{value:.4f}"

    judge_delta = delta(
        "judge 正确率",
        judge_pass_rate(baseline_a) if baseline_a else None,
        judge_pass_rate(ours),
        format_pct,
        higher_is_better=True,
    )
    hallucination_delta = delta(
        "幻觉率",
        before("hallucination_rate"),
        after("hallucination_rate"),
        format_pct,
        higher_is_better=False,
    )
    support_delta = delta(
        "引用支撑率",
        before("citation_support_rate"),
        after("citation_support_rate"),
        format_pct,
        higher_is_better=True,
    )
    p95_delta = delta(
        "P95 延迟",
        before("latency_p95_ms"),
        after("latency_p95_ms"),
        as_ms,
        higher_is_better=False,
    )
    cost_delta = delta(
        "平均单题成本",
        before("cost_cny_mean"),
        after("cost_cny_mean"),
        as_cny,
        higher_is_better=False,
    )

    lines = [
        "- 复现开源基线并量化其缺陷（固定 DAG 伪多 Agent、增量更新实为文档级全量重建），"
        "重构为 Supervisor 动态路由 + 并行 Worker + 独立 Verifier 架构："
        f"judge 正确率 {judge_delta}",
        "- 自建 30 题核心评测集（单跳 8 / 多跳 10 / 多文档综合 7 / 应拒答 5，"
        "格式按 80 题设计），实现引用支撑率与幻觉率自动评估："
        f"引用支撑率 {support_delta}，幻觉率 {hallucination_delta}",
        "- 设计 context 隔离机制（worker 仅回传结构化摘要 + 任务账本）："
        f"P95 延迟 {p95_delta}，平均单题成本 {cost_delta}",
        "- 工具层以 MCP 协议实现，支持 Agent 动态发现与调用；"
        "通过 prompt 前缀稳定化利用前缀缓存，并把批量实验排在低峰时段"
        "（spec §7.5 / §7.7）",
    ]
    if fault_records:
        reports = summarize_faults(fault_records, fault_logs)
        worst = max(reports, key=lambda report: report.fault_rate, default=None)
        if worst is not None and worst.recovery_rate is not None:
            lines.append(
                f"- 工具层故障注入实验：故障率 {format_pct(worst.fault_rate)} 时"
                f"恢复率 {format_pct(worst.recovery_rate)}、"
                f"最终成功率 {format_pct(worst.final_success_rate)}"
                "——恢复率与最终成功率分开报，不用「加了降级」冒充「更准」"
            )
    unmeasurable = [
        system for system, rows in grouped.items() if not retrieval_measurable(rows)
    ]
    if regressions:
        lines.append("")
        lines.append(
            f"> 本节有 {len(regressions)} 条方向为负（{'、'.join(regressions)}）："
            "同一段话里进步与退步写法一样时，读者只能自己比大小——**退步不许对外当作亮点**"
            "（这里的 `X → Y` 只是记录，不是成绩）。"
        )
    if unmeasurable:
        lines.append(
            "> "
            + "、".join(f"`{system}`" for system in unmeasurable)
            + " 为独立进程（spec §9.3 要求原样运行），检索指标不可测，"
            "上面第 1、2 条不引用它的 recall / MRR。"
        )
    lines.append("")
    lines.append(
        "> 每一条里的数字都由本文件上文各表生成、来源是 `eval/runs/*.jsonl`；"
        "某条写着「缺对照记录，勿对外引用」时，说明对应实验还没跑，**不许手工补数字**"
        "（spec §15.1 里的 `X% → Y%` 只能来自记录）。"
    )
    return "\n".join(lines)


def build_generated_block(
    *,
    main_records: Sequence[RunRecord],
    variance_records: Sequence[RunRecord] = (),
    fault_records: Sequence[RunRecord] = (),
    fault_logs: Sequence[FaultRunLog] = (),
    calibration: CalibrationReport | None = None,
    main_phase: str = "main",
) -> str:
    """拼出两个标记之间的整段内容（手写区不在这里，渲染器不碰）。"""
    if not main_records:
        raise ReportError(
            "没有主实验记录，不生成任何表格：空表比没有表更危险——"
            "它看起来像结论，其实只是还没跑（spec §15）。"
        )
    parts = [
        BEGIN_MARKER,
        "",
        f"_本段由 `python -m eval.runners.render_report` 生成；数字全部来自 "
        f"`eval/runs/` 下的 JSONL 记录（数据时间：{data_stamp(main_records)}）。_",
        "",
        "### 1. 主表（系统对照，spec §9.3）",
        "",
        main_table(main_records, phase=main_phase),
        "",
        "### 2. 成本表（spec §7.7 / §9.3）",
        "",
        cost_table(main_records, phase=main_phase),
        "",
        "### 3. 拒答指标（spec §9.2）",
        "",
        refusal_table(main_records, phase=main_phase),
        "",
        "### 4. judge 校准（spec §9.5）",
        "",
        (
            render_calibration_table(calibration)
            if calibration is not None
            else "_本期未提供校准结果（缺 `eval/judges/calibration.json`）："
            "跑 Task 34 的 `calibrate` 并落盘后再渲染。"
            "**拿到校准数据之前，主表里的「judge 正确率」不可对外引用。**_"
        ),
        "",
        "### 5. 重复实验与方差（spec §9.4）",
        "",
        variance_section(main_records, variance_records, main_phase=main_phase),
        "",
        "### 6. 故障注入与恢复（spec §10）",
        "",
        fault_table(fault_records, fault_logs),
        "",
        "### 7. 成果条目（spec §15.1）",
        "",
        render_result_bullets(
            main_records,
            fault_records=fault_records,
            fault_logs=fault_logs,
            phase=main_phase,
        ),
        "",
        END_MARKER,
    ]
    return "\n".join(parts)


def splice_generated(existing: str, generated: str) -> str:
    """把生成段替换进文档：没有标记就追加；标记不成对就报错，绝不猜（设计 3）。"""
    has_begin = BEGIN_MARKER in existing
    has_end = END_MARKER in existing
    if has_begin != has_end:
        raise ReportError(
            "文档里的生成段标记不成对：只剩一个标记，无法判断该替换哪一段。"
            "先手工修好标记（或删掉残留标记），再让渲染器接管。"
        )
    if not has_begin:
        body = existing.rstrip("\n")
        return f"{body}\n\n{generated}\n" if body else f"{generated}\n"
    start = existing.index(BEGIN_MARKER)
    end = existing.index(END_MARKER, start) + len(END_MARKER)
    return existing[:start] + generated.rstrip("\n") + existing[end:]


def _first_difference(current: str, expected: str) -> str:
    """定位第一处不同并给出行号：只报「不一致」会让人从头找，等于没有信息。"""
    current_lines = current.splitlines()
    expected_lines = expected.splitlines()
    for index, (left, right) in enumerate(zip(current_lines, expected_lines), start=1):
        if left != right:
            return (
                f"第 {index} 行与 JSONL 不一致：\n"
                f"  文档：{left}\n"
                f"  数据：{right}"
            )
    return (
        f"生成段行数与数据不一致：文档 {len(current_lines)} 行、"
        f"数据 {len(expected_lines)} 行（有人手改过生成段）"
    )


def check_document(path: str | Path, generated: str) -> str | None:
    """文档与数据是否一致：一致返回 `None`，不一致返回差异描述（设计 2）。"""
    target = Path(path)
    if not target.exists():
        return f"文档不存在：{target}（先跑一次渲染生成它）"
    existing = target.read_text(encoding="utf-8")
    expected = splice_generated(existing, generated)
    if existing == expected:
        return None
    return _first_difference(existing, expected)


def calibration_from_json(payload: dict[str, Any]) -> CalibrationReport:
    """把 Task 34 落盘的校准 JSON 还原成 `CalibrationReport`。

    为什么要转换而不是留着 dict 直接用：字段名一旦对不上（比如有人把 `kappa` 写成 `k`），
    报告会静默少一列；经数据类构造就等于顺手做了一次 schema 校验，缺字段立刻抛。
    """
    required = (
        "total",
        "agreement_rate",
        "kappa",
        "judge_pass_rate",
        "human_pass_rate",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ReportError(
            f"校准 JSON 缺字段 {missing}：需要 Task 34 的完整校准报告"
        )
    return CalibrationReport(
        total=int(payload["total"]),
        agreement_rate=float(payload["agreement_rate"]),
        kappa=float(payload["kappa"]),
        judge_pass_rate=float(payload["judge_pass_rate"]),
        human_pass_rate=float(payload["human_pass_rate"]),
        divergences=tuple(
            Divergence(**item) for item in payload.get("divergences", ())
        ),
        cost_cny=float(payload.get("cost_cny", 0.0)),
    )


def load_calibration(path: str | Path) -> CalibrationReport | None:
    """读校准 JSON；文件不存在返回 `None`（报告里如实写「未提供」，不编一个数出来）。"""
    target = Path(path)
    if not target.exists():
        return None
    return calibration_from_json(json.loads(target.read_text(encoding="utf-8")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 eval/runs/*.jsonl 渲染成对外结果文档（spec §15）"
    )
    parser.add_argument("--runs", default=str(DEFAULT_MAIN), help="主实验记录 JSONL")
    parser.add_argument(
        "--variance", default=str(DEFAULT_VARIANCE), help="重复实验记录 JSONL（可缺）"
    )
    parser.add_argument(
        "--fault", default=str(DEFAULT_FAULT_OUT), help="故障注入记录 JSONL（可缺）"
    )
    parser.add_argument(
        "--logs",
        default=str(DEFAULT_FAULT_LOGS),
        help="故障注入日志 JSONL（与 --fault 成对）",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION),
        help="judge 校准 JSON（Task 34 产出，可缺）",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="结果文档路径")
    parser.add_argument(
        "--check", action="store_true", help="只校验文档与 JSONL 是否一致，不写文件"
    )
    parser.add_argument(
        "--stdout", action="store_true", help="把生成段打到标准输出，不写文件"
    )
    args = parser.parse_args(argv)

    main_records = load_records(args.runs)
    if not main_records:
        print(f"没有读到主实验记录：{args.runs}（先跑 eval.runners.run_experiment）")
        return EXIT_NO_DATA

    fault_records = load_records(args.fault)
    # 日志只在有记录时才读：对不上号的日志交给 summarize_faults 抛错，
    # 不允许出现"某条记录不知道属于哪档故障率"（Task 36 设计 7）
    fault_logs = load_fault_logs(args.logs) if fault_records else []
    generated = build_generated_block(
        main_records=main_records,
        variance_records=load_records(args.variance),
        fault_records=fault_records,
        fault_logs=fault_logs,
        calibration=load_calibration(args.calibration),
    )

    if args.stdout:
        print(generated)
        return EXIT_OK

    if args.check:
        problem = check_document(args.out, generated)
        if problem is None:
            print(f"OK：{args.out} 与 {args.runs} 一致")
            return EXIT_OK
        print(problem)
        return EXIT_DRIFT

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8") if target.exists() else DOC_HEADER
    target.write_text(splice_generated(existing, generated), encoding="utf-8")
    print(f"已写入 {target}（校验：python -m eval.runners.render_report --check）")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
