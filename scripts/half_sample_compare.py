r"""半样本重跑 vs 30 格主表：把五个口径并排算出来（只读、零 API 花费）。

**为什么要有这个脚本。** 内部偏差日志第三十五节那张「五个口径并列」的表是唯一能引用的
`ours` 前后对照——它必须能被第三个人现场重算出来，而不是只活在某次会话的临时脚本里。
本脚本只读 `eval/runs/*.jsonl`，不联网、不调模型、不写任何状态。

**五个口径为什么必须并排。** 列 2 → 列 1（30 格 → 半样本）之间同时变了两件事：**代码换了**
（图通道缺陷修好 + 检索层两步上线）**且样本减半**——直接相减的差不是「修完代码的收益」。
只有列 3（`ours` 30 格里的同一批 15 题）才是干净的「同 15 题、只差代码」对照。同理，右两列
把 `baseline_a` 也按同一批 15 题切一次，否则会拿「`ours` 15 题」去比「基线 30 题」。

**指标算法不在这里**：五个口径全部经 `eval.runners.run_experiment.summarize_records`
（→ `eval.metrics.aggregate`），与主表、与结果文档同一条代码路径。本脚本只做分组与切片，
外加几条**计数**（degraded 条数 / 最大延迟 / gold 进池题数）——计数不是指标，就地算。

用法（仓库根目录）：

    .\.venv\Scripts\python.exe scripts\half_sample_compare.py
    .\.venv\Scripts\python.exe scripts\half_sample_compare.py --json data\half-sample-compare.json
    .\.venv\Scripts\python.exe scripts\half_sample_compare.py --per-task   # 逐题并排，便于抽查

退出码：0 = 成功；1 = 输入文件缺失。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# `eval/` 不在可编辑安装的 `src/` 里，直接跑脚本时 sys.path[0] 是 `scripts/`，
# 所以要把仓库根补进来（与 `scripts/review_taskset.py` 同一处理）。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runners.run_experiment import RunRecord, summarize_records  # noqa: E402

DEFAULT_HALF = ROOT / "eval" / "runs" / "main-ours-half15.jsonl"
DEFAULT_MAIN = ROOT / "eval" / "runs" / "main.jsonl"

# 表里要摆的指标（顺序即打印顺序）；键名必须与 `aggregate` 的输出一致。
METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("recall@8", "recall@8"),
    ("mrr", "MRR"),
    ("judge_score_mean", "judge 均值"),
    ("citation_support_rate", "引用支撑率"),
    ("hallucination_rate", "幻觉率"),
    ("correct_refusal_rate", "正确拒答率"),
    ("over_refusal_rate", "过度拒答率"),
    ("steps_mean", "steps 均值"),
    ("latency_p50_ms", "延迟 p50(s)"),
    ("latency_p95_ms", "延迟 p95(s)"),
    ("cost_cny_total", "成本合计(¥)"),
)

MILLIS_PER_SECOND = 1000.0
# 表头里有中文：终端里一个汉字占两格，`str.ljust` 按字符数补空格会错列。
LABEL_WIDTH = 24
CELL_WIDTH = 18

# 表头用的短名（键必须与 `build_subsets` 的键一致）：全名太长会把列挤串。
SHORT_LABELS: dict[str, str] = {
    "ours half15 (rerun)": "ours half15",
    "ours 30 grid": "ours 30",
    "ours 30 grid (same 15)": "ours 30 same15",
    "baseline_a 30 grid": "baseline_a 30",
    "baseline_a (same 15)": "baseline_a same15",
}


class CompareError(RuntimeError):
    """输入文件缺失或不自洽。"""


def display_width(text: str) -> int:
    """终端列数：宽字符（`W`/`F`，汉字/全角）记 2，其余记 1。"""

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad_right(text: str, width: int) -> str:
    """按**终端列数**右侧补齐（`ljust` 的宽字符版本）。"""

    return text + " " * max(0, width - display_width(text))


def pad_left(text: str, width: int) -> str:
    """按**终端列数**左侧补齐（`rjust` 的宽字符版本）。"""

    return " " * max(0, width - display_width(text)) + text


def load_records(path: Path) -> list[RunRecord]:
    """按行读 JSONL；一行一条 `RunRecord`（校验交给 pydantic，不在这里手搓）。"""

    if not path.exists():
        raise CompareError(f"缺文件：{path}")
    records: list[RunRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(RunRecord.model_validate(json.loads(line)))
    if not records:
        raise CompareError(f"文件里没有记录：{path}")
    return records


def build_subsets(
    half: Sequence[RunRecord], main: Sequence[RunRecord]
) -> dict[str, list[RunRecord]]:
    """五个口径，顺序即打印顺序（左 → 右与偏差日志第三十五节的表一致）。"""

    half_ids = {record.task_id for record in half}
    return {
        "ours half15 (rerun)": list(half),
        "ours 30 grid": [r for r in main if r.system == "ours"],
        "ours 30 grid (same 15)": [
            r for r in main if r.system == "ours" and r.task_id in half_ids
        ],
        "baseline_a 30 grid": [r for r in main if r.system == "baseline_a"],
        "baseline_a (same 15)": [
            r for r in main if r.system == "baseline_a" and r.task_id in half_ids
        ],
    }


def describe(rows: Sequence[RunRecord]) -> dict[str, Any]:
    """一个口径的指标（来自 `summarize_records`）+ 几条计数。

    `summarize_records` 按 `system` 分组，所以单一口径要取它唯一的那个分组；空切片直接
    返回只带 `runs=0` 的口径，让「这半张表没数据」显式可见，而不是静默少一行。
    """

    if not rows:
        return {"runs": 0}
    summary = summarize_records(list(rows))
    entry = summary[next(iter(summary))]
    payload: dict[str, Any] = dict(entry)
    # 计数（不是指标）：退化条数、最大延迟、以及「gold 进过检索池」的题数。
    payload["degraded_runs"] = sum(1 for r in rows if r.degraded)
    payload["latency_max_s"] = max(r.latency_ms for r in rows) / MILLIS_PER_SECOND
    answerable = [r for r in rows if r.answerable]
    payload["answerable_runs"] = len(answerable)
    payload["gold_in_pool_runs"] = sum(
        1 for r in answerable if set(r.gold_ids) & set(r.retrieved_ids)
    )
    return payload


def format_metric(key: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if key.startswith("latency_p"):
        return f"{value / MILLIS_PER_SECOND:.1f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_table(descriptions: dict[str, dict[str, Any]]) -> str:
    """人读的并排表：一行一个口径、一列一个指标。"""

    labels = list(descriptions)
    headers = [SHORT_LABELS.get(label, label) for label in labels]
    lines = [
        pad_right("口径", LABEL_WIDTH) + "".join(pad_left(h, CELL_WIDTH) for h in headers),
        "-" * (LABEL_WIDTH + CELL_WIDTH * len(labels)),
        pad_right("runs", LABEL_WIDTH)
        + "".join(pad_left(str(descriptions[l].get("runs", 0)), CELL_WIDTH) for l in labels),
    ]
    for key, title in METRIC_KEYS:
        cells = [format_metric(key, descriptions[label].get(key)) for label in labels]
        lines.append(pad_right(title, LABEL_WIDTH) + "".join(pad_left(c, CELL_WIDTH) for c in cells))
    counts = ("degraded_runs", "latency_max_s", "answerable_runs", "gold_in_pool_runs")
    for key in counts:
        cells = [format_metric(key, descriptions[label].get(key)) for label in labels]
        lines.append(pad_right(key, LABEL_WIDTH) + "".join(pad_left(c, CELL_WIDTH) for c in cells))
    return "\n".join(lines)


def render_per_task(half: Sequence[RunRecord], main: Sequence[RunRecord]) -> str:
    """逐题并排：半样本这一轮 vs 30 格主表里的同一题（主表缺这题时写 `--`）。"""

    main_by_task = {r.task_id: r for r in main if r.system == "ours"}
    lines = ["逐题（half15 vs 主表同一题）"]
    for record in sorted(half, key=lambda r: r.task_id):
        other = main_by_task.get(record.task_id)
        left = (
            f"j={record.judge_score} t={record.latency_ms / MILLIS_PER_SECOND:.1f}s "
            f"steps={record.steps} deg={int(record.degraded)}"
        )
        if other is None:
            lines.append(f"  {record.task_id:16s} {record.type:11s} {left} | --")
            continue
        right = (
            f"j={other.judge_score} t={other.latency_ms / MILLIS_PER_SECOND:.1f}s "
            f"steps={other.steps} deg={int(other.degraded)}"
        )
        lines.append(f"  {record.task_id:16s} {record.type:11s} {left} | {right}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="半样本重跑 vs 30 格主表（只读）")
    parser.add_argument("--half", type=Path, default=DEFAULT_HALF, help="半样本重跑记录")
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN, help="30 格主表记录")
    parser.add_argument("--per-task", action="store_true", help="额外打印逐题并排")
    parser.add_argument("--json", type=Path, default=None, help="把五个口径落成 JSON")
    args = parser.parse_args(argv)

    half = load_records(args.half)
    main_records = load_records(args.main)
    subsets = build_subsets(half, main_records)
    print(f"[load] half={args.half}（{len(half)} 条）main={args.main}（{len(main_records)} 条）")
    descriptions = {label: describe(rows) for label, rows in subsets.items()}
    print(render_table(descriptions))
    print()
    print(
        "[口径] 列 2 与列 1 之间同时换了代码与样本量——**只有列 3 是「同 15 题、只差代码」**；\n"
        "       15 格是半样本（分层抽样、种子 20260923），不能当 30 格主表；\n"
        "       judge 均值绝对不可对外引用（校准集 `eval/judges/calibration.json` 仍未产出）。"
    )
    if args.per_task:
        print()
        print(render_per_task(half, main_records))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "half": str(args.half),
                    "main": str(args.main),
                    "subsets": descriptions,
                    "note": "指标来自 eval.runners.run_experiment.summarize_records；计数就地算",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[write] {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    try:
        sys.exit(main())
    except CompareError as error:
        print(f"[error] {error}", file=sys.stderr)
        sys.exit(1)
