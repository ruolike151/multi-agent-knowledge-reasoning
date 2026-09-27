r"""把 `eval/runs/*.jsonl` 里的运行记录接回 `data/akr.sqlite` 的 trace。

为什么需要这个脚本：评测记录与 trace **用的不是同一个 `run_id`**。

* 评测侧（`eval/runners/run_experiment.py`）把 `run_id` 拼成
  `{phase}-{system}-{task_id}-r{repeat}`，写进 `RunRecord`；
* 编排侧（`akr.orchestration.runner.run_once`）在没收到 `run_id` 时**自己生成**一个
  `secrets.token_hex(RUN_ID_BYTES)`，trace / 账本 / 检查点都挂在这个 hex 上。

`run_ours` 调 `run_once` 时没传 `run_id`，于是两本账的主键对不上：拿
`main-ours-core-sh1-001-r1` 去 `trace_events` 查，一条都查不到。归因（「这 15 条墙钟
触顶的是哪些题」）因此不能靠主键直连，只能重建映射。

本脚本的重建口径写死为两条**可核对**的线索，不用任何模型、不花 API 预算：

1. 题目文本：`ledgers.payload.question` 必须与任务集的 `question` **逐字相同**；
2. 起始时刻：`RunRecord.started_at`（带时区的 ISO 串）与 trace 首条事件的
   `ts`（epoch 秒）取最近的配对，且**一对一**（贪心按时间差升序认领）。

第 2 条是必要的：同一道题在冒烟、主实验、重跑里会被跑多次，题目文本会撞车，
只能靠时刻消歧；第 1 条是必要的：时刻只在「同一道题」内部才有意义，不能跨题配。

配对结果里给了两个**不算可信度、只算事实**的字段：

* `wall_residual_seconds`：`latency_ms` 减「trace 首事件领先量」再减 `wall_seconds`
  的残差。**这个残差必须接近 0，否则就是配错了**——推理见下；
* `budget_exhausted` 的维度，以及它**出现的顺序位置**：预算事件排在终态转移**之后**
  时，说明终态是路由自己定的（闸门只是跟着记了一笔），不是闸门逼出来的。

残差判据比对「墙钟 vs 秒表」直接比更可靠，因为 trace 的 `ts` 打在事件**结束**时刻：
`runner` 是先拿到 `result.usage` 再 `tracer.record("llm", ...)`（见 `runner.py`
的 `_call_and_charge`），所以**首条事件的 `ts` 已经包含了第一次 plan 调用的耗时**。
直接拿 `wall_seconds` 比 `latency_ms`，短运行（如 3 秒的拒答）会因为 plan 调用占了
全部耗时而被误判成配错；残差判据把这段领先量扣掉，30 条实测残差都在 −0.95s ~ −0.01s。

用法（仓库根目录）：

    # 主实验的 per-run 明细（默认只算 system=ours，`--systems all` 连基线一起）
    .\.venv\Scripts\python.exe scripts\join_run_trace.py

    # 只要汇总（按题型 × 是否触顶）
    .\.venv\Scripts\python.exe scripts\join_run_trace.py --summary

    # 落盘成 JSON（给偏差日志/报告引用）
    .\.venv\Scripts\python.exe scripts\join_run_trace.py --json data\main-trace-join.json

退出码：0 = 成功；1 = 前置条件不满足（记录/库/任务集缺文件、指纹校验不过）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from akr.orchestration.runner import FINALIZING_STATES
from akr.mcp_server.server import DEFAULT_DB_PATH

DEFAULT_RUNS = Path("eval/runs/main.jsonl")
DEFAULT_TASKSET = Path("eval/taskset/core.jsonl")
DEFAULT_DB = Path(DEFAULT_DB_PATH)

# 配错的判据：残差超过这么多秒就标 `mismatch`。见模块 docstring 的推导：
# 残差 = 评测秒表 − trace 首事件领先量 − trace 首末事件之差，配对了必然接近 0
# （实测 30 条落在 −0.95s ~ −0.01s；两秒的容差能吸收秒表精度与竞态）。
MISMATCH_SECONDS = 2.0
# trace 首条事件必然晚于 `run_once` 的秒表起点，但差了 5 分钟以上就不可能是同一次运行。
MAX_PAIR_GAP_SECONDS = 300.0


class JoinError(RuntimeError):
    """前置条件不满足（缺文件、缺表、任务集对不上）。"""


def _epoch(started_at: str) -> float:
    """`RunRecord.started_at` → epoch 秒。

    记录里写的是**带时区**的 ISO 串（`+00:00`），所以能直接换算；不带时区的一律报错，
    否则会被当成本地时间静默错 8 小时——本项目所有时刻对比都栽在这个坑上过。
    """

    stamp = datetime.fromisoformat(started_at)
    if stamp.tzinfo is None:
        raise JoinError(f"started_at 缺少时区信息，无法与 trace 对齐：{started_at!r}")
    return stamp.timestamp()


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise JoinError(f"任务集不存在：{path}")
    tasks: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tasks[row["task_id"]] = row
    if not tasks:
        raise JoinError(f"任务集是空的：{path}")
    return tasks


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise JoinError(f"运行记录不存在：{path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_traces(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """从 `ledgers` + `trace_events` 读出每条 trace 的题目与首末时刻。"""

    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    missing = {"ledgers", "trace_events"} - tables
    if missing:
        raise JoinError(f"trace 库缺表 {sorted(missing)}：{DEFAULT_DB} 不是这份实验的库")
    cursor = conn.cursor()
    traces: list[dict[str, Any]] = []
    rows = cursor.execute("SELECT run_id, payload FROM ledgers").fetchall()
    for run_id, payload in rows:
        events = cursor.execute(
            "SELECT seq, ts, kind, name, state, prev_state, detail, duration_ms,"
            " tokens_in, tokens_cached, tokens_out, cost_cny"
            " FROM trace_events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        if not events:
            continue
        ledger = json.loads(payload)
        traces.append(
            {
                "trace_run_id": run_id,
                "question": ledger.get("question", ""),
                "first_ts": float(events[0]["ts"]),
                "last_ts": float(events[-1]["ts"]),
                "events": [dict(event) for event in events],
            }
        )
    return traces


def pair_records(
    records: Sequence[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    traces: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按「同题 + 最近时刻」做一对一贪心配对。

    贪心顺序是全局时间差升序：先认领最铁的配对，避免某条记录把别人的 trace 抢走。
    """

    candidates: list[tuple[float, int, int]] = []
    for record_index, record in enumerate(records):
        task = tasks.get(record["task_id"])
        if task is None:
            raise JoinError(f"记录里的 task_id 不在任务集里：{record['task_id']}")
        started = _epoch(record["started_at"])
        for trace_index, trace in enumerate(traces):
            if trace["question"] != task["question"]:
                continue
            gap = abs(trace["first_ts"] - started)
            if gap <= MAX_PAIR_GAP_SECONDS:
                candidates.append((gap, record_index, trace_index))
    candidates.sort()

    taken_records: set[int] = set()
    taken_traces: set[int] = set()
    pairs: dict[int, tuple[int, float]] = {}
    for gap, record_index, trace_index in candidates:
        if record_index in taken_records or trace_index in taken_traces:
            continue
        taken_records.add(record_index)
        taken_traces.add(trace_index)
        pairs[record_index] = (trace_index, gap)
    return [
        {"record_index": index, "trace_index": trace_index, "gap_seconds": gap}
        for index, (trace_index, gap) in sorted(pairs.items())
    ]


def summarize_pair(record: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """把一条记录 + 一条 trace 压成偏差日志要用的字段。"""

    events = trace["events"]
    llm_events = [event for event in events if event["kind"] == "llm"]
    exhausted = [
        index
        for index, event in enumerate(events)
        if event["kind"] == "error" and event["name"] == "budget_exhausted"
    ]
    # 终态转移的第一次出现：`FINALIZING_STATES` 里的状态一旦进入就不再回头，
    # 所以「首次进入」的序号就是这条运行决定收尾的时刻。
    finalizing_at: int | None = None
    for index, event in enumerate(events):
        state = event["state"]
        if event["kind"] == "state" and state is not None:
            if state in {member.value for member in FINALIZING_STATES}:
                finalizing_at = index
                break
    return {
        "run_id": record["run_id"],
        "trace_run_id": trace["trace_run_id"],
        "system": record["system"],
        "phase": record["phase"],
        "task_id": record["task_id"],
        "type": record["type"],
        "judge_score": record.get("judge_score"),
        "state": record.get("state"),
        "degraded": bool(record.get("degraded")),
        "degraded_notice": "预算已耗尽" in (record.get("answer") or ""),
        "latency_ms": record.get("latency_ms"),
        "steps": record.get("steps"),
        "wall_seconds": round(trace["last_ts"] - trace["first_ts"], 3),
        "llm_calls": len(llm_events),
        "retrieved": len(record.get("retrieved_ids", ())),
        "gold_hit": bool(set(record.get("gold_ids", ())) & set(record.get("retrieved_ids", ()))),
        "budget_exhausted_at": exhausted,
        "finalizing_at": finalizing_at,
        # 闸门排在终态之后 ⇒ 终态不是闸门逼出来的，是路由自己定的
        "gate_after_decision": bool(
            exhausted and finalizing_at is not None and exhausted[0] > finalizing_at
        ),
        "pair_gap_seconds": None,  # 由调用方填
        "wall_residual_seconds": None,  # 由调用方填
        "mismatch": None,          # 由调用方填
    }


def build_rows(
    records: Sequence[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    traces: Sequence[dict[str, Any]],
    *,
    systems: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = None if systems is None else set(systems)
    filtered = [
        (index, record)
        for index, record in enumerate(records)
        if wanted is None or record["system"] in wanted
    ]
    pairs = pair_records([record for _, record in filtered], tasks, traces)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        index, record = filtered[pair["record_index"]]
        del index
        row = summarize_pair(record, traces[pair["trace_index"]])
        row["pair_gap_seconds"] = round(pair["gap_seconds"], 3)
        latency = float(record.get("latency_ms") or 0.0) / 1000.0
        residual = latency - pair["gap_seconds"] - row["wall_seconds"]
        row["wall_residual_seconds"] = round(residual, 3)
        row["mismatch"] = abs(residual) > MISMATCH_SECONDS
        rows.append(row)
    return rows


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'task':16}{'type':12}{'wall_s':>8}{'lat_s':>8}{'llm':>4}{'steps':>6}"
        f"{'state':9}{'deg':>4}{'notice':>7}{'judge':>7}  gate"
    )
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda item: -item["wall_seconds"]):
        gate = "—"
        if row["budget_exhausted_at"]:
            gate = "决策后" if row["gate_after_decision"] else "**打断**"
        judge = "—" if row["judge_score"] is None else f"{row['judge_score']:.2f}"
        lines.append(
            f"{row['task_id']:16}{row['type']:12}{row['wall_seconds']:8.1f}"
            f"{(row['latency_ms'] or 0) / 1000:8.1f}{row['llm_calls']:4}"
            f"{row['steps'] if row['steps'] is not None else 0:6}{str(row['state']):9}"
            f"{'是' if row['degraded'] else '否':>4}"
            f"{'是' if row['degraded_notice'] else '否':>7}"
            f"{judge:>7}"
            f"  {gate}"
        )
    return "\n".join(lines)


def format_summary(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["题型 × 闸门（`决策后` = 终态由路由决定，闸门只跟着记了一笔）", ""]
    header = f"{'type':14}{'runs':>5}{'gate':>5}{'决策后':>7}{'打断':>5}{'judge均值':>10}"
    lines.extend([header, "-" * len(header)])
    types = sorted({row["type"] for row in rows})
    for kind in types:
        group = [row for row in rows if row["type"] == kind]
        gated = [row for row in group if row["budget_exhausted_at"]]
        scored = [row["judge_score"] for row in group if row["judge_score"] is not None]
        mean = sum(scored) / len(scored) if scored else float("nan")
        lines.append(
            f"{kind:14}{len(group):5}{len(gated):5}"
            f"{sum(1 for row in gated if row['gate_after_decision']):7}"
            f"{sum(1 for row in gated if not row['gate_after_decision']):5}{mean:10.2f}"
        )
    gated = [row for row in rows if row["budget_exhausted_at"]]
    clean = [row for row in rows if not row["budget_exhausted_at"]]
    for label, group in (("触顶", gated), ("未触顶", clean)):
        scored = [row["judge_score"] for row in group if row["judge_score"] is not None]
        mean = sum(scored) / len(scored) if scored else float("nan")
        lines.append(f"\n{label}：{len(group)} 条，judge 均值 {mean:.2f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把运行记录接回 trace（只读）")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS), help="运行记录 JSONL")
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET), help="任务集 JSONL")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="trace 库 SQLite")
    parser.add_argument("--systems", default="ours", help="逗号分隔；`all` 表示全部")
    parser.add_argument("--summary", action="store_true", help="只打汇总")
    parser.add_argument("--json", default=None, help="把配对结果写成 JSON")
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks(Path(args.taskset))
        records = load_records(Path(args.runs))
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        try:
            traces = load_traces(conn)
        finally:
            conn.close()
    except JoinError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    systems = None if args.systems.strip() == "all" else [
        item.strip() for item in args.systems.split(",") if item.strip()
    ]
    wanted = len(records) if systems is None else sum(
        1 for record in records if record["system"] in systems
    )
    rows = build_rows(records, tasks, traces, systems=systems)
    print(f"记录 {Path(args.runs)}：相关 {wanted} 条，配到 trace {len(rows)} 条")
    if len(rows) != wanted:
        print(f"[warn] 有 {wanted - len(rows)} 条记录没配上 trace（同题同刻的 trace 不够）")
    mismatched = [row for row in rows if row["mismatch"]]
    if mismatched:
        print(f"[warn] {len(mismatched)} 条的残差 >{MISMATCH_SECONDS:.1f}s，先当配错查")

    print()
    print(format_summary(rows) if args.summary else format_table(rows))
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
