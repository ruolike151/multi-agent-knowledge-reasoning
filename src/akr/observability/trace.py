"""追踪：状态转移、模型调用、工具调用各落一条事件，同时写 SQLite 与 JSONL。

为什么两个落点都要：

- SQLite：给程序读。评测 runner（Task 35）与 `summarize_run` 都直接查表聚合。
- JSONL：给人读。一次运行的完整时序就是一份可直接贴进报告的证据，且进程崩溃后仍可复盘。

为什么不用 OpenTelemetry：本项目的 trace 消费者只有「自己的指标函数」和「自己看」，
引入 OTel 只会多一层 SDK 依赖与一个不可独立评测的失败点（spec §16 YAGNI）。
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

ALLOWED_KINDS: tuple[str, ...] = ("state", "llm", "tool", "error", "note")

DEFAULT_DETAIL_LIMIT = 500
TRUNCATION_SUFFIX = "…[truncated]"

_TABLE = "trace_events"


@dataclass(frozen=True)
class TraceEvent:
    """一条 trace 记录。字段名与 SQLite 列名逐字一致，避免任何映射层。"""

    run_id: str
    seq: int
    ts: float
    kind: str
    name: str
    status: str = "ok"
    prev_state: str | None = None
    state: str | None = None
    duration_ms: int = 0
    model: str | None = None
    model_version: str | None = None
    mode: str | None = None
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    cost_cny: float = 0.0
    detail: str = ""

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_cached + self.tokens_out

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "TraceEvent":
        """按列顺序还原事件；列数不符直接报错，不做静默截断。"""

        if len(row) != len(_COLUMNS):
            raise ValueError(f"trace 行字段数不匹配：期望 {len(_COLUMNS)}，实际 {len(row)}")
        return cls(**dict(zip(_COLUMNS, row)))


_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(TraceEvent))
_COLUMN_LIST = ", ".join(_COLUMNS)


def init_trace_schema(conn: sqlite3.Connection) -> None:
    """幂等建表。trace 表由本模块自管，避免 storage 层反向依赖上层（见 README 文件结构）。"""

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            run_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ok',
            prev_state TEXT,
            state TEXT,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            model TEXT,
            model_version TEXT,
            mode TEXT,
            tokens_in INTEGER NOT NULL DEFAULT 0,
            tokens_cached INTEGER NOT NULL DEFAULT 0,
            tokens_out INTEGER NOT NULL DEFAULT 0,
            cost_cny REAL NOT NULL DEFAULT 0.0,
            detail TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (run_id, seq)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS trace_events_kind_idx ON {_TABLE} (run_id, kind)"
    )
    conn.commit()


class TraceRecorder:
    """一次运行的 trace 写入器。用同一个 `run_id` 重新构造即为「续写」（断点恢复用）。"""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        jsonl_path: str | Path | None = None,
        detail_limit: int = DEFAULT_DETAIL_LIMIT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not run_id:
            raise ValueError("run_id 不能为空")
        if detail_limit < 1:
            raise ValueError("detail_limit 必须 >= 1")

        self._conn = conn
        self.run_id = run_id
        self.detail_limit = detail_limit
        self._clock = clock
        self._jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        init_trace_schema(conn)
        self._next_seq = self._read_next_seq()

    def _read_next_seq(self) -> int:
        row = self._conn.execute(
            f"SELECT COALESCE(MAX(seq), 0) FROM {_TABLE} WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()
        return int(row[0]) + 1

    def record(self, kind: str, name: str, **fields: Any) -> TraceEvent:
        """写一条事件。未知 `kind` 与未知字段都会报错，不会静默丢弃。"""

        if kind not in ALLOWED_KINDS:
            raise ValueError(f"kind 只能是 {list(ALLOWED_KINDS)} 之一，收到 {kind!r}")
        if "detail" in fields:
            fields["detail"] = self._truncate(str(fields["detail"]))

        event = TraceEvent(
            run_id=self.run_id,
            seq=self._next_seq,
            ts=self._clock(),
            kind=kind,
            name=name,
            **fields,
        )
        self._persist(event)
        self._next_seq += 1
        return event

    def _truncate(self, detail: str) -> str:
        if len(detail) <= self.detail_limit:
            return detail
        return detail[: self.detail_limit] + TRUNCATION_SUFFIX

    def _persist(self, event: TraceEvent) -> None:
        values = tuple(getattr(event, column) for column in _COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT INTO {_TABLE} ({_COLUMN_LIST}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_json_dict(), ensure_ascii=False) + "\n")
                handle.flush()

    def events(self, run_id: str | None = None) -> list[TraceEvent]:
        """按 seq 升序读回事件。默认读本次运行的 `run_id`。"""

        target = run_id or self.run_id
        rows = self._conn.execute(
            f"SELECT {_COLUMN_LIST} FROM {_TABLE} WHERE run_id = ? ORDER BY seq ASC",
            (target,),
        ).fetchall()
        return [TraceEvent.from_row(row) for row in rows]

    def close(self) -> None:
        """只提交，不关闭 conn：连接由构造它的调用方持有。"""

        self._conn.commit()


@dataclass(frozen=True)
class RunSummary:
    """一次运行的 trace 聚合结果（Task 33/35/37 的输入）。"""

    run_id: str
    steps: int = 0
    llm_calls: int = 0
    tokens: int = 0
    cost_cny: float = 0.0
    illegal_transitions: int = 0
    error_count: int = 0
    p50_ms: int = 0
    p95_ms: int = 0


def summarize_run(conn: sqlite3.Connection, run_id: str) -> RunSummary:
    """从 trace 聚合一次运行的指标——全项目唯一的 trace→指标入口。

    口径说明（写进报告时必须一致）：

    - `steps` = `kind="state"` 的事件数，即状态转移次数。
    - `illegal_transitions` = `kind="state"` 且 `status="rejected"` 的事件数，即状态机驳回次数。
    - `error_count` = 所有 `status="error"` 的事件数（工具失败、未预期异常、预算耗尽都算）。
    - `tokens` = `tokens_in + tokens_cached + tokens_out`，与 `TraceEvent.total_tokens` 同口径。
    - 分位数只统计 `duration_ms > 0` 的事件，避免把「纯记录型」事件算成零耗时样本。
    """

    init_trace_schema(conn)
    rows = conn.execute(
        f"SELECT kind, status, duration_ms, tokens_in, tokens_cached, tokens_out, cost_cny "
        f"FROM {_TABLE} WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    steps = llm_calls = illegal = errors = 0
    tokens = 0
    cost = 0.0
    durations: list[int] = []
    for kind, status, duration_ms, token_in, token_cached, token_out, row_cost in rows:
        if kind == "state":
            steps += 1
            if status == "rejected":
                illegal += 1
        if kind == "llm":
            llm_calls += 1
        if status == "error":
            errors += 1
        tokens += int(token_in) + int(token_cached) + int(token_out)
        cost += float(row_cost)
        if duration_ms:
            durations.append(int(duration_ms))

    return RunSummary(
        run_id=run_id,
        steps=steps,
        llm_calls=llm_calls,
        tokens=tokens,
        cost_cny=round(cost, 6),
        illegal_transitions=illegal,
        error_count=errors,
        p50_ms=_nearest_rank(durations, 50),
        p95_ms=_nearest_rank(durations, 95),
    )


def _nearest_rank(values: Sequence[int], percentile: int) -> int:
    """最近秩百分位，定义与 Task 33 的 `metrics.percentile` 一致。

    故意各留一份：评测层（Task 33）要能独立测试；若让 trace 反向 import 评测层，
    依赖方向就反了（storage/observability 不应依赖 eval）。
    """

    if not values:
        return 0
    ordered = sorted(values)
    index = math.ceil(percentile / 100 * len(ordered)) - 1
    return ordered[max(index, 0)]
