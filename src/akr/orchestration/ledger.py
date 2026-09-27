"""任务账本、幂等结果与检查点（spec §5.3 / §5.4 / §10）。

三个数据面：

1. **账本**（`TaskLedger`）：Supervisor context 的唯一载体。文档原文永不进入
   Supervisor，只留 `SubTask.result_ref` 指向 artifact store 里的结构化摘要。
2. **幂等结果**（`node_results` 表）：键为 `(run_id, task_id, node)`，写一次读多次。
   参考项目把已完成节点塞进 state 的 `set`，序列化到 checkpoint 时不可靠（spec §10），
   这里改成数据库主键，持久且可查询。
3. **检查点**（`ledgers` 表）：按 `run_id` 整存账本 JSON。字段会演进，所以不拆列；
   账本规模是「每 run 十条子任务」，没有按字段查询的需求。

恢复路径的省钱逻辑：Task 31 派发子任务前先 `load_node_result`，命中即跳过 worker
调用。worker 是唯一花 LLM 钱的环节，这条短路直接决定 Task 36 的「恢复率/恢复成本」。
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from akr.orchestration.budget import BudgetState
from akr.storage.artifact_store import Artifact, ArtifactStore

SUBTASK_KINDS: tuple[str, ...] = ("retrieve", "graph", "verify")
SUBTASK_STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed")
UNFINISHED_STATUSES: tuple[str, ...] = ("pending", "running")

# `result_ref` 的分隔符。artifact store 的 doc_id / chunk_id 禁止斜杠与 `..`
# （Task 12 的路径穿越防护），所以引用不能写成 "run/task" 形式。
RESULT_REF_SEPARATOR = "#"


class LedgerFormatError(RuntimeError):
    """持久化格式不合法：缺字段、枚举越界、JSON 损坏。

    这类错误必须抛出来。静默返回空账本会让恢复路径从头再跑一遍——成本翻倍，
    而且从 trace 上完全看不出来（Global Constraints 禁止静默吞异常）。
    """


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise LedgerFormatError(f"{where} 缺少必需字段：{key}")
    return data[key]


@dataclass(frozen=True)
class Citation:
    """一条断言对应的证据位置。字段名与 spec §5.4 一致。"""

    doc_id: str
    chunk_id: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "chunk_id": self.chunk_id, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Citation:
        return cls(
            doc_id=str(_require(data, "doc_id", "citation")),
            chunk_id=str(_require(data, "chunk_id", "citation")),
            source=str(_require(data, "source", "citation")),
        )


@dataclass(frozen=True)
class CostRecord:
    """一个子任务的成本归因。字段与 trace（Task 23）口径一致。

    `tokens` 是总量（命中 + 未命中 + 输出），`hit_tokens` / `miss_tokens` /
    `output_tokens` 是同一笔用量的**明细**。明细不是装饰：编排层把 worker 的用量写回
    trace 时要落进各自的列，而 spec §7.5 的「缓存命中带来的成本下降有数据」只能从这三列
    算出来。2026-09-25 之前这里只有总量，编排层就把总量写进了 `tokens_out` 一列——
    worker 的调用因此既丢了输入明细、又凭空多出输出 token（偏差日志第三十二节）。
    """

    steps: int = 0
    tokens: int = 0
    cost_cny: float = 0.0
    latency_ms: int = 0
    hit_tokens: int = 0
    miss_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_cny": self.cost_cny,
            "latency_ms": self.latency_ms,
            "hit_tokens": self.hit_tokens,
            "miss_tokens": self.miss_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostRecord:
        return cls(
            steps=int(data.get("steps", 0)),
            tokens=int(data.get("tokens", 0)),
            cost_cny=float(data.get("cost_cny", 0.0)),
            latency_ms=int(data.get("latency_ms", 0)),
            hit_tokens=int(data.get("hit_tokens", 0)),
            miss_tokens=int(data.get("miss_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
        )


@dataclass(frozen=True)
class WorkerReport:
    """worker 回传给 Supervisor 的**结构化摘要**（spec §5.4 第 2 条）。

    不允许自由文本：`findings` 每条是一个断言，`citations` 与之一一对应，
    这样 Task 30 的引用支撑率与 Task 33 的指标才能自动计算。
    `open_questions` 是 Supervisor 决定「是否追加子任务」的输入，不参与压缩。
    """

    findings: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    open_questions: list[str] = field(default_factory=list)
    cost: CostRecord = field(default_factory=CostRecord)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": list(self.findings),
            "citations": [citation.to_dict() for citation in self.citations],
            "confidence": self.confidence,
            "open_questions": list(self.open_questions),
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerReport:
        findings = data.get("findings", [])
        open_questions = data.get("open_questions", [])
        if not isinstance(findings, list) or not isinstance(open_questions, list):
            raise LedgerFormatError("WorkerReport 的 findings / open_questions 必须是列表")
        return cls(
            findings=[str(item) for item in findings],
            citations=[Citation.from_dict(raw) for raw in data.get("citations", [])],
            confidence=float(data.get("confidence", 0.0)),
            open_questions=[str(item) for item in open_questions],
            cost=CostRecord.from_dict(data.get("cost", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> WorkerReport:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LedgerFormatError("WorkerReport 的文本不是合法 JSON") from exc
        return cls.from_dict(data)


@dataclass
class SubTask:
    """一个子任务。`goal` 是 worker 唯一可见的输入（spec §5.4 隔离规则第 1 条）。"""

    task_id: str
    goal: str
    kind: str
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)
    attempt: int = 0
    result_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "kind": self.kind,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "attempt": self.attempt,
            "result_ref": self.result_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubTask:
        kind = str(_require(data, "kind", "subtask"))
        if kind not in SUBTASK_KINDS:
            raise LedgerFormatError(
                f"未知子任务类型：{kind}；合法值：{', '.join(SUBTASK_KINDS)}"
            )
        status = str(data.get("status", "pending"))
        if status not in SUBTASK_STATUSES:
            raise LedgerFormatError(
                f"未知子任务状态：{status}；合法值：{', '.join(SUBTASK_STATUSES)}"
            )
        return cls(
            task_id=str(_require(data, "task_id", "subtask")),
            goal=str(_require(data, "goal", "subtask")),
            kind=kind,
            status=status,
            depends_on=[str(item) for item in data.get("depends_on", [])],
            attempt=int(data.get("attempt", 0)),
            result_ref=data.get("result_ref"),
        )


@dataclass
class TaskLedger:
    """一次问答的任务账本：Supervisor context 的主体，也是检查点的载荷。"""

    question: str
    answerable: bool = True
    subtasks: dict[str, SubTask] = field(default_factory=dict)
    revise_round: int = 0
    budget: BudgetState = field(default_factory=BudgetState)

    def add(self, subtask: SubTask) -> SubTask:
        """登记子任务。重复 `task_id` 抛 `ValueError`：一次运行内 id 必须唯一，
        否则幂等键 `(run_id, task_id, node)` 会指向两个不同的意图。"""

        if subtask.task_id in self.subtasks:
            raise ValueError(f"子任务 {subtask.task_id} 已存在：task_id 必须在一次运行内唯一")
        if subtask.kind not in SUBTASK_KINDS:
            raise ValueError(f"未知子任务类型：{subtask.kind}")
        self.subtasks[subtask.task_id] = subtask
        return subtask

    def mark(
        self,
        task_id: str,
        status: str,
        *,
        result_ref: str | None = None,
        bump_attempt: bool = False,
    ) -> SubTask:
        """更新子任务状态。状态枚举在这里单点校验，避免状态名拼错后只写进表不报错。"""

        if status not in SUBTASK_STATUSES:
            raise ValueError(f"未知子任务状态：{status}；合法值：{', '.join(SUBTASK_STATUSES)}")
        subtask = self.subtasks.get(task_id)
        if subtask is None:
            raise KeyError(f"账本中没有子任务 {task_id}")
        subtask.status = status
        if result_ref is not None:
            subtask.result_ref = result_ref
        if bump_attempt:
            subtask.attempt += 1
        return subtask

    def unfinished_ids(self) -> list[str]:
        """尚未终态的子任务 id（保持登记顺序）。Task 31 用它判断是否还要回 WORKING。"""

        return [
            task_id
            for task_id, subtask in self.subtasks.items()
            if subtask.status in UNFINISHED_STATUSES
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answerable": self.answerable,
            "revise_round": self.revise_round,
            "budget": self.budget.to_dict(),
            "subtasks": [subtask.to_dict() for subtask in self.subtasks.values()],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> TaskLedger:
        ledger = cls(
            question=str(_require(data, "question", "ledger")),
            answerable=bool(data.get("answerable", True)),
            revise_round=int(data.get("revise_round", 0)),
            budget=BudgetState.from_dict(_require(data, "budget", "ledger"), clock=clock),
        )
        for raw in data.get("subtasks", []):
            subtask = SubTask.from_dict(raw)
            ledger.subtasks[subtask.task_id] = subtask
        return ledger


LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledgers (
    run_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS node_results (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    node TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, task_id, node)
);
"""


def init_ledger_schema(conn: sqlite3.Connection) -> None:
    """建表。幂等：`IF NOT EXISTS` + 不做任何破坏性迁移。"""

    conn.executescript(LEDGER_SCHEMA)
    conn.commit()


def save_ledger(conn: sqlite3.Connection, ledger: TaskLedger, *, run_id: str) -> None:
    """按 `run_id` 覆盖式保存（一次运行一行）。

    用 UPSERT 而不是先 DELETE 再 INSERT：恢复后再次保存同一次运行的账本
    是正常操作（每步都可能检查点），不该产生多行，也不该报错。
    """

    payload = json.dumps(ledger.to_dict(), ensure_ascii=False)
    conn.execute(
        "INSERT INTO ledgers (run_id, payload, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(run_id) DO UPDATE SET payload = excluded.payload, "
        "updated_at = excluded.updated_at",
        (run_id, payload),
    )
    conn.commit()


def load_ledger(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> TaskLedger | None:
    """读取检查点。没有该运行返回 `None`；payload 损坏则抛 `LedgerFormatError`。"""

    row = conn.execute("SELECT payload FROM ledgers WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row[0])
    except json.JSONDecodeError as exc:
        raise LedgerFormatError(f"run_id={run_id} 的账本 payload 不是合法 JSON") from exc
    try:
        return TaskLedger.from_dict(data, clock=clock)
    except (KeyError, TypeError, ValueError) as exc:
        raise LedgerFormatError(f"run_id={run_id} 的账本结构不合法：{exc}") from exc


def record_node_result(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    task_id: str,
    node: str,
    payload: dict[str, Any],
) -> bool:
    """幂等写入节点结果。首次写入返回 `True`，重复写入返回 `False` 且**不覆盖**。

    不覆盖是刻意的：键已存在说明该节点成功跑过，覆盖会把「同输入不同输出」
    这类 bug 掩盖掉。不可序列化的 payload 让 `json.dumps` 抛 `TypeError` 向上传，
    而不是在数据库层变成一条脏记录。
    """

    text = json.dumps(payload, ensure_ascii=False)
    cursor = conn.execute(
        "INSERT OR IGNORE INTO node_results (run_id, task_id, node, payload) "
        "VALUES (?, ?, ?, ?)",
        (run_id, task_id, node, text),
    )
    conn.commit()
    return cursor.rowcount == 1


def load_node_result(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    task_id: str,
    node: str,
) -> dict[str, Any] | None:
    """读取节点结果；未命中返回 `None`（Task 31 用它决定跳过还是执行）。"""

    row = conn.execute(
        "SELECT payload FROM node_results WHERE run_id = ? AND task_id = ? AND node = ?",
        (run_id, task_id, node),
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError as exc:
        raise LedgerFormatError(
            f"节点结果不是合法 JSON：run_id={run_id} task_id={task_id} node={node}"
        ) from exc


def result_ref_for(*, run_id: str, task_id: str) -> str:
    """生成 `result_ref`。`run_id` 含分隔符会让引用无法还原，直接拒绝。"""

    if not run_id or not task_id:
        raise ValueError("run_id 与 task_id 都不能为空")
    if RESULT_REF_SEPARATOR in run_id:
        raise ValueError(f"run_id 不得包含 {RESULT_REF_SEPARATOR!r}：{run_id!r}")
    return f"{run_id}{RESULT_REF_SEPARATOR}{task_id}"


def save_worker_report(
    artifacts: ArtifactStore,
    *,
    run_id: str,
    task_id: str,
    report: WorkerReport,
) -> str:
    """把 WorkerReport 落进 artifact store，返回账本里要存的引用字符串。

    正文不进账本、更不进 Supervisor context（spec §5.4 第 3 条）：
    账本里只有一个几十字节的 `result_ref`。
    `page=0` 表示这不是 PDF 页面产物，而是运行期对象。
    """

    ref = result_ref_for(run_id=run_id, task_id=task_id)
    text = report.to_json()
    artifacts.put(
        Artifact(
            chunk_id=task_id,
            doc_id=run_id,
            text=text,
            char_range=(0, len(text)),
            page=0,
        )
    )
    return ref


def load_worker_report(artifacts: ArtifactStore, ref: str) -> WorkerReport:
    """按 `result_ref` 取回 WorkerReport。引用格式不合法抛 `LedgerFormatError`。"""

    run_id, separator, task_id = ref.partition(RESULT_REF_SEPARATOR)
    if not separator or not run_id or not task_id:
        raise LedgerFormatError(
            f"result_ref 格式不合法：{ref!r}；期望 '<run_id>{RESULT_REF_SEPARATOR}<task_id>'"
        )
    return WorkerReport.from_json(artifacts.get(run_id, task_id).text)
