"""故障注入实验：把"工具会不会挂"变成可控的实验条件（spec §10）。

这个模块只做一件事：在同一个任务、同一套语料、同一个系统上，把工具失败率从 0% 调到
20%、50%，看恢复率与最终成功率怎么变。

两条定义边界写在最前面，因为报告里的每个数字都要按它读：

- **故障只注入在网关层。** 被测系统收到的异常与真实工具失败形状一致，它无从知道
  自己正在被测试；在系统内部插桩（"到第 3 步就失败"）会让系统有机会走特殊分支，
  测出来的就不是真实恢复能力（设计 1）。
- **恢复率与最终成功率是两件事。** 前者问"故障发生时还能不能给出答案"，后者问
  "最后答对了没有"。降级路径能保证前者，保证不了后者（设计 6）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from akr.agents.tool_gateway import ToolCallError, ToolGateway

# 同 `run_experiment.py`：`eval/` 不在可编辑安装的 `src/` 里，直接执行脚本时
# sys.path[0] 是 `eval/runners/`，`import eval.*` 会 ModuleNotFoundError。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judges.judge import JUDGE_PASS_THRESHOLD
from eval.runners.run_experiment import (
    DEFAULT_TASKSET,
    JUDGE_BUDGET_CNY,
    RUNS_DIR,
    BudgetExceeded,
    BudgetGuard,
    ExperimentError,
    JudgeScorer,
    RecordingGateway,
    RunRecord,
    Scorer,
    SystemOutput,
    artifact_evidence_reader,
    run_system,
    save_records,
    select_variance_subset,
    tool_context_from_env,
)
from eval.runners.run_ledger import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_MAX_RUNS,
    RunCountExceeded,
    RunLedger,
    RunLedgerError,
)
from eval.taskset.schema import EvalTask, load_taskset

# ---- 实验条件写死在代码里：故障曲线上的每个点都能从这里追溯（同 Task 35 的立场）----
FAULT_RATES: tuple[float, ...] = (0.0, 0.2, 0.5)   # 0% 必须有：它是曲线的参照点（设计 5）
FAULT_MODES: tuple[str, ...] = ("error", "timeout")
FAULT_SEED = 20260924      # 与 Task 35 的采样种子分开：两件事不共用一个随机流
FAULT_REPEATS = 3          # 每档 3 次：单次运行的波动不该被读成"故障率的影响"
FAULT_TASK_LIMIT = 6       # 6 题 × 3 档 × 2 模式 × 3 次 = 108 次运行（成本见「实现要点」）
FAULT_BUDGET_CNY = 1.0     # 故障实验单独一本账：触顶即停，已跑完的记录照样落盘
FAULT_SYSTEM = "ours"      # 只测有降级路径的目标系统；两组的横向对照由 Task 35 负责
DEFAULT_FAULT_OUT = RUNS_DIR / "fault.jsonl"
DEFAULT_FAULT_LOGS = RUNS_DIR / "fault-logs.jsonl"   # 注入账：Task 37 靠它重算恢复率

# 网关的五个工具名写全，是为了让 FaultConfig 里的拼写错误当场报错
GATEWAY_TOOLS: tuple[str, ...] = (
    "search_documents",
    "walk_graph",
    "resolve_entity",
    "fetch_chunk",
    "list_documents",
)
DEFAULT_FAULT_TOOLS: tuple[str, ...] = ("search_documents",)

_FAULT_MESSAGES = {
    "error": "检索工具注入故障：调用失败",
    "timeout": "检索工具注入故障：调用超时",
}


class InjectedFault(ToolCallError):
    """注入的工具失败。

    继承 `ToolCallError` 而不是随便抛一个异常（设计 3）：系统的降级路径就是按这个类型
    写的。注入一个系统 catch 不到的异常，测出来的"系统崩溃了"只是我们自己的测试脚手
    架坏了，跟可靠性无关。
    """


class InjectedTimeout(InjectedFault):
    """注入的工具超时。

    真实网关目前还没有超时检测，但超时是工具层最常见的失败形态；先把它建成独立类型，
    注入出来的故障谱才和真实世界对得上，报告里也才敢把两种模式分开画曲线。
    """


@dataclass(frozen=True)
class FaultConfig:
    """一次故障注入的条件。

    `tools` 默认只碰检索：目标系统的降级路径就长在检索这条工具上。想注入别的工具时
    必须显式写出来，因为"碰了哪些工具"是实验条件的一部分，不能隐含在代码里。
    """

    rate: float
    mode: str = "error"
    seed: int = FAULT_SEED
    tools: tuple[str, ...] = DEFAULT_FAULT_TOOLS

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"故障率必须在 [0, 1] 之间：{self.rate}")
        if self.mode not in FAULT_MODES:
            raise ValueError(f"未知故障模式：{self.mode}；可选：{FAULT_MODES}")
        if not self.tools:
            raise ValueError("至少要指定一个被注入的工具")
        unknown = sorted(set(self.tools) - set(GATEWAY_TOOLS))
        if unknown:
            # 拼错的工具名如果不报错，实验会安静地"注入 0 次"，然后报告里
            # 出现一条平坦的曲线——比报错危险得多
            raise ValueError(f"未知工具：{unknown}；可选：{GATEWAY_TOOLS}")


@dataclass(frozen=True)
class FaultInjectionLog:
    """一副骰子的账：目标工具被调了几次、注入了几次。

    `total_calls` 只数**射程内**的工具（默认 `search_documents`）：非目标工具不可能被
    注入，把它们算进分母只会把注入率稀释成一个没有含义的数字。
    """

    total_calls: int
    injected: int
    injected_rate: float


class FaultyGateway:
    """按固定序列在目标工具上注入失败的网关。

    五个工具方法全部透传，只有射程内的工具会被改写。**每次实验运行都新建一个实例**：
    `random.Random(seed)` 从零开始，于是同一档位下每次运行都在同样的调用位置注入，
    重复运行之间才可比；若跨运行复用同一个实例，第 2 次运行的故障位置就接着第 1 次
    的随机流往后排，两次运行根本不是同一组条件（设计 2）。
    """

    def __init__(self, inner: ToolGateway, config: FaultConfig) -> None:
        self._inner = inner
        self._config = config
        self._random = random.Random(config.seed)
        self._total_calls = 0
        self._injected = 0

    def log(self) -> FaultInjectionLog:
        rate = self._injected / self._total_calls if self._total_calls else 0.0
        return FaultInjectionLog(
            total_calls=self._total_calls,
            injected=self._injected,
            injected_rate=rate,
        )

    def _fault(self) -> InjectedFault:
        message = _FAULT_MESSAGES[self._config.mode]
        if self._config.mode == "timeout":
            return InjectedTimeout(message)
        return InjectedFault(message)

    def _guard(self, tool: str) -> None:
        """调用前判定：只有射程内的工具才计数、才掷骰子。"""
        if tool not in self._config.tools:
            return
        self._total_calls += 1
        if self._random.random() < self._config.rate:
            self._injected += 1
            raise self._fault()

    # ---- 以下是五个工具的透传：签名与 ToolGateway 一致，参数原样转发 ----

    def search_documents(
        self,
        query: str,
        top_k: int = 8,
        doc_filters: Any = None,
        two_stage_docs: int = 0,
        expand_neighbors: int = 0,
    ) -> dict[str, Any]:
        self._guard("search_documents")
        return self._inner.search_documents(
            query,
            top_k=top_k,
            doc_filters=doc_filters,
            two_stage_docs=two_stage_docs,
            expand_neighbors=expand_neighbors,
        )

    def walk_graph(
        self, entity: str, hops: int = 2, relation_types: Any = None
    ) -> dict[str, Any]:
        self._guard("walk_graph")
        return self._inner.walk_graph(entity, hops=hops, relation_types=relation_types)

    def resolve_entity(self, mention: str, context: str = "") -> dict[str, Any]:
        self._guard("resolve_entity")
        return self._inner.resolve_entity(mention, context=context)

    def fetch_chunk(self, chunk_id: str) -> dict[str, Any]:
        self._guard("fetch_chunk")
        return self._inner.fetch_chunk(chunk_id)

    def list_documents(self) -> dict[str, Any]:
        self._guard("list_documents")
        return self._inner.list_documents()

    def close(self) -> None:
        self._inner.close()


@dataclass(frozen=True)
class FaultRunLog:
    """一次运行（一个任务 × 一次重复）对应的故障条件与注入账。

    故障条件不写进 `RunRecord`（那是 Task 35 定死的格式，不该为一次实验加字段），
    而是留在这里，靠 `run_id` 一对一关联（设计 7）。
    """

    mode: str
    rate: float
    repeat: int
    task_id: str
    run_id: str
    log: FaultInjectionLog

    def to_jsonl_line(self) -> str:
        """一行 JSON：故障条件与注入账一起落盘。

        没有这份账，报告层就只剩"哪档跑了多少次"，算不出"几次真的被注入了故障"——
        恢复率与最终成功率都必须建立在"这次运行确实坏了"这个前提上。
        """
        return json.dumps(
            {
                "mode": self.mode,
                "rate": self.rate,
                "repeat": self.repeat,
                "task_id": self.task_id,
                "run_id": self.run_id,
                "log": asdict(self.log),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> "FaultRunLog":
        """读回一行。坏行直接抛（`KeyError` / `TypeError`），绝不静默丢数。"""
        payload = json.loads(line)
        log = payload.pop("log")
        return cls(log=FaultInjectionLog(**log), **payload)


@dataclass(frozen=True)
class FaultReport:
    """一个（档位 × 模式）格子的结果。

    `recovery_rate` 的分母是**真的发生过注入的运行数**（`injected_runs`），不是全部
    运行数：0% 档没有故障可恢复，这一格按"不可测"写 `None`，而不是写 1.0
    （"零故障时我们恢复得完美"）或 0.0（"我们完全没恢复"）——两种填法都是在替读者
    下结论，和 Task 35 把 `tool_call_success` 写成 `None` 是同一条原则。
    """

    fault_rate: float
    mode: str
    runs: int
    injected_calls: int
    injected_runs: int
    recovered: int
    recovery_rate: float | None
    final_success_rate: float
    degraded_rate: float


def summarize_faults(
    records: Sequence[RunRecord],
    logs: Sequence[FaultRunLog],
) -> list[FaultReport]:
    """按（档位 × 模式）汇总故障实验。

    排序是「档位升序、模式名升序」，于是报告里的行顺序与故障曲线一致，读表的人不用
    自己重排。日志与记录必须一一对应：缺一条就抛——那意味着落盘环节丢了东西，此时
    算出来的每一格都不可信。
    """
    by_id = {record.run_id: record for record in records}
    groups: dict[tuple[str, float], list[FaultRunLog]] = {}
    for entry in logs:
        if entry.run_id not in by_id:
            raise ExperimentError(f"找不到记录：故障日志里的 run_id 在本次记录中不存在：{entry.run_id}")
        groups.setdefault((entry.mode, entry.rate), []).append(entry)

    reports: list[FaultReport] = []
    for mode, rate in sorted(groups, key=lambda key: (key[1], key[0])):
        entries = groups[(mode, rate)]
        rows = [by_id[entry.run_id] for entry in entries]
        injected_runs = sum(1 for entry in entries if entry.log.injected > 0)
        # "恢复"的口径：这次运行确实被注入了故障，但最后还是拿到了分数
        recovered = sum(
            1
            for entry, record in zip(entries, rows)
            if entry.log.injected > 0 and record.judge_score is not None
        )
        # "最终成功"的口径：分数过了二值化阈值（Task 34 的 JUDGE_PASS_THRESHOLD）
        passed = sum(
            1
            for record in rows
            if record.judge_score is not None and record.judge_score >= JUDGE_PASS_THRESHOLD
        )
        count = len(rows)
        reports.append(
            FaultReport(
                fault_rate=rate,
                mode=mode,
                runs=count,
                injected_calls=sum(entry.log.injected for entry in entries),
                injected_runs=injected_runs,
                recovered=recovered,
                recovery_rate=(recovered / injected_runs) if injected_runs else None,
                final_success_rate=passed / count,
                degraded_rate=sum(1 for record in rows if record.degraded) / count,
            )
        )
    return reports


def save_fault_logs(path: str | Path, logs: Sequence[FaultRunLog]) -> Path:
    """把注入账**追加**写进 JSONL，与 Task 35 的记录文件同城同格式（一行一条）。

    追加而不是覆盖：同一档位重跑一次是为了"多攒几次重复"，覆盖写会把上一轮攒的
    证据抹掉；而恢复率恰恰是最需要样本量的指标。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for entry in logs:
            handle.write(entry.to_jsonl_line() + "\n")
    return target


def load_fault_logs(path: str | Path) -> list[FaultRunLog]:
    """读回注入账：空行跳过，坏行直接抛。"""
    target = Path(path)
    if not target.exists():
        return []
    logs: list[FaultRunLog] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                logs.append(FaultRunLog.from_jsonl_line(line))
    return logs


def _bind(
    run_task: Callable[[EvalTask, ToolGateway], SystemOutput],
    gateway: ToolGateway,
) -> Callable[[EvalTask], SystemOutput]:
    """把"这次运行该用的网关"绑进 `run_task`。

    绑在这一层、而不是把故障配置塞进系统的构造函数：被测系统的接口里不该出现"我正在
    被注入故障"这件事（设计 1）。`run_system` 要的是 `(task) -> SystemOutput`，
    于是这里补上那一个参数。
    """

    def bound(task: EvalTask) -> SystemOutput:
        return run_task(task, gateway)

    return bound


def run_fault_experiment(
    *,
    tasks: Sequence[EvalTask],
    run_task: Callable[[EvalTask, ToolGateway], SystemOutput],
    scorer: Scorer,
    fetch_evidence: Callable[[str], str],
    gateway: ToolGateway,
    rates: Sequence[float] = FAULT_RATES,
    modes: Sequence[str] = FAULT_MODES,
    repeats: int = FAULT_REPEATS,
    seed: int = FAULT_SEED,
    guard: BudgetGuard | None = None,
    judge_guard: BudgetGuard | None = None,
    ledger: RunLedger | None = None,
    clock: Callable[[], float] = time.monotonic,
    sink: list[RunRecord] | None = None,
    log_sink: list[FaultRunLog] | None = None,
) -> tuple[list[RunRecord], list[FaultReport]]:
    """跑完整张故障矩阵：模式 × 档位 × 重复 × 题。

    循环顺序固定，且**每一格都新建网关**：注入序列与调用次序绑定，于是运行之间的差异
    只能来自系统本身，不能来自"这次骰子从哪儿接着摇"（设计 2）。

    `sink` / `log_sink` 是给中断准备的：预算触顶时 `BudgetExceeded` 照抛（不吞异常），
    但已经跑完的记录与注入账会实时追加进调用方给的列表，落盘那一侧才不会只剩一个空
    文件。两份脚手架之所以都要落盘：记录喂指标，注入账喂恢复率——少任何一份，
    报告里的那一栏就只能靠猜。

    `ledger` 是**整场实验的运行次数**闸门（默认与 Task 35 共用 `data/run-ledger.json`）：
    故障矩阵本来就有 108 次运行，不跟主实验记在同一本账上，"60 次"这个上限就只管得住一半。
    judge 的调用依旧不进这本账——它跟着运行走，钱记在 `judge_guard` 那本。
    """
    records: list[RunRecord] = []
    logs: list[FaultRunLog] = []
    for mode in modes:
        for rate in rates:
            config = FaultConfig(rate=rate, mode=mode, seed=seed)
            # 档位与模式进 phase：run_id 由 phase 拼出来，6 个格子的 id 因此不会互撞，
            # 报告层也能按 phase 直接筛出某一档（Task 35 的 run_id 规则原样复用，设计 7）
            phase = f"fault-{mode}-{rate:.1f}"
            for repeat in range(1, repeats + 1):
                for task in tasks:
                    if ledger is not None:
                        # 先记账、后调用：中断时账本只会多记，不会少记
                        ledger.claim(phase=phase, system=FAULT_SYSTEM, task_id=task.task_id)
                    faulty = FaultyGateway(gateway, config)
                    # 包法必须是 RecordingGateway(FaultyGateway(真实网关))（设计 4）：
                    # 注入的失败照样进调用账，tool_call_success 才同时反映两种失败；
                    # 反过来包在里层，注入的失败就绕过了记账，指标会虚高
                    recorded = RecordingGateway(faulty)
                    record = run_system(
                        FAULT_SYSTEM,
                        task,
                        phase=phase,
                        repeat=repeat,
                        run_task=_bind(run_task, recorded),
                        scorer=scorer,
                        fetch_evidence=fetch_evidence,
                        gateway=recorded,
                        guard=guard,
                        judge_guard=judge_guard,
                        clock=clock,
                    )
                    records.append(record)
                    if sink is not None:
                        sink.append(record)
                    entry = FaultRunLog(
                        mode=mode,
                        rate=rate,
                        repeat=repeat,
                        task_id=task.task_id,
                        run_id=record.run_id,
                        log=faulty.log(),
                    )
                    logs.append(entry)
                    if log_sink is not None:
                        log_sink.append(entry)
    return records, summarize_faults(records, logs)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：跑一场故障注入实验，把记录追加进 JSONL，把汇总打到 stdout。

    刻意与 Task 35 的 runner 分开：那边的阶段是 `main` / `variance`，这里的阶段是
    `fault-<mode>-<rate>`，两套记录写进不同文件，互不覆盖（设计 7）。
    """
    # 重依赖延后 import：单测 import 本模块时不必加载向量库 / MCP / neo4j 那一整套
    from akr.agents.tool_gateway import InProcessGateway
    from akr.config import Settings
    from akr.mcp_server.server import DEFAULT_ARTIFACTS_ROOT, DEFAULT_DB_PATH
    from akr.model.client import LLMClient
    from akr.observability.trace import summarize_run
    from akr.orchestration.runner import run_once
    from akr.storage import sqlite_store
    from akr.storage.artifact_store import ArtifactStore

    parser = argparse.ArgumentParser(description="工具故障注入实验（恢复率与最终成功率分离度量）")
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET), help="任务集 JSONL 路径")
    parser.add_argument(
        "--limit",
        type=int,
        default=FAULT_TASK_LIMIT,
        help="题量（按题型分层抽，默认 6 题；矩阵大小 = 题量 × 档位 × 模式 × 重复）",
    )
    parser.add_argument(
        "--rates",
        default=",".join(f"{rate:g}" for rate in FAULT_RATES),
        help="逗号分隔的故障率档位，默认 0,0.2,0.5（0 档不许去掉，它是曲线的参照点）",
    )
    parser.add_argument("--modes", default=",".join(FAULT_MODES), help="逗号分隔的故障模式")
    parser.add_argument("--repeats", type=int, default=FAULT_REPEATS, help="每格重复次数")
    parser.add_argument("--out", default=str(DEFAULT_FAULT_OUT), help="记录输出路径")
    parser.add_argument(
        "--logs",
        default=str(DEFAULT_FAULT_LOGS),
        help="注入账输出路径（Task 37 靠它重算恢复率，别删）",
    )
    parser.add_argument("--peak", action="store_true", help="按高峰价计价（默认低峰价，spec §7.7）")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "整场实验的运行次数硬上限（默认沿用账本，账本不存在时 "
            f"{DEFAULT_MAX_RUNS} 次）；故障矩阵默认 108 次，装不进 60 次的账"
        ),
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help="运行次数账本路径（默认 data/run-ledger.json；与对照实验 runner 共用一本账）",
    )
    args = parser.parse_args(argv)

    # 次数账本先于一切重依赖检查（与 Task 35 同一条）：账本不可用、或额度已用完时，
    # 连工具上下文都不该去搭——故障矩阵本来就装不进 60 次的账，更不该跑半场才发现
    try:
        ledger = RunLedger(args.ledger, max_runs=args.max_runs)
    except RunLedgerError as exc:
        parser.error(str(exc))
    if ledger.cap_changed:
        print(
            f"[warn] 运行次数上限被显式改为 {ledger.max_runs} 次（账本：{args.ledger}）——"
            f"这是看得见的一次改闸，不是默认值",
            file=sys.stderr,
        )
    if ledger.remaining == 0:
        print(
            json.dumps(
                {
                    "stopped": (
                        f"运行次数已用完：{ledger.used}/{ledger.max_runs} 次"
                        f"（账本：{args.ledger}；要抬上限显式传 --max-runs，要开新账用 --ledger）"
                    )
                },
                ensure_ascii=False,
            )
        )
        return 4

    settings = Settings.from_env()
    db_path = os.environ.get("AKR_DB_PATH", str(DEFAULT_DB_PATH))
    artifacts_root = os.environ.get("AKR_ARTIFACTS_ROOT", str(DEFAULT_ARTIFACTS_ROOT))
    conn = sqlite_store.connect(db_path)
    sqlite_store.init_schema(conn)
    artifacts = ArtifactStore(artifacts_root)
    # 与 Task 35 同样用同进程网关：故障注入测的是编排层的恢复能力，
    # 不是 MCP 进程通信本身的稳定性（那是另一件事，需要另一组实验）
    context = tool_context_from_env(
        db_path=db_path,
        artifacts_root=artifacts_root,
        neo4j_uri=os.environ.get("AKR_NEO4J_URI"),
    )
    gateway = InProcessGateway(context)
    client = LLMClient(settings)
    scorer = JudgeScorer(client, peak=args.peak)
    fetch_evidence = artifact_evidence_reader(artifacts)

    def run_ours(task: EvalTask, run_gateway: ToolGateway) -> SystemOutput:
        """目标系统跑一次；`run_gateway` 是**本次运行**该用的网关（可能带着故障）。

        不传 `budget` = `run_once` 用 `Budget()` 的默认五维上限：那是**单次运行**的
        熔断（Task 25），与实验层的 `BudgetGuard`（整场实验的现金）是两本账，
        混用会让"预算触顶"在两处各有一套判定（Task 35 的实现要点 5 说的是同一件事）。
        """
        result = run_once(
            question=task.question,
            gateway=run_gateway,
            client=client,
            conn=conn,
            artifacts=artifacts,
        )
        summary = summarize_run(conn, result.run_id)
        return SystemOutput(
            answer=result.answer,
            citations=tuple(citation.chunk_id for citation in result.citations),
            cost_cny=result.cost_cny,
            cost_source="metered",
            steps=result.steps,
            illegal_transitions=summary.illegal_transitions,
            latency_ms=0.0,          # 秒表由 run_system 统一持有
            degraded=result.degraded,
            state=result.state,
        )

    tasks = load_taskset(args.taskset)
    # 复用 Task 35 的分层采样：故障曲线必须在题型上也有代表，否则"50% 故障下成功率
    # 掉了一半"可能只是抽到的题变难了（spec §9.4 的同一套理由）
    subset = select_variance_subset(tasks, size=args.limit, seed=FAULT_SEED)
    rates = [float(item) for item in args.rates.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if not rates or not modes:
        parser.error("--rates 与 --modes 都不能为空")

    # 两本账：系统调用一本、judge 一本（与 Task 35 一致，spec §7.7）
    guard = BudgetGuard(FAULT_BUDGET_CNY)
    judge_guard = BudgetGuard(JUDGE_BUDGET_CNY)
    out_path = Path(args.out)
    logs_path = Path(args.logs)

    collected: list[RunRecord] = []
    collected_logs: list[FaultRunLog] = []
    reports: list[FaultReport] = []
    exit_code = 0
    try:
        _, reports = run_fault_experiment(
            tasks=subset,
            run_task=run_ours,
            scorer=scorer,
            fetch_evidence=fetch_evidence,
            gateway=gateway,
            rates=rates,
            modes=modes,
            repeats=args.repeats,
            guard=guard,
            judge_guard=judge_guard,
            ledger=ledger,           # 与 Task 35 共用一本账：60 次是整场实验的总数
            sink=collected,          # 触顶中断时，已跑完的记录不会跟着异常一起消失
            log_sink=collected_logs,  # 注入账同理：恢复率的分母不能丢
        )
    except BudgetExceeded as exc:
        # 触顶即停：半张故障曲线也是证据，丢掉它才是最坏的结果。
        # 中断路径不打汇总（矩阵不完整，任何一格的比率都不该被当作结论引用）
        exit_code = 3
        print(json.dumps({"stopped": str(exc), "records": len(collected)}, ensure_ascii=False))
    except RunCountExceeded as exc:
        # 次数闸门触顶同样是"矩阵不完整"：只落盘、不汇总（同 Task 35 的退出码 4）
        exit_code = 4
        print(json.dumps({"stopped": str(exc), "records": len(collected)}, ensure_ascii=False))
    finally:
        save_records(out_path, collected)
        # 注入账与记录一起落盘：Task 37 读这两份文件就能离线重算每个格子（设计 7）
        save_fault_logs(logs_path, collected_logs)
        gateway.close()

    print(json.dumps({"run_ledger": ledger.state.to_json_dict()}, ensure_ascii=False))
    print(json.dumps([asdict(report) for report in reports], ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":  # pragma: no cover - 仅作为 CLI 入口
    raise SystemExit(main())
