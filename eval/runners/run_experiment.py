"""两组系统的同口径对照实验 runner（spec §9.3 / §9.4）。

这是整份计划里唯一产出实验主表的地方，因此它只做三件事：

1. 用同一套任务、同一套工具、同一套指标函数、同一份记录格式跑两组系统；
2. 把"检索到了什么""花了多少钱"这类容易被系统自己美化的数字，改由外部计量；
3. 任何不可测的量都如实标缺（None），绝不填一个看起来合理的数。

计划原文是**三组**对照组（多一组 Baseline B：参考项目固定 DAG 原样运行）。Baseline B 已按
2026-09-24 的决定放弃（缺少"它在本机真能跑起来"的直接证据），因此本文件收窄为两组：
`baseline_a`（单 Agent ReAct）与 `ours`（真多 Agent）。偏差与理由见内部偏差日志
第二十节；由此带来的结论边界（本表只能回答"单 Agent 加工具不够用吗"，不能回答"我们的改造
是否优于原固定 DAG 实现"）必须随报告一起写明。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from akr.agents.errors import ToolResultError
from akr.agents.tool_gateway import ToolGateway
from akr.observability.trace import summarize_run
from akr.orchestration.budget import Budget
from akr.orchestration.runner import run_once
from akr.orchestration.verifier import split_claims
from akr.storage.artifact_store import ArtifactNotFound

# `eval/` 不在可编辑安装的 `src/` 里（`.venv` 只有 `__editable__.akr-0.1.0.pth` 指向 src），
# 所以 `python eval\runners\run_experiment.py` 时 sys.path[0] 是 `eval/runners/`，
# `import eval.baselines...` 会 ModuleNotFoundError（2026-09-25 实跑复现）。把仓库根补进
# sys.path，脚本被直接执行与被 pytest 导入（pythonpath=src:.）两种走法才等价。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.baselines.react_agent import MAX_WALL_SECONDS, ReActAgent
from eval.judges.judge import MAX_EVIDENCE_CHARS, entailment_support, judge_answer
from eval.metrics import MetricInput, aggregate, tool_call_success_rate
from eval.runners.run_ledger import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_MAX_RUNS,
    RunCountExceeded,
    RunLedger,
    RunLedgerError,
)
from eval.taskset.schema import EvalTask, load_taskset

# ---- 实验条件全部写死在代码里：报告里的每个数字都能从这里追溯 ----
SYSTEMS: tuple[str, ...] = ("baseline_a", "ours")
MAIN_REPEATS = 1            # 主表：核心集每题跑一次，成本优先（spec §7.7）
VARIANCE_REPEATS = 3        # 方差：子集每题跑三次，用来估重复运行的波动（spec §9.4）
VARIANCE_SUBSET_SIZE = 12
VARIANCE_SEED = 20260923    # 固定种子：同一份任务集在任何机器上抽出同一批题
MAIN_BUDGET_CNY = 2.5       # spec §7.7 给主实验的额度
JUDGE_BUDGET_CNY = 1.0      # judge 单独一本账：超了先停 judge，而不是停掉整场实验
# 单次运行的墙钟闸门，与 `akr.orchestration.budget.Budget.max_wall_seconds` 同值。
# 为什么这个数要在 runner 里再落一份常量、并配一个 CLI：2026-09-25 实测两组**同值**的
# 120 秒实际上只勒住 `ours`——`baseline_a` 最慢的一条 55.3 秒（30 格里的最大值），
# 而 `ours` 的 30 格里 14 格跑满 120 秒、5 格定向样本里 3 格被截断。于是"放宽到多少"
# 必须是一次**看得见的改闸**（`--wall-seconds`，改了打到 stderr），而不是悄悄改库里的默认值。
MAIN_WALL_SECONDS = 120.0
DEFAULT_MODEL = "deepseek-flash"
RUNS_DIR = Path("eval/runs")
DEFAULT_TASKSET = Path("eval/taskset/core.jsonl")
BUDGET_EPSILON = 1e-9       # 浮点比较容差：刚好花到上限算通过，不因 0.0000001 崩掉


class ExperimentError(RuntimeError):
    """实验条件不满足（任务集不够抽、预算上限非法、成本数为负）。"""


class BudgetExceeded(ExperimentError):
    """预算触顶：调用方应当中断整场实验，而不是"跑完再算总账"。"""


class BudgetExceededWithRecord(BudgetExceeded):
    """预算触顶，但**这一次运行的记录已经组装好了**，随异常一起交回调用方。

    为什么必须带上记录：`BudgetGuard.charge` 是在判分之后调的，此刻这一次运行的钱**已经真实
    花掉**、次数也已经被账本扣掉。2026-09-25 主实验实测，闸门顶掉的那一笔让记录随异常一起丢了
    （`data/run-ledger.json` 记了 41 次、`eval/runs/main.jsonl` 只有 40 条），既花了钱又没有证据。
    于是把记录放进异常：`execute_phase` 先落进 sink，再把中断抛给调用方。
    """

    def __init__(self, message: str, record: "RunRecord") -> None:
        super().__init__(message)
        self.record = record


# ---- 记录格式：两组系统必须用同一套字段，Task 37 只读这些字段 ----


@dataclass(frozen=True)
class SystemOutput:
    """一组系统对一道题的输出。

    刻意**不含** `retrieved_ids` / `tool_successes`：那两项由 `RecordingGateway`
    在工具调用现场计量（设计 2），系统自报的检索结果一律不采信。也刻意不含 judge 结果——
    判分属于评测侧，不是被测系统的输出。
    """

    answer: str
    citations: tuple[str, ...]
    cost_cny: float
    cost_source: str          # metered：我们自己的 Usage 计量；reported：外部进程自报；unmetered：拿不到
    steps: int
    illegal_transitions: int
    latency_ms: float
    degraded: bool
    state: str


@dataclass(frozen=True)
class ToolCallRecord:
    """一次工具调用的账：调的是哪个工具、成不成、返回了哪些 chunk。"""

    tool: str
    success: bool
    chunk_ids: tuple[str, ...] = ()


class RecordingGateway:
    """包一层真实网关，在工具调用发生的现场记账（设计 2）。

    为什么在网关层记、不让系统自报：两组系统共用同一套工具（spec §9.3），
    在网关层取 `search_documents` 返回的 chunk_id 序列，两组的检索指标就是同一次测量的产物；
    让每个系统汇报"我检索到了什么"，等于让选手自己记分。
    """

    def __init__(self, inner: ToolGateway) -> None:
        self._inner = inner
        self.calls: list[ToolCallRecord] = []

    def mark(self) -> int:
        """给当前调用序号打标记，把指标窗口限定在"这一次运行"内。"""
        return len(self.calls)

    def since(self, mark: int) -> tuple[tuple[str, ...], tuple[bool, ...]]:
        """返回 mark 之后的（去重保序的 chunk_id，逐次调用的成败）。"""
        window = self.calls[mark:]
        retrieved: list[str] = []
        seen: set[str] = set()
        for call in window:
            for chunk_id in call.chunk_ids:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    retrieved.append(chunk_id)
        return tuple(retrieved), tuple(call.success for call in window)

    def since_calls(self, mark: int) -> tuple[tuple[str, ...], ...]:
        """返回 mark 之后**每次 `search_documents` 调用各自的** chunk_id（保序保重复）。

        与 `since` 的区别是"不做跨调用去重"：`since` 服务主表口径（一次运行内所有
        检索结果拼成一条序列再截前 K），这里服务逐次调用口径——被主表截掉的那几轮
        检索在这一份里看得见。只记 `search_documents`：`fetch_chunk` 之类的调用不是
        一次"检索"，混进来会把逐次 recall 的分母弄脏。
        """
        return tuple(
            call.chunk_ids for call in self.calls[mark:] if call.tool == "search_documents"
        )

    # ---- 五个工具：只有检索需要记账到 chunk 粒度，其余四个记成败即可 ----

    def search_documents(
        self,
        query: str,
        top_k: int = 8,
        doc_filters: Any = None,
        two_stage_docs: int = 0,
        expand_neighbors: int = 0,
    ) -> dict[str, Any]:
        try:
            payload = self._inner.search_documents(
                query,
                top_k=top_k,
                doc_filters=doc_filters,
                two_stage_docs=two_stage_docs,
                expand_neighbors=expand_neighbors,
            )
        except Exception:
            # 失败也要留一条账，但异常必须原样往上抛：记账不等于吞异常
            self.calls.append(ToolCallRecord(tool="search_documents", success=False))
            raise
        try:
            chunk_ids = tuple(str(hit["chunk_id"]) for hit in payload["hits"])
        except (KeyError, TypeError) as exc:
            # 返回结构坏了也算一次失败调用：它和"引擎超时"一样，都让这一题的量测失真
            self.calls.append(ToolCallRecord(tool="search_documents", success=False))
            raise ToolResultError(f"search_documents 返回结构非法：{payload!r}") from exc
        self.calls.append(ToolCallRecord(tool="search_documents", success=True, chunk_ids=chunk_ids))
        return payload

    def walk_graph(self, entity: str, hops: int = 2, relation_types: Any = None) -> dict[str, Any]:
        return self._recorded(
            "walk_graph",
            self._inner.walk_graph,
            entity,
            hops=hops,
            relation_types=relation_types,
        )

    def resolve_entity(self, mention: str, context: str = "") -> dict[str, Any]:
        return self._recorded("resolve_entity", self._inner.resolve_entity, mention, context=context)

    def fetch_chunk(self, chunk_id: str) -> dict[str, Any]:
        return self._recorded("fetch_chunk", self._inner.fetch_chunk, chunk_id)

    def list_documents(self) -> dict[str, Any]:
        return self._recorded("list_documents", self._inner.list_documents)

    def close(self) -> None:
        self._inner.close()

    def _recorded(
        self,
        tool: str,
        func: Callable[..., dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """不关心返回内容的工具统一走这里：记一笔成败，异常照样往上抛。"""
        try:
            payload = func(*args, **kwargs)
        except Exception:
            self.calls.append(ToolCallRecord(tool=tool, success=False))
            raise
        self.calls.append(ToolCallRecord(tool=tool, success=True))
        return payload


class BudgetGuard:
    """调用前检查、触顶即抛的预算闸门（设计 6）。

    为什么不复用 Task 25 的 `BudgetState`：那个管的是**单次运行内部**五个维度的熔断（步数、
    token、墙钟…），是编排层为了不跑飞而设的；这里管的是**整场实验**的现金上限（主实验 ¥2.5 /
    judge ¥1.0），只有一维，语义是"钱花完就停手"。两者生命周期与单位都不同，混用会让
    "预算触顶"这件事在两个地方各有一套判定。

    为什么必须"调用前检查"：¥5 是硬上限，超了就没有下一次机会。跑完再算总账，
    意味着超支的那一笔已经花出去且无法回滚。
    """

    def __init__(self, limit_cny: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if limit_cny <= 0:
            raise ExperimentError(f"预算上限必须为正数：{limit_cny}")
        self._limit_cny = float(limit_cny)
        self._spent_cny = 0.0
        self._clock = clock        # 注入单调钟：单测里不必真等，也便于以后记时间戳

    @property
    def limit_cny(self) -> float:
        return self._limit_cny

    @property
    def spent_cny(self) -> float:
        return self._spent_cny

    @property
    def remaining_cny(self) -> float:
        return max(self._limit_cny - self._spent_cny, 0.0)

    def charge(self, cost_cny: float) -> None:
        """记一笔支出；超出上限就抛 `BudgetExceeded`，且**超支的那笔不落账**。"""
        cost = float(cost_cny)
        if cost < 0:
            raise ExperimentError(f"cost_cny 不能为负：{cost}")
        if self._spent_cny + cost > self._limit_cny + BUDGET_EPSILON:
            # 不落账是有意的：已花金额必须始终是"真实花掉的钱"，
            # 否则中断后的记录会显示一个比实际更大的数字，成本表就不可信了
            raise BudgetExceeded(
                f"预算触顶：已花 {self._spent_cny:.4f} 元，本笔 {cost:.4f} 元，"
                f"上限 {self._limit_cny:.2f} 元"
            )
        self._spent_cny += cost


def resolve_budget_limits(
    *,
    max_cost_cny: float,
    judge_cost_cny: float,
) -> tuple[float, float]:
    """校验两道钱闸门的上限，返回 `(系统侧, judge 侧)`。

    为什么要有这两个旋钮：`MAIN_BUDGET_CNY` / `JUDGE_BUDGET_CNY` 是 spec §7.7 分的额度，
    但用户 2026-09-25 定的控闸口径是**运行次数**（`--max-runs`，见 `run_ledger.py`）。
    两道闸门同时生效时，钱先触顶就会把"跑满 60 次"这件事打断，而中断时那一笔的记录
    还没组装完就随异常丢了——量出来的次数与花的钱都对不上。所以上限必须能在命令行上
    显式改，且改了一定打到 stderr（与 `--max-runs` 同一套"改闸要看得见"的口径）。

    非正数 / NaN 在这里就拒掉：`BudgetGuard` 自己也会拒非正数，但那时已经连过
    Neo4j 与向量库了；参数错误应该在**开跑之前**以退出码 2 结束。
    """
    checked: list[float] = []
    for flag, value in (
        ("--max-cost-cny", max_cost_cny),
        ("--judge-cost-cny", judge_cost_cny),
    ):
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ExperimentError(f"{flag} 必须是正数：{value}")
        checked.append(number)
    return checked[0], checked[1]


@dataclass(frozen=True)
class ScoreResult:
    """judge 侧的一次判分结果。

    `calls` 记模型调用次数（1 次打分 + 每条断言 1 次蕴含），成本表要能对上"每题几次调用"。
    """

    judge_score: float
    supports: tuple[bool, ...]
    cost_cny: float
    calls: int


class Scorer(Protocol):
    """判分器协议：runner 只认这个签名，换 judge 实现不影响实验流程。"""

    def score(
        self,
        *,
        task: EvalTask,
        answer: str,
        citations: Sequence[str],
        fetch_evidence: Callable[[str], str],
    ) -> ScoreResult: ...


class JudgeScorer:
    """把 Task 34 的 judge 装成 runner 认识的 `Scorer`（设计 8 的落地）。

    判分自己也会花钱、也会失败，所以它单独记 `cost_cny` / `calls`；它抛异常时
    runner 只把 `judge_score` 留空，而不是把这一题算成 0 分——"系统崩了"和
    "答案答错了"在结论叙述里是两件事，在数据里也必须是两件事。
    """

    def __init__(self, client: Any, *, peak: bool = False, repair: Any = None) -> None:
        self._client = client
        self._peak = bool(peak)     # 计价时段：DeepSeek 高峰价是低峰的两倍（spec §7.7）
        self._repair = repair

    def score(
        self,
        *,
        task: EvalTask,
        answer: str,
        citations: Sequence[str],
        fetch_evidence: Callable[[str], str],
    ) -> ScoreResult:
        evidence = _collect_evidence(citations, fetch_evidence)
        verdict, raw = judge_answer(
            self._client, task, answer, evidence=evidence, repair=self._repair
        )
        cost = raw.usage.cost_cny(peak=self._peak)
        calls = 1
        supports: list[bool] = []
        # 逐条断言判蕴含：切分口径复用 Task 30 的 split_claims，
        # 与目标系统 Verifier 用的是同一把尺子（spec §9.2 的"同一口径"）
        for claim in split_claims(answer):
            supported, entail_raw = entailment_support(
                self._client, claim, evidence, repair=self._repair
            )
            supports.append(bool(supported))
            cost += entail_raw.usage.cost_cny(peak=self._peak)
            calls += 1
        return ScoreResult(
            judge_score=float(verdict.score),
            supports=tuple(supports),
            cost_cny=cost,
            calls=calls,
        )


def _collect_evidence(citations: Sequence[str], fetch_evidence: Callable[[str], str]) -> str:
    """把引用到的原文拼成给 judge 的证据块：去重、保序、按上限截断。

    截断上限复用 Task 34 的 `MAX_EVIDENCE_CHARS`，不在这里另定一个数：
    judge 的输入长度同时决定判分质量与花钱速度，这个数只能有一处定义。
    """
    parts: list[str] = []
    seen: set[str] = set()
    for chunk_id in citations:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        text = fetch_evidence(chunk_id) or ""
        if text:
            parts.append(text)
    return "\n---\n".join(parts)[:MAX_EVIDENCE_CHARS]


# 拒答判据：两组共用同一份文本标记（设计 4）。
# 为什么不看目标系统的 REFUSE 状态：只有目标系统有状态机，拿状态当判据等于给一组
# 特殊待遇，正确拒答率与过度拒答率就不可比了
REFUSAL_MARKERS: tuple[str, ...] = (
    "无法确定",
    "无法回答",
    "无法从知识库",
    "无法据此判断",
    "未找到",
    "没有找到",
    "知识库中不存在",
    "不足以回答",
)


def is_refusal_text(text: str) -> bool:
    """答案是不是一次拒答：只看上面那份共享标记，不看状态、不看系统自报。"""
    return any(marker in text for marker in REFUSAL_MARKERS)


class RunRecord(BaseModel):
    """一次（系统 × 任务 × 轮次）运行的完整记录，也是两组系统唯一的产出格式（设计 1）。

    JSONL 一行一条、两组同构（`system` 字段区分），Task 37 只读这个文件出表。
    `extra="forbid"`：多出一个字段就说明有人绕过了格式，宁可报错也不要一张来源不明的表。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    system: str = Field(min_length=1)
    phase: str = Field(min_length=1)              # main / variance
    repeat: int = Field(ge=1)
    task_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    answerable: bool
    answer: str
    citations: tuple[str, ...] = ()
    retrieved_ids: tuple[str, ...] = ()
    # 逐次 `search_documents` 调用各自的 chunk_id。缺省空元组 = 旧 JSONL 回读或
    # 这次运行没走记账网关，此时逐次口径不可测（不会退化成 0.0）。
    retrieved_per_call: tuple[tuple[str, ...], ...] = ()
    gold_ids: tuple[str, ...] = ()
    supports: tuple[bool, ...] = ()
    answered: bool
    judge_score: float | None = None
    latency_ms: float = 0.0
    cost_cny: float = 0.0
    # 成本拆三列：合计给指标层（Task 33 只认 cost_cny），明细给报告层。
    # 为什么拆：预算表要能回答"这张对照表花了多少钱、其中几成花在尺子（judge）上"——
    # 只有一个合计数字，就没法解释"为什么我们只跑了 30 题×2 组"
    system_cost_cny: float = 0.0
    judge_cost_cny: float = 0.0
    # metered：我们自己的 Usage 计量；reported / unmetered：留给外部进程基线的口径。
    # 本轮两组都走 metered，另两个取值是接口保留项，不表示表里出现过这种数
    cost_source: str = "metered"
    steps: int = 0
    illegal_transitions: int = 0
    tool_call_success: float | None = None         # None = 不可测（未接网关），不是"全失败"
    degraded: bool = False
    state: str = ""
    error: str | None = None
    model: str = DEFAULT_MODEL
    started_at: str = ""

    def to_metric_input(self) -> MetricInput:
        """转成指标层的输入：字段一一对应，不做任何"顺手算一下"的事。"""
        return MetricInput(
            task_id=self.task_id,
            type=self.type,
            answerable=self.answerable,
            retrieved_ids=self.retrieved_ids,
            retrieved_per_call=self.retrieved_per_call,
            gold_ids=self.gold_ids,
            supports=self.supports,
            answered=self.answered,
            judge_score=self.judge_score,
            latency_ms=self.latency_ms,
            cost_cny=self.cost_cny,
            steps=self.steps,
            illegal_transitions=self.illegal_transitions,
        )

    def to_jsonl_line(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "RunRecord":
        return cls.model_validate_json(line)


def select_variance_subset(
    tasks: Sequence[EvalTask],
    *,
    size: int = VARIANCE_SUBSET_SIZE,
    seed: int = VARIANCE_SEED,
) -> list[EvalTask]:
    """按题型比例分层抽 `size` 题（spec §9.4），固定种子保证可复现。

    为什么分层：核心集四类题的占比差别很大，纯随机抽 12 题很容易某一类一题都没抽到，
    于是"方差"只覆盖了部分题型，却看起来像是整体结论。分层后每类都有代表，
    再用最大余额法把余数补齐，保证总数严格等于 `size`。
    """
    if size <= 0:
        raise ExperimentError(f"subset size 必须为正整数：{size}")
    if size > len(tasks):
        raise ExperimentError(f"无法抽出 {size} 题：任务集只有 {len(tasks)} 题")
    grouped: dict[str, list[EvalTask]] = {}
    for task in tasks:
        grouped.setdefault(str(task.type), []).append(task)
    quotas = _stratified_quotas({kind: len(pool) for kind, pool in grouped.items()}, size)
    for kind, quota in quotas.items():
        if quota > len(grouped[kind]):
            raise ExperimentError(
                f"无法抽出 {size} 题：{kind} 只有 {len(grouped[kind])} 题，却要抽 {quota} 题"
            )
    rng = random.Random(seed)
    picked: list[EvalTask] = []
    for kind in sorted(grouped):
        # 先按 task_id 排序再打乱：结果只由 seed 与任务集内容决定，
        # 与任务集在文件里的排列顺序无关（否则改一次文件顺序，方差集就换了）
        pool = sorted(grouped[kind], key=lambda item: item.task_id)
        rng.shuffle(pool)
        picked.extend(pool[: quotas[kind]])
    order = {task.task_id: index for index, task in enumerate(tasks)}
    return sorted(picked, key=lambda item: order[item.task_id])


def _stratified_quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    """按各类占比分配名额，余数用最大余额法补给小数部分最大的那几类。"""
    total = sum(counts.values())
    exact = {kind: count * size / total for kind, count in counts.items()}
    quotas = {kind: int(math.floor(value)) for kind, value in exact.items()}
    remainder = size - sum(quotas.values())
    ranked = sorted(
        exact,
        key=lambda kind: (-(exact[kind] - quotas[kind]), kind),
    )
    for kind in ranked[:remainder]:
        quotas[kind] += 1
    return quotas


def build_runners(
    *,
    client: Any,
    gateway: ToolGateway,
    artifacts: Any,
    conn: Any,
    budget: Any = None,
    wall_seconds: float | None = None,
) -> dict[str, Callable[[EvalTask], SystemOutput]]:
    """把两组系统装配成同一个签名：`(EvalTask) -> SystemOutput`。

    这是"两组系统的差异只在编排策略"这句话在代码里的落点：语料、工具、模型、任务集
    全部从参数进来、两组共享；不一样的只有下面两个 `run_*` 的函数体。

    `wall_seconds` 同时喂给两组：目标系统的墙钟在 `Budget.max_wall_seconds`（库里的默认值
    留给 `Budget()` 的其它使用者），基线在自己那一侧管。**只有两边都拿到同一个值**，
    "资源盒子对称"才不是一句写在文档里的话（2026-09-25：`ours` 被 120 秒截断时基线
    根本没有时间闸门，那一批数据因此只能定性看）。
    """

    def run_ours(task: EvalTask) -> SystemOutput:
        ours_budget = budget
        if ours_budget is None and wall_seconds is not None:
            # 库里的字段是 int；CLI 收 float 是为了将来允许小数秒，落进 Budget 时统一取整
            ours_budget = Budget(max_wall_seconds=int(wall_seconds))
        result = run_once(
            question=task.question,
            gateway=gateway,
            client=client,
            conn=conn,
            artifacts=artifacts,
            budget=ours_budget,
        )
        summary = summarize_run(conn, result.run_id)
        return SystemOutput(
            answer=result.answer,
            citations=tuple(citation.chunk_id for citation in result.citations),
            cost_cny=result.cost_cny,
            cost_source="metered",
            steps=result.steps,
            # 非法转移次数是 trace 聚合出来的，不是系统自报的（spec §10）
            illegal_transitions=summary.illegal_transitions,
            latency_ms=0.0,          # 由 runner 的秒表统一测，见 run_system
            degraded=result.degraded,
            state=result.state,
        )

    def run_baseline_a(task: EvalTask) -> SystemOutput:
        result = ReActAgent(
            client,
            gateway,
            max_wall_seconds=(
                wall_seconds if wall_seconds is not None else MAX_WALL_SECONDS
            ),
        ).run(task.question)
        return SystemOutput(
            answer=result.answer,
            citations=tuple(citation.chunk_id for citation in result.citations),
            cost_cny=result.cost_cny,
            cost_source="metered",
            steps=result.steps,
            illegal_transitions=0,   # 没有状态机：结构性为 0，报告里要注明"不适用"
            latency_ms=float(result.latency_ms),
            degraded=result.degraded,
            state="degraded" if result.degraded else "done",
        )

    return {"baseline_a": run_baseline_a, "ours": run_ours}


def artifact_evidence_reader(artifacts: Any) -> Callable[[str], str]:
    """`chunk_id -> 原文正文`；取不到就返回空串。

    与 Task 31 的取证据口径一致：取不到原文就让判分降级，而不是抛异常炸掉整轮实验。
    这里由 chunk_id 反推 doc_id，是因为候选答案可能来自 Baseline A——它的引用
    没经过我们的账本，拿不到 `Citation.doc_id`。
    """

    def fetch(chunk_id: str) -> str:
        doc_id, separator, _ = chunk_id.partition("#")
        if not separator or not doc_id:
            return ""
        try:
            return artifacts.get(doc_id, chunk_id).text
        except ArtifactNotFound:
            return ""

    return fetch


def tool_context_from_env(
    *,
    db_path: str | None = None,
    artifacts_root: str | None = None,
    neo4j_uri: str | None = None,
) -> Any:
    """按环境变量装配 5 个工具的依赖，路径默认值与 `akr.mcp_server.server.main` 完全一致。

    为什么必须有这个函数：`build_tool_context` 的 `chroma_path` / `bm25_path` 默认是
    `None`，而 `None` 不是"用默认路径"，是**不建**向量库与 BM25 —— `search_documents`
    紧接着就会在 `ctx.vectors.query(...)` 上抛
    `AttributeError: 'NoneType' object has no attribute 'query'`。2026-09-25 冒烟实测
    两种系统 4/4 全部倒在这里，所以两个 runner 共用这一个入口，而不是各写一份。

    `AKR_RERANK_USAGE_PATH` 与 `server.main` 读同一个变量：工具层默认档已是
    `rerank="cross"`（2026-09-26 开线上默认），不记账的端到端跑会**花掉 token 但查不到痕迹**。
    与线上装配口径一致，这里也把它透传下去（不设 = 不落账本）。

    重依赖仍然延后到调用时 import：单测 import 本模块不必加载向量库与 neo4j。
    """
    from akr.mcp_server.server import (
        DEFAULT_ARTIFACTS_ROOT,
        DEFAULT_BM25_PATH,
        DEFAULT_CHROMA_PATH,
        DEFAULT_DB_PATH,
        DEFAULT_UNIT_MAP_PATH,
        DEFAULT_VECTOR_FOLD,
        build_tool_context,
    )

    return build_tool_context(
        db_path=db_path or os.environ.get("AKR_DB_PATH", str(DEFAULT_DB_PATH)),
        artifacts_root=artifacts_root
        or os.environ.get("AKR_ARTIFACTS_ROOT", str(DEFAULT_ARTIFACTS_ROOT)),
        chroma_path=os.environ.get("AKR_CHROMA_PATH", str(DEFAULT_CHROMA_PATH)),
        unit_map_path=os.environ.get("AKR_UNIT_MAP_PATH", str(DEFAULT_UNIT_MAP_PATH)),
        bm25_path=os.environ.get("AKR_BM25_PATH", str(DEFAULT_BM25_PATH)),
        neo4j_uri=neo4j_uri if neo4j_uri is not None else os.environ.get("AKR_NEO4J_URI"),
        vector_fold=os.environ.get("AKR_VECTOR_FOLD", DEFAULT_VECTOR_FOLD),
        rerank_usage_path=os.environ.get("AKR_RERANK_USAGE_PATH") or None,
    )


def run_system(
    system: str,
    task: EvalTask,
    *,
    phase: str,
    repeat: int,
    run_task: Callable[[EvalTask], SystemOutput],
    scorer: Scorer,
    fetch_evidence: Callable[[str], str],
    gateway: RecordingGateway | None = None,
    guard: BudgetGuard | None = None,
    judge_guard: BudgetGuard | None = None,
    clock: Callable[[], float] = time.monotonic,
    model: str = DEFAULT_MODEL,
) -> RunRecord:
    """跑一次系统，产出一条 `RunRecord`。

    固定顺序：开窗口（mark）→ 跑系统 → 停表 → 判分 → 记账 → 组装记录。
    这段代码的立场就是三个"不"：

    - **失败不伪造**：任何一种失败都写进 `error`，`judge_score` 留 None（设计 8）；
    - **不可测不填数**：没有网关就没有检索指标，写 None 而不是 0（设计 3）；
    - **记账不吞异常**：工具与判分的异常该抛就抛，记录只是"顺便留下的证据"。

    `guard` 管主实验（系统调用）的钱，`judge_guard` 管 judge 的钱——两本账分开记，
    是因为"系统烧钱"和"尺子烧钱"在预算表里是两件事（spec §7.7）。任一触顶都抛
    `BudgetExceeded` 中断整场实验。
    """
    mark = gateway.mark() if gateway is not None else None
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    errors: list[str] = []
    output: SystemOutput | None = None
    start = clock()
    try:
        output = run_task(task)
    except Exception as exc:
        # 系统崩了：如实记下来，不吞异常、也不在这里重试（重试属于预算范畴，Task 25）
        errors.append(f"{type(exc).__name__}: {exc}")
    measured_ms = (clock() - start) * 1000.0

    retrieved_ids: tuple[str, ...] = ()
    retrieved_per_call: tuple[tuple[str, ...], ...] = ()
    tool_call_success: float | None = None
    if gateway is not None and mark is not None:
        retrieved_ids, successes = gateway.since(mark)
        # 同一窗口的第二份视图：不跨调用去重，逐次调用各留一条（逐次口径用它）；
        # 两份视图来自同一次测量，不存在"两次测量拼起来"的口径裂缝
        retrieved_per_call = gateway.since_calls(mark)
        # 窗口内一次工具调用都没有时，Task 33 的口径返回 0.0；这是"该口径的已知边界"，
        # 由报告注明，而不是在这里偷偷改成 None 又和"不可测"混在一起
        tool_call_success = tool_call_success_rate(successes)

    judge_score: float | None = None
    supports: tuple[bool, ...] = ()
    judge_cost = 0.0
    if output is not None:
        try:
            score = scorer.score(
                task=task,
                answer=output.answer,
                citations=output.citations,
                fetch_evidence=fetch_evidence,
            )
        except Exception as exc:
            # judge 挂了不等于系统答错：分数留空、原因写进 error（设计 8）
            errors.append(f"judge: {type(exc).__name__}: {exc}")
        else:
            judge_score = float(score.judge_score)
            supports = tuple(bool(item) for item in score.supports)
            judge_cost = float(score.cost_cny)

    system_cost = float(output.cost_cny) if output is not None else 0.0
    budget_error: BudgetExceeded | None = None
    if guard is not None:
        try:
            guard.charge(system_cost)
        except BudgetExceeded as exc:
            # 钱已经花掉了：不在这里抛，先把记录组装完再连记录一起抛（见 BudgetExceededWithRecord）
            budget_error = exc
            errors.append(f"budget: {exc}")
    if judge_guard is not None and judge_cost > 0:
        try:
            judge_guard.charge(judge_cost)
        except BudgetExceeded as exc:
            budget_error = budget_error or exc
            errors.append(f"budget: {exc}")

    answer = output.answer if output is not None else ""
    # 拒答判定对两组用同一份文本标记（设计 4）；空答案不算"答了"
    answered = bool(answer.strip()) and not is_refusal_text(answer)
    # 秒表由 runner 统一持有：只有它测出来的延迟，在"自己写循环的 Baseline A"和
    # 目标系统之间才是可比的；系统自报值只在秒表拿不到读数时兜底
    latency_ms = measured_ms
    if latency_ms <= 0 and output is not None:
        latency_ms = float(output.latency_ms)
    record = RunRecord(
        run_id=f"{phase}-{system}-{task.task_id}-r{repeat}",
        system=system,
        phase=phase,
        repeat=repeat,
        task_id=task.task_id,
        type=str(task.type),
        answerable=bool(task.answerable),
        answer=answer,
        citations=tuple(output.citations) if output is not None else (),
        retrieved_ids=retrieved_ids,
        retrieved_per_call=retrieved_per_call,
        gold_ids=tuple(task.gold_chunk_ids),
        supports=supports,
        answered=answered,
        judge_score=judge_score,
        latency_ms=latency_ms,
        cost_cny=system_cost + judge_cost,
        system_cost_cny=system_cost,
        judge_cost_cny=judge_cost,
        cost_source=output.cost_source if output is not None else "metered",
        steps=output.steps if output is not None else 0,
        illegal_transitions=output.illegal_transitions if output is not None else 0,
        tool_call_success=tool_call_success,
        degraded=output.degraded if output is not None else False,
        state=output.state if output is not None else "error",
        error="; ".join(errors) if errors else None,
        model=model,
        started_at=started_at,
    )
    if budget_error is not None:
        # 记录已组装：先把它交回给 execute_phase（进 sink、落盘），再中断整场实验
        raise BudgetExceededWithRecord(str(budget_error), record)
    return record


def save_records(path: str | Path, records: Sequence[RunRecord]) -> Path:
    """把记录**追加**写到 JSONL：一行一条，随时可被 Task 37 直接读。

    为什么是追加而不是覆盖：主实验与方差实验是两次独立运行，方差那一次只覆盖 12 题子集；
    如果第二次是覆盖写，主表的 30 题记录就没了。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_jsonl_line() + "\n")
    return target


def load_records(path: str | Path) -> list[RunRecord]:
    """读回 JSONL：空行跳过，坏行直接抛（pydantic 校验失败），绝不静默丢数。"""
    target = Path(path)
    if not target.exists():
        return []
    records: list[RunRecord] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(RunRecord.from_jsonl_line(line))
    return records


def summarize_records(records: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    """按系统分组汇总：分组在这里做，指标算法全部来自 Task 33（不许就地重算）。"""
    grouped: dict[str, list[MetricInput]] = {}
    costs: dict[str, float] = {}
    for record in records:
        grouped.setdefault(record.system, []).append(record.to_metric_input())
        costs[record.system] = costs.get(record.system, 0.0) + record.cost_cny
    summary: dict[str, dict[str, Any]] = {}
    for system, rows in grouped.items():
        # aggregate 只在"没有结果行"时抛 ValueError，这里的分组天然非空
        entry: dict[str, Any] = dict(aggregate(rows))
        entry["runs"] = len(rows)
        entry["cost_cny_total"] = costs[system]
        summary[system] = entry
    return summary


def execute_phase(
    *,
    phase: str,
    systems: Sequence[str],
    tasks: Sequence[EvalTask],
    repeats: int,
    runners: Mapping[str, Callable[[EvalTask], SystemOutput]],
    scorer: Scorer,
    fetch_evidence: Callable[[str], str],
    gateway: RecordingGateway | None = None,
    guard: BudgetGuard | None = None,
    judge_guard: BudgetGuard | None = None,
    ledger: RunLedger | None = None,
    skip: Collection[tuple[str, str, str, int]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    model: str = DEFAULT_MODEL,
    sink: list[RunRecord] | None = None,
) -> list[RunRecord]:
    """跑完一个阶段（`main` 或 `variance`）的全部运行，返回记录。

    顺序固定为 轮次 → 系统 → 任务：顺序固定，成本归因与 trace 序号才对得上（Task 35 约定）。

    三道闸门在这一层汇合，且**都在发请求之前**判定：

    - `guard`：这一场实验的钱（默认 `MAIN_BUDGET_CNY`，CLI 可用 `--max-cost-cny` 改）；
    - `judge_guard`：judge 这本账（默认 `JUDGE_BUDGET_CNY`，CLI 可用 `--judge-cost-cny` 改）；
    - `ledger`：**整场实验的运行次数**（用户 2026-09-25 定的 60 次，跨进程落盘）。

    三者任一触顶都直接抛（`BudgetExceeded` / `RunCountExceeded`），由调用方中断整场实验——
    "跑完再算总账"在这里等于把超支的那一笔先花出去。`sink` 是给中断准备的：已跑完的记录
    实时追加进去，落盘那一侧才不会只剩一个空文件（与 Task 36 的故障注入 runner 同一套做法）。

    `skip` 是 2026-09-25 主实验中场中断后的**续跑**入口（CLI 的 `--resume`）：键与 `run_id` 同构
    `(phase, system, task_id, repeat)`，命中的运行**既不发请求、也不动次数账本**——次数是硬闸门，
    不能因为"再跑一遍同一个格子"而被重复扣掉。为什么要续跑而不是重跑整场：重跑会把已经落盘的
    记录再追加一遍，同一题在报告里出现两次（`save_records` 是追加写）。
    """
    records: list[RunRecord] = sink if sink is not None else []
    done = skip or frozenset()
    for repeat in range(1, repeats + 1):
        for system in systems:
            for task in tasks:
                if (phase, system, task.task_id, repeat) in done:
                    continue
                if ledger is not None:
                    # 先记账、后调用：中断时账本只会多记，不会少记（宁可少跑，不可超跑）
                    ledger.claim(phase=phase, system=system, task_id=task.task_id)
                try:
                    records.append(
                        run_system(
                            system,
                            task,
                            phase=phase,
                            repeat=repeat,
                            run_task=runners[system],
                            scorer=scorer,
                            fetch_evidence=fetch_evidence,
                            gateway=gateway,
                            guard=guard,
                            judge_guard=judge_guard,
                            clock=clock,
                            model=model,
                        )
                    )
                except BudgetExceededWithRecord as exc:
                    # 触顶的那一笔同样要落盘：钱花了、次数扣了，记录不能丢（否则账目对不上）
                    records.append(exc.record)
                    raise BudgetExceeded(str(exc)) from exc
    return records


def completed_run_keys(records: Sequence[RunRecord]) -> set[tuple[str, str, str, int]]:
    """把已有记录压成续跑用的跳表键 `(phase, system, task_id, repeat)`。

    用 `run_id` 的四个组成部分而不是 `run_id` 字符串：`--resume` 要按"哪个格子跑过"判断，
    而不是按字符串是否相等——将来 `run_id` 的拼法一变，续跑不该跟着失效。
    """
    return {(r.phase, r.system, r.task_id, r.repeat) for r in records}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：跑一场对照实验，把记录写进 JSONL，把汇总打到 stdout。

    刻意不做的事：不生成 markdown 表（Task 37 负责）、不替人下结论、不自动调 rubric
    （spec §9.5 的第 3 条是人的决定）。脚本只产出可信的数字与可复核的原始记录。

    退出码：0 = 跑完；2 = 参数/账本不可用；3 = 钱的闸门触顶；4 = 次数的闸门触顶。
    """
    # 重依赖延后到这里再 import：单测 import 本模块时不必加载向量库 / MCP / neo4j 那一整套
    from akr.agents.tool_gateway import InProcessGateway
    from akr.config import Settings
    from akr.mcp_server.server import DEFAULT_ARTIFACTS_ROOT, DEFAULT_DB_PATH
    from akr.model.client import LLMClient
    from akr.storage import sqlite_store
    from akr.storage.artifact_store import ArtifactStore

    parser = argparse.ArgumentParser(description="两组系统的同口径对照实验 runner")
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET), help="任务集 JSONL 路径")
    parser.add_argument(
        "--phase",
        choices=("main", "variance"),
        default="main",
        help="main：核心集全量跑 1 次；variance：分层抽 12 题跑 3 次",
    )
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="逗号分隔，默认两组全跑")
    parser.add_argument("--out", default=None, help="记录输出路径，默认 eval/runs/<phase>.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（联调用，0 = 全部）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "续跑：读 --out 里已有的记录，跳过已跑过的 (phase, system, task, repeat) 格子，"
            "只跑缺的那些（既不重复发请求，也不重复扣次数）；汇总表按「已有 + 本次」一起算"
        ),
    )
    parser.add_argument("--peak", action="store_true", help="按高峰价计价（默认低峰价，spec §7.7）")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "整场实验的运行次数硬上限（默认沿用账本，账本不存在时 "
            f"{DEFAULT_MAX_RUNS} 次）；显式传值即抬/收上限，会打到 stderr 提醒"
        ),
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER_PATH),
        help="运行次数账本路径（默认 data/run-ledger.json；与故障注入 runner 共用一本账）",
    )
    parser.add_argument(
        "--max-cost-cny",
        type=float,
        default=MAIN_BUDGET_CNY,
        help=(
            f"系统侧的钱闸门（默认 ¥{MAIN_BUDGET_CNY}，spec §7.7）；"
            "显式传值即改闸，会打到 stderr 提醒"
        ),
    )
    parser.add_argument(
        "--judge-cost-cny",
        type=float,
        default=JUDGE_BUDGET_CNY,
        help=(
            f"judge 这本账的钱闸门（默认 ¥{JUDGE_BUDGET_CNY}，spec §7.7）；"
            "显式传值即改闸，会打到 stderr 提醒"
        ),
    )
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=MAIN_WALL_SECONDS,
        help=(
            f"单次运行的墙钟闸门（默认 {MAIN_WALL_SECONDS:.0f} 秒，与 "
            "`Budget.max_wall_seconds` 同值），**两组同值生效**；"
            "显式传值即改闸，会打到 stderr 提醒"
        ),
    )
    args = parser.parse_args(argv)

    # 次数账本先于一切重依赖检查：账本不可用、或额度已经用完时，连工具上下文都不该去搭
    # （快速失败省掉的是一次 Neo4j / 向量库连接，也让"没额度了"这件事不必跑半场才知道）
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

    try:
        max_cost_cny, judge_cost_cny = resolve_budget_limits(
            max_cost_cny=args.max_cost_cny,
            judge_cost_cny=args.judge_cost_cny,
        )
    except ExperimentError as exc:
        parser.error(str(exc))
    if (
        abs(max_cost_cny - MAIN_BUDGET_CNY) > BUDGET_EPSILON
        or abs(judge_cost_cny - JUDGE_BUDGET_CNY) > BUDGET_EPSILON
    ):
        print(
            f"[warn] 钱闸门被显式改为 系统侧 ¥{max_cost_cny:.2f} / judge ¥{judge_cost_cny:.2f}"
            f"（默认 ¥{MAIN_BUDGET_CNY} / ¥{JUDGE_BUDGET_CNY}）——"
            f"这是看得见的一次改闸，不是默认值",
            file=sys.stderr,
        )

    # 墙钟与钱一样：非正数 / NaN 在这里就以退出码 2 结束，且**在开跑之前**。
    # 0 绝不能被解释成"不设闸门"——`BudgetState.elapsed() >= 0` 恒真，等于每一步都触顶。
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error(f"--wall-seconds 必须是正数：{args.wall_seconds}")
    if abs(args.wall_seconds - MAIN_WALL_SECONDS) > BUDGET_EPSILON:
        print(
            f"[warn] 墙钟闸门被显式改为 {args.wall_seconds:.0f} 秒"
            f"（默认 {MAIN_WALL_SECONDS:.0f} 秒）——两组同值生效，"
            f"这是看得见的一次改闸，不是默认值",
            file=sys.stderr,
        )

    settings = Settings.from_env()
    db_path = os.environ.get("AKR_DB_PATH", str(DEFAULT_DB_PATH))
    artifacts_root = os.environ.get("AKR_ARTIFACTS_ROOT", str(DEFAULT_ARTIFACTS_ROOT))
    conn = sqlite_store.connect(db_path)
    sqlite_store.init_schema(conn)
    artifacts = ArtifactStore(artifacts_root)
    # 工具本身经 MCP 暴露，但实验里用同进程网关直调：少一层进程通信，
    # 成本与延迟归因都更干净（进程隔离的验收放在 Task 36 的故障注入里做）
    context = tool_context_from_env(
        db_path=db_path,
        artifacts_root=artifacts_root,
        neo4j_uri=os.environ.get("AKR_NEO4J_URI"),
    )
    gateway = RecordingGateway(InProcessGateway(context))
    client = LLMClient(settings)
    runners = build_runners(
        client=client,
        gateway=gateway,
        artifacts=artifacts,
        conn=conn,
        wall_seconds=args.wall_seconds,
    )
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    unknown = sorted(set(systems) - set(runners))
    if unknown:
        parser.error(f"未知系统：{unknown}；可选：{sorted(runners)}")

    tasks = load_taskset(args.taskset)
    selected = tasks if args.phase == "main" else select_variance_subset(tasks)
    if args.limit:
        selected = selected[: args.limit]
    repeats = MAIN_REPEATS if args.phase == "main" else VARIANCE_REPEATS

    scorer = JudgeScorer(client, peak=args.peak)
    fetch_evidence = artifact_evidence_reader(artifacts)
    guard = BudgetGuard(max_cost_cny)             # 主实验一本账（默认 spec §7.7 的 ¥2.5）
    judge_guard = BudgetGuard(judge_cost_cny)     # judge 单独一本账（默认 spec §7.7 的 ¥1.0）
    out_path = Path(args.out) if args.out else RUNS_DIR / f"{args.phase}.jsonl"

    # 续跑：跳表只认 --out 里已经落盘的格子。不重复发请求、不重复扣次数，
    # 但落盘仍是追加写（旧记录留在原地），所以汇总时要"已有 + 本次"一起算。
    existing: list[RunRecord] = []
    skip: set[tuple[str, str, str, int]] = set()
    if args.resume:
        existing = load_records(out_path)
        skip = completed_run_keys(existing)
        print(
            f"[resume] {out_path} 已有 {len(existing)} 条记录，"
            f"本次跳过 {len(skip)} 个格子（账本已用 {ledger.used}/{ledger.max_runs} 次）",
            file=sys.stderr,
        )

    records: list[RunRecord] = []
    exit_code = 0
    try:
        execute_phase(
            phase=args.phase,
            systems=systems,
            tasks=selected,
            repeats=repeats,
            runners=runners,
            scorer=scorer,
            fetch_evidence=fetch_evidence,
            gateway=gateway,
            guard=guard,
            judge_guard=judge_guard,
            ledger=ledger,
            skip=skip,
            model=settings.model,
            sink=records,       # 触顶中断时，已跑完的记录不会跟着异常一起消失
        )
    except BudgetExceeded as exc:
        # 触顶即停：已经跑到的记录照样落盘。半张表也是证据，丢掉它才是最坏的结果
        exit_code = 3
        print(json.dumps({"stopped": str(exc)}, ensure_ascii=False))
    except RunCountExceeded as exc:
        # 次数闸门与钱闸门分开报：一个说"钱花完了"，一个说"次数用完了"，
        # 接下来该做的事不同（前者等低峰或追加额度，后者抬上限或换一本新账）
        exit_code = 4
        print(json.dumps({"stopped": str(exc)}, ensure_ascii=False))
    finally:
        save_records(out_path, records)
        gateway.close()

    print(json.dumps({"run_ledger": ledger.state.to_json_dict()}, ensure_ascii=False))
    # 汇总口径：续跑时"已有 + 本次"一起算，否则印出来的表只覆盖本次跑的这一小截
    print(json.dumps(summarize_records([*existing, *records]), ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":  # pragma: no cover - 仅作为 CLI 入口
    raise SystemExit(main())
