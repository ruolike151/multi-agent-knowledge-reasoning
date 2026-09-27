"""端到端执行器：一次问答 = 一次状态机推进。

三个不变量在这里变成代码事实：
1. 控制流只能来自转移表（每次转移都过 `check_transition`）；
2. 每个节点幂等（`(run_id, task_id, node)` 落 SQLite，命中即短路）；
3. 每次模型调用同时计入预算与 trace（预算上限因此可验证）。
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from akr.agents.errors import ToolResultError
from akr.agents.graph_walker import run_graph_walk
from akr.agents.retriever import run_retrieve
from akr.agents.tool_gateway import ToolGateway
from akr.model.client import LLMClient, LLMResult
from akr.observability.trace import TraceRecorder, init_trace_schema, summarize_run
from akr.orchestration.budget import Budget, BudgetState
from akr.orchestration.ledger import (
    Citation,
    SubTask,
    TaskLedger,
    WorkerReport,
    init_ledger_schema,
    load_ledger,
    load_node_result,
    load_worker_report,
    record_node_result,
    save_ledger,
    save_worker_report,
)
from akr.orchestration.states import (
    INITIAL_STATE,
    TERMINAL_STATES,
    InvalidTransition,
    State,
    UnknownStateName,
    check_all_transitions,
    check_transition,
    state_from_name,
)
from akr.orchestration.supervisor import (
    DraftAnswer,
    PlanOutcome,
    build_subtasks,
    collect_citations,
    decide_next,
    is_retrieval_scope_note,
    plan_question,
    synthesize,
)
from akr.orchestration.verifier import VerifyResult, should_revise, verify_answer
from akr.storage.artifact_store import ArtifactNotFound, ArtifactStore

PLAN_NODE = "plan"
ROUTE_NODE = "route"
WORKER_NODE = "worker"
VERIFY_NODE = "verify"
SYNTHESIZE_NODE = "synthesize"
RUN_ID_BYTES = 6

DEGRADED_NOTICE = "预算已耗尽，未能完成校验，以下内容仅供参考。"
# 校验没过（`VERIFYING → DEGRADED`，路由器自己判的收敛边）时的收尾声明。与预算触顶
# 必须用两句话：2026-09-25 主实验里 `ours` 30 条有 21 条最终答案是降级草稿，其中 7 条
# 根本没有预算事件，却也被标成「预算已耗尽」——把归因说错，等于给判分器和读者喂假信息。
UNVERIFIED_NOTICE = "校验未通过，以下内容未经核实，仅供参考。"
REFUSE_NOTICE = "现有语料无法回答该问题。"


# 检索策略（2026-09-25 定案，证据在内部偏差日志第三十一节）：
#   * 查询文本用**原问句**，不用规划器改写过的子任务目标；
#   * 工具层开「文档内重搜」，第二段的过滤文档数 = 3。
# 只读 A/B（`scripts/retrieval_ab.py`，25 条可答题、零 DeepSeek 花费）实测进前 8 的
# gold chunk 条数：子任务目标 6/25 → 原问句 9/25 → 原问句 + 前 3 个文档内重搜 **10/25**。
# 代价是每轮多一次嵌入 + 一次 BM25（本机毫秒级），**不增加任何模型调用**。
TWO_STAGE_DOCS = 3

# 邻块扩窗的线上取值。**默认 0 = 关闭 = 历史行为**：Task 5 只把旋钮接到编排层，
# 开不开由 Task 8 的「待拍板清单」定（先看 `data/retrieval-tool-smoke.json` →
# `diffs` 是否为 0 处，再看扩窗那一档的命中题数，才决定改这个数）。
# 池深（`POOL_DEPTH`）与重排（`RERANK`）**不从这里传**：它们已经按工具层默认值
# （`tools.POOL_DEPTH_DEFAULT` / `tools.RERANK_DEFAULT` / `tools.RRF_K_DEFAULT` /
# `tools.RERANK_LOCK_DEFAULT`）生效，网关刻意不传这几个形参——工具层默认值就是线上口径，
# 在这里再复制一份常量只会在两处之间产生漂移。见内部偏差日志第四十三、四十四节。
EXPAND_NEIGHBORS = 0

def _retrieve(subtask: SubTask, *, gateway: ToolGateway, client: LLMClient | None,
              mode: str = "non-thinking", question: str | None = None) -> WorkerReport:
    # Retriever 只接受 kind="retrieve"（Task 27 的入参硬校验）：纠错环的子任务带
    # kind="verify"，这里换成同内容的 retrieve 副本再复用同一个 worker，
    # 而不是放宽 worker 的入参校验去容纳编排层的标记。
    #
    # `question` 是原问句（编排层从账本上取）。只有首轮检索才换成原问句：纠错环的
    # 子任务目标里带着 Verifier 的反馈，那正是它要查的东西，换掉等于让补证据白跑。
    query = question if (question and subtask.kind == "retrieve") else None
    if subtask.kind != "retrieve":
        subtask = SubTask(
            task_id=subtask.task_id,
            goal=subtask.goal,
            kind="retrieve",
            depends_on=list(subtask.depends_on),
        )
    return run_retrieve(
        subtask,
        gateway=gateway,
        client=client,
        mode="non-thinking",
        query=query,
        two_stage_docs=TWO_STAGE_DOCS,
        expand_neighbors=EXPAND_NEIGHBORS,
    )


def _graph(subtask: SubTask, *, gateway: ToolGateway,
           client: LLMClient | None) -> WorkerReport:
    return run_graph_walk(subtask, gateway=gateway, client=client)


# 派发表的值统一是 (subtask, *, gateway, client) 形状，避免在循环里写 if/elif。
DISPATCHERS: Mapping[str, Callable[..., WorkerReport]] = {
    "retrieve": _retrieve,
    "graph": _graph,
    # 纠错环补证据 = 带着 Verifier 反馈再检索一次，因此复用同一个 worker
    "verify": _retrieve,
}

# 需要「原问句」这个额外入参的子任务类型。检索用它当查询文本（理由见 `_retrieve`），
# 图步走自己的实体解析、不接这个参数——所以这里是一个显式清单，而不是给所有
# 派发器都塞一个用不上的形参。
QUERY_AWARE_KINDS: frozenset[str] = frozenset({"retrieve"})

# 预算触顶时每个状态的收敛目标：必须是转移表里真实存在的边。
BUDGET_EXIT: dict[State, State] = {
    State.PLANNING: State.SYNTHESIZING,
    State.DISPATCHING: State.WORKING,
    State.WORKING: State.VERIFYING,
    State.REVISE: State.WORKING,
    State.VERIFYING: State.DEGRADED,
    State.SYNTHESIZING: State.DONE,
    State.REFUSE: State.DONE,
    State.DEGRADED: State.DONE,
}

# 收尾状态本身不再花模型调用，预算触顶时不能由收敛边把它们的正常分支抢走：
# `SYNTHESIZING` / `DEGRADED` 要用 `_degraded_draft` 把已有证据摊开并写明未经校验，
# `REFUSE` 要写 `REFUSE_NOTICE`。若在这里直接走 `DEGRADED → DONE`，最终答案就是空的。
FINALIZING_STATES: frozenset[State] = frozenset(
    {State.SYNTHESIZING, State.DEGRADED, State.REFUSE}
)


class RunResult(BaseModel):
    """一次问答的最终结果；`cost_cny` / `steps` 一律来自 trace 聚合，不是循环里的局部变量。"""

    run_id: str
    question: str
    state: str
    answer: str
    citations: list[Citation] = []
    cost_cny: float = 0.0
    steps: int = 0
    degraded: bool = False
    trace_path: str | None = None


def _advance(
    tracer: TraceRecorder,
    conn: Any,
    ledger: TaskLedger,
    budget_state: BudgetState,
    run_id: str,
    current: State,
    target: State,
    *,
    journal: _ReplayJournal | None = None,
) -> State:
    """唯一的状态推进出口：先过转移表，再落账本，最后写 trace。

    `journal` 只在续跑时传：重放已经记录过的转移不再重复写 trace，`steps` 因此不会翻倍。
    """
    check_transition(current, target)  # 非法转移在这里就炸，不允许任何绕过表的捷径
    ledger.budget = budget_state
    save_ledger(conn, ledger, run_id=run_id)  # 每步落库：进程挂掉后 load_ledger 能接着跑
    if journal is None or not journal.replayed(target):
        tracer.record("state", target.value, prev_state=current.value, state=target.value)
    return target


def _charge_llm(
    tracer: TraceRecorder,
    budget_state: BudgetState,
    *,
    node: str,
    result: LLMResult,
) -> None:
    """每次真实模型调用的唯一记账点：预算与 trace 必须同时更新，否则 ¥5 上限是假的。"""
    usage = result.usage
    cost = usage.cost_cny(peak=True)  # 按高峰价估，宁可高估也不让预算被穿透
    budget_state.charge(llm_calls=1, tokens=usage.total_tokens, cost_cny=cost)
    tracer.record(
        "llm",
        node,
        mode=result.mode,
        model="deepseek-flash",
        tokens_in=usage.miss_tokens,
        tokens_cached=usage.hit_tokens,
        tokens_out=usage.output_tokens,
        cost_cny=cost,
    )


def _checkpoint(
    conn: Any, tracer: TraceRecorder, *, run_id: str, task_id: str, node: str
) -> dict[str, Any] | None:
    """幂等键 `(run_id, task_id, node)`：命中就短路，并留一条 checkpoint_hit 供评测核对。"""
    cached = load_node_result(conn, run_id=run_id, task_id=task_id, node=node)
    if cached is not None:
        tracer.record("note", "checkpoint_hit", detail=f"{node}:{task_id}")
    return cached


def _route_key(current: State, ledger: TaskLedger) -> str:
    """路由决策的幂等键按「离开哪个状态 + 第几轮纠错」定位，恢复时能算出同一个键。"""
    return f"{ROUTE_NODE}-{current.value.lower()}-r{ledger.revise_round}"


def _dispatch_subtask(
    tracer: TraceRecorder,
    conn: Any,
    ledger: TaskLedger,
    budget_state: BudgetState,
    artifacts: ArtifactStore,
    *,
    run_id: str,
    task_id: str,
    gateway: ToolGateway,
    client: LLMClient | None,
) -> WorkerReport | None:
    """派发一个子任务；工具失败不炸整个 run，只把这条子任务标 failed 并记 trace。"""
    cached = _checkpoint(conn, tracer, run_id=run_id, task_id=task_id, node=WORKER_NODE)
    if cached is not None:
        ref = cached.get("result_ref", "")
        ledger.mark(task_id, "done", result_ref=ref)
        return load_worker_report(artifacts, ref)

    subtask = ledger.subtasks[task_id]
    ledger.mark(task_id, "running")
    dispatcher = DISPATCHERS.get(subtask.kind)
    if dispatcher is None:
        # 未注册的 kind 是编程错误：直接抛，绝不静默跳过
        raise ValueError(f"未注册的子任务类型：{subtask.kind!r}")
    extra: dict[str, Any] = {}
    if subtask.kind in QUERY_AWARE_KINDS:
        extra["question"] = ledger.question
    try:
        report = dispatcher(subtask, gateway=gateway, client=client, **extra)
    except ToolResultError as exc:
        ledger.mark(task_id, "failed", bump_attempt=True)
        tracer.record("error", "tool_failed", status="error", detail=f"{task_id}: {exc}")
        return None

    ref = save_worker_report(artifacts, run_id=run_id, task_id=task_id, report=report)
    record_node_result(
        conn, run_id=run_id, task_id=task_id, node=WORKER_NODE, payload={"result_ref": ref}
    )
    ledger.mark(task_id, "done", result_ref=ref)
    budget_state.charge(
        steps=report.cost.steps, tokens=report.cost.tokens, cost_cny=report.cost.cost_cny
    )
    # token 明细按列落账（spec §7.5 的缓存命中率只能从这三列算）：
    # `tokens_in` = 未命中前缀缓存的输入、`tokens_cached` = 命中、`tokens_out` = 输出。
    # 2026-09-25 之前这里把**总量**写进 `tokens_out`、另外两列留 0，worker 的调用因此
    # 既丢了输入明细、又把输入算成了输出（偏差日志第三十二节）。
    hit, miss, output = (
        report.cost.hit_tokens,
        report.cost.miss_tokens,
        report.cost.output_tokens,
    )
    if hit + miss + output != report.cost.tokens:
        # 明细缺失只可能来自旧工件（成本记录里还没有三列明细，`from_dict` 补 0）。
        # 这种情况下总量不能丢，但也不许假装它是输出——写进 `tokens_in` 并留一条 note，
        # 让读到异常分布的人能查到原因，而不是对着一个编出来的明细做结论。
        tracer.record(
            "note",
            "token_split_missing",
            detail=f"{task_id}: tokens={report.cost.tokens}（旧工件无明细，按未命中输入列记）",
        )
        miss = max(report.cost.tokens - hit - output, 0)
    tracer.record(
        "llm",
        f"{WORKER_NODE}:{subtask.kind}",
        mode="non-thinking",
        model="deepseek-flash",
        tokens_in=miss,
        tokens_cached=hit,
        tokens_out=output,
        cost_cny=report.cost.cost_cny,
        duration_ms=report.cost.latency_ms,
        detail=task_id,
    )
    return report


def _provisional_draft(
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    drop_negative_findings: bool = False,
) -> DraftAnswer:
    """待校验草稿：直接把 Worker 的 findings 拼起来，不花一次模型调用。

    校验要回答的是「证据支不支撑结论」，不是「文笔好不好」，所以草稿可以是素的。

    `drop_negative_findings=True` 时把「检索范围说明」剔出校验输入：这类句子被
    `split_claims` 切成断言后必然判 unsupported，必然白烧一轮 REVISE + worker:verify
    （这是 verify 均值 2.47 次/题的主因）。只影响校验输入——账本、合成上下文与
    降级答案里的原话一字不改，所以 `eval/metrics.py` 的历史口径不受影响。
    全是否定式时保留原样：把非空草稿滤成空会被读成「一条证据都没取到」而错误降级。
    """

    findings: list[str] = []
    for report in reports.values():
        findings.extend(report.findings)
    if drop_negative_findings:
        kept = [finding for finding in findings if not is_retrieval_scope_note(finding)]
        if kept:
            findings = kept
    citations = collect_citations(ledger, reports)
    return DraftAnswer(text="\n".join(findings), chunk_ids=[c.chunk_id for c in citations])


def _degraded_draft(
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    notice: str = DEGRADED_NOTICE,
) -> DraftAnswer:
    """降级收尾：不再调用模型，只把已有证据摊开并标明未经校验。

    `notice` 是「为什么降级」的唯一出口：预算触顶传 `DEGRADED_NOTICE`，校验没过传
    `UNVERIFIED_NOTICE`。两种降级都不花钱、都不改证据，区别只在给读者/判分器的归因。
    """

    provisional = _provisional_draft(ledger, reports)
    text = f"{notice}\n{provisional.text}".strip()
    return DraftAnswer(text=text, chunk_ids=provisional.chunk_ids, dropped_chunk_ids=[])


def _evidence_reader(
    artifacts: ArtifactStore, citations: list[Citation]
) -> Callable[[str], str]:
    """把 chunk_id 映射回 artifact store 里的原文正文。

    取不到就返回空串，交给 Verifier 的兜底逻辑把该断言降级——比抛异常中断整轮校验好。
    """
    doc_of = {citation.chunk_id: citation.doc_id for citation in citations}

    def fetch(chunk_id: str) -> str:
        doc_id = doc_of.get(chunk_id)
        if doc_id is None:
            return ""
        try:
            return artifacts.get(doc_id, chunk_id).text
        except ArtifactNotFound:
            return ""

    return fetch


class _ReplayJournal:
    """续跑重放时的「既有状态轨迹」游标。

    续跑一律从 `INITIAL_STATE` 重放整个循环：幂等键让所有已完成的节点命中检查点短路，
    所以重放不产生任何真实调用，却让「plan / route / worker 都命中了检查点」成为可观测事实，
    而不用去猜「上次挂在哪一步」。重放出来的转移在 trace 里已经有记录，这里按顺序把它们消费掉，
    于是同一段路径不会被写两遍（`steps` 不会在恢复后翻倍，状态序列也仍然逐边合法）。
    一旦重放与历史分叉（换了模型、换了语料导致规划不同），就丢掉游标，之后的转移照常记录。
    """

    __slots__ = ("_pending",)

    def __init__(self, tracer: TraceRecorder, run_id: str) -> None:
        self._pending = [
            event.state for event in tracer.events(run_id) if event.kind == "state"
        ]

    def replayed(self, target: State) -> bool:
        """返回 True 表示这条转移与历史逐条对齐，trace 里已经有它，不必再写。"""

        if self._pending and self._pending[0] == target.value:
            self._pending.pop(0)
            return True
        self._pending.clear()
        return False


def _load_reports(
    artifacts: ArtifactStore, ledger: TaskLedger
) -> dict[str, WorkerReport]:
    """账本里存的是 result_ref 引用，正文在 artifact store；恢复时按引用读回来。"""
    reports: dict[str, WorkerReport] = {}
    for task_id, subtask in ledger.subtasks.items():
        if subtask.status != "done" or not subtask.result_ref:
            continue
        reports[task_id] = load_worker_report(artifacts, subtask.result_ref)
    return reports


def _load_draft(conn: Any, run_id: str) -> DraftAnswer | None:
    cached = load_node_result(
        conn, run_id=run_id, task_id=SYNTHESIZE_NODE, node=SYNTHESIZE_NODE
    )
    return DraftAnswer.model_validate(cached) if cached is not None else None


def _route(
    tracer: TraceRecorder,
    conn: Any,
    ledger: TaskLedger,
    budget_state: BudgetState,
    reports: Mapping[str, WorkerReport],
    *,
    run_id: str,
    current: State,
    client: LLMClient,
    verifier_feedback: str | None = None,
) -> State:
    """问一次路由表「下一步去哪」；答案必须落在 legal_targets 里（由 Task 29 保证）。"""
    key = _route_key(current, ledger)
    cached = _checkpoint(conn, tracer, run_id=run_id, task_id=key, node=ROUTE_NODE)
    if cached is not None:
        return state_from_name(cached["next_state"])

    decision, results = decide_next(
        client, ledger, reports, current=current, verifier_feedback=verifier_feedback
    )
    for result in results:
        _charge_llm(tracer, budget_state, node=ROUTE_NODE, result=result)
    record_node_result(
        conn, run_id=run_id, task_id=key, node=ROUTE_NODE, payload=decision.model_dump()
    )
    return state_from_name(decision.next_state)


def run_once(
    *,
    question: str,
    gateway: ToolGateway,
    client: LLMClient,
    conn: Any,
    artifacts: ArtifactStore,
    budget: Budget | None = None,
    jsonl_path: str | Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    run_id: str | None = None,
) -> RunResult:
    """跑完一次问答。

    传 `run_id` 且该 run 已有账本 = 续跑：从 `INITIAL_STATE` 重放整个循环，已完成的节点
    全部命中检查点短路，因此既不重复花钱，也不需要猜「上次挂在哪一步」。
    """
    check_all_transitions()  # 启动自检：转移表不良构的话，下面所有结论都不成立
    init_ledger_schema(conn)
    init_trace_schema(conn)

    active_run_id = run_id or secrets.token_hex(RUN_ID_BYTES)
    tracer = TraceRecorder(conn, run_id=active_run_id, jsonl_path=jsonl_path)

    resumed = load_ledger(conn, active_run_id, clock=clock) if run_id else None
    if resumed is None:
        budget_state = BudgetState(budget=budget, clock=clock)
        ledger = TaskLedger(question=question, budget=budget_state)
        journal = None
    else:
        ledger = resumed
        budget_state = ledger.budget  # 已用量从账本恢复，续跑不会把预算重置回零
        journal = _ReplayJournal(tracer, active_run_id)

    state = INITIAL_STATE
    reports = _load_reports(artifacts, ledger)
    draft = _load_draft(conn, active_run_id) or DraftAnswer()
    degraded = False
    budget_exit = False
    verifier_feedback: str | None = None

    try:
        while state not in TERMINAL_STATES:
            exhausted = budget_state.exhausted()
            if exhausted is not None:
                # 预算触顶只能走表里存在的收敛边：不存在「直接跳 DONE」这条路
                if not budget_exit:
                    tracer.record(
                        "error", "budget_exhausted", status="error", detail=f"维度 {exhausted}"
                    )
                    budget_exit = True
                if state is State.VERIFYING or BUDGET_EXIT[state] is State.DEGRADED:
                    degraded = True
                # 收尾状态仍要用 budget_exit 标记（这样 SYNTHESIZING 不会再调模型），
                # 但推进到 DONE 的动作交给它们的正常分支去做。
                if state not in FINALIZING_STATES:
                    state = _advance(
                        tracer, conn, ledger, budget_state, active_run_id, state,
                        BUDGET_EXIT[state], journal=journal,
                    )
                    continue

            if state is State.PLANNING:
                cached = _checkpoint(
                    conn, tracer, run_id=active_run_id, task_id=PLAN_NODE, node=PLAN_NODE
                )
                if cached is not None:
                    outcome = PlanOutcome.model_validate(cached)
                else:
                    outcome, plan_result = plan_question(
                        client, question, feedback=verifier_feedback
                    )
                    _charge_llm(tracer, budget_state, node=PLAN_NODE, result=plan_result)
                    record_node_result(
                        conn, run_id=active_run_id, task_id=PLAN_NODE, node=PLAN_NODE,
                        payload=outcome.model_dump(),
                    )
                if not ledger.subtasks:  # 续跑时账本里已有子任务，绝不能重复添加
                    for subtask in build_subtasks(outcome):
                        ledger.add(subtask)
                try:
                    target = state_from_name(outcome.next_state)
                    check_transition(state, target)
                except (InvalidTransition, UnknownStateName) as exc:
                    # 规划给出的目标不合法：记一条 rejected（评测要统计这个数），退回最稳的一条路
                    tracer.record(
                        "state", "rejected", status="rejected", state=outcome.next_state,
                        detail=f"plan 给出非法目标 {outcome.next_state!r}：{exc}",
                    )
                    target = State.DISPATCHING
                state = _advance(
                    tracer, conn, ledger, budget_state, active_run_id, state, target,
                    journal=journal,
                )
                continue

            if state is State.DISPATCHING:
                # DISPATCHING 只算「这一轮要派给谁」，真正的执行留给 WORKING
                tracer.record(
                    "note", "dispatch_plan",
                    detail=",".join(ledger.unfinished_ids()) or "无待派发子任务",
                )
                state = _advance(
                    tracer, conn, ledger, budget_state, active_run_id, state, State.WORKING,
                    journal=journal,
                )
                continue

            if state is State.WORKING:
                # 先路由再派发：路由回答的是「这一轮做完之后往哪走」，它的输入是账本与已有
                # 报告，派发只负责补齐缺的子任务，两者顺序不影响收敛（无子任务可派时有闸）。
                target = _route(
                    tracer, conn, ledger, budget_state, reports,
                    run_id=active_run_id, current=state, client=client,
                )
                # 账本里每个子任务都过一遍派发器：已完成的靠幂等键命中检查点短路，
                # 续跑因此既不重复花钱，也能在 trace 里留下 worker 的 checkpoint_hit。
                for task_id, subtask in list(ledger.subtasks.items()):
                    if subtask.status == "failed":
                        # 工具失败是本次运行的终态：不重试，让证据缺失把草稿推向 DEGRADED
                        continue
                    report = _dispatch_subtask(
                        tracer, conn, ledger, budget_state, artifacts,
                        run_id=active_run_id, task_id=task_id, gateway=gateway, client=client,
                    )
                    if report is not None:
                        reports[task_id] = report
                if target is State.WORKING and not ledger.unfinished_ids():
                    # 路由想继续干活却没有可派的子任务：这里必须拦住，否则空转烧钱
                    tracer.record("note", "route_noop_guard", detail="无待派子任务，直接进校验")
                    target = State.VERIFYING
                state = _advance(
                    tracer, conn, ledger, budget_state, active_run_id, state, target,
                    journal=journal,
                )
                continue

            if state is State.VERIFYING:
                if ledger.unfinished_ids():
                    # 还有没做完的子任务：先回 WORKING 补完，校验不能拿残缺证据下结论
                    target = State.WORKING
                else:
                    provisional = _provisional_draft(
                        ledger, reports, drop_negative_findings=True
                    )
                    if not provisional.text.strip():
                        tracer.record(
                            "error", "no_evidence", status="error", detail="没有 findings 可校验"
                        )
                        degraded = True
                        target = State.DEGRADED
                    else:
                        key = f"{VERIFY_NODE}-r{ledger.revise_round}"
                        cached = _checkpoint(
                            conn, tracer, run_id=active_run_id, task_id=key, node=VERIFY_NODE
                        )
                        if cached is not None:
                            verify_result = VerifyResult.model_validate(cached)
                        else:
                            verify_result, verify_llm = verify_answer(
                                client,
                                provisional,
                                fetch_evidence=_evidence_reader(
                                    artifacts, collect_citations(ledger, reports)
                                ),
                            )
                            _charge_llm(tracer, budget_state, node=VERIFY_NODE, result=verify_llm)
                            record_node_result(
                                conn, run_id=active_run_id, task_id=key, node=VERIFY_NODE,
                                payload=verify_result.model_dump(),
                            )
                        if should_revise(verify_result, revise_round=ledger.revise_round):
                            # 代码级硬规则：只要存在没被原文支撑的断言就先修，不把决定权交给路由
                            verifier_feedback = verify_result.feedback
                            target = State.REVISE
                        else:
                            target = _route(
                                tracer, conn, ledger, budget_state, reports,
                                run_id=active_run_id, current=state, client=client,
                                verifier_feedback=verify_result.feedback,
                            )
                state = _advance(
                    tracer, conn, ledger, budget_state, active_run_id, state, target,
                    journal=journal,
                )
                continue

            if state is State.REVISE:
                feedback = verifier_feedback or "Verifier 判定存在无支撑断言"
                ledger.revise_round += 1
                task_id = f"t-revise-{ledger.revise_round}"
                ledger.add(
                    SubTask(
                        task_id=task_id,
                        goal=f"补证据（第 {ledger.revise_round} 轮）：{feedback}",
                        kind="verify",
                    )
                )
                tracer.record("note", "revise_round", detail=f"round={ledger.revise_round}")
                # 补证据子任务当场派发：它的报告就是下一轮路由决策的输入，不派发就没得判。
                # 反馈因此不只是 prompt 里的一句话，而是有 task_id、有结果、有成本的账本条目。
                report = _dispatch_subtask(
                    tracer, conn, ledger, budget_state, artifacts,
                    run_id=active_run_id, task_id=task_id, gateway=gateway, client=client,
                )
                if report is not None:
                    reports[task_id] = report
                state = _advance(
                    tracer, conn, ledger, budget_state, active_run_id, state, State.WORKING,
                    journal=journal,
                )
                continue

            if state is State.SYNTHESIZING:
                cached = _checkpoint(
                    conn, tracer, run_id=active_run_id, task_id=SYNTHESIZE_NODE,
                    node=SYNTHESIZE_NODE,
                )
                if cached is not None:
                    draft = DraftAnswer.model_validate(cached)
                elif budget_exit:
                    # 预算已触顶：不再调用模型，用已有证据收尾并标明未经校验
                    draft = _degraded_draft(ledger, reports)
                    degraded = True
                else:
                    draft, synth_result = synthesize(
                        client, question, ledger, reports, degraded=False
                    )
                    _charge_llm(tracer, budget_state, node=SYNTHESIZE_NODE, result=synth_result)
                    record_node_result(
                        conn, run_id=active_run_id, task_id=SYNTHESIZE_NODE,
                        node=SYNTHESIZE_NODE, payload=draft.model_dump(),
                    )
                state = _advance(
                    tracer, conn, ledger, budget_state, active_run_id, state, State.DONE,
                    journal=journal,
                )
                continue

            # REFUSE / DEGRADED 都在这里直接收尾
            if not draft.text:
                if state is State.REFUSE:
                    draft = DraftAnswer(text=REFUSE_NOTICE)
                else:
                    # 走到 DEGRADED 就是降级，必须在结果里标出来：`degraded` 是报告读的
                    # 「这条答案未经校验」，只有预算触顶那条路标过它，等于漏报。
                    degraded = True
                    draft = _degraded_draft(
                        ledger,
                        reports,
                        notice=(
                            DEGRADED_NOTICE if budget_exit else UNVERIFIED_NOTICE
                        ),
                    )
            state = _advance(
                tracer, conn, ledger, budget_state, active_run_id, state, State.DONE,
                journal=journal,
            )
    except Exception as exc:
        # 未预期异常必须留痕后再抛：静默吞掉会让 error_count 失去意义
        tracer.record(
            "error", "unhandled_exception", status="error", detail=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        tracer.close()

    summary = summarize_run(conn, active_run_id)
    return RunResult(
        run_id=active_run_id,
        question=question,
        state=state.value,
        answer=draft.text,
        citations=collect_citations(ledger, reports),
        cost_cny=summary.cost_cny,
        steps=summary.steps,
        degraded=degraded,
        trace_path=str(jsonl_path) if jsonl_path else None,
    )


def resume_run(
    run_id: str,
    *,
    gateway: ToolGateway,
    client: LLMClient,
    conn: Any,
    artifacts: ArtifactStore,
    budget: Budget | None = None,
    jsonl_path: str | Path | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """从 SQLite 检查点接着跑：已完成的节点不重放，已花的钱不重复花。"""
    ledger = load_ledger(conn, run_id, clock=clock)
    if ledger is None:
        raise FileNotFoundError(f"没有找到 run_id={run_id!r} 的账本，无法恢复")
    return run_once(
        question=ledger.question,
        gateway=gateway,
        client=client,
        conn=conn,
        artifacts=artifacts,
        budget=budget,
        jsonl_path=jsonl_path,
        clock=clock,
        run_id=run_id,
    )
