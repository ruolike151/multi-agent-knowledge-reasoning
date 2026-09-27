import pytest

from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.observability.trace import TraceRecorder, init_trace_schema, summarize_run
from akr.orchestration.budget import Budget
from akr.orchestration.ledger import (
    Citation,
    SubTask,
    TaskLedger,
    WorkerReport,
    init_ledger_schema,
    load_ledger,
)
from akr.orchestration.runner import (
    DEGRADED_NOTICE,
    RUN_ID_BYTES,
    RunResult,
    UNVERIFIED_NOTICE,
    _degraded_draft,
    _provisional_draft,
    resume_run,
    run_once,
)
from akr.orchestration.states import LEGAL_TRANSITIONS, State
from akr.storage.artifact_store import Artifact, ArtifactStore
from akr.storage.sqlite_store import connect, init_schema

QUESTION = "甲公司 2024 年的营业收入是多少？"

PLAN_JSON = (
    '{"answerable": true, "reason": "先检索年报", "next_state": "DISPATCHING",'
    ' "subtasks": [{"task_id": "t1", "goal": "查甲公司营业收入",'
    ' "kind": "retrieve", "depends_on": []}]}'
)
ROUTE_JSON = '{"next_state": "VERIFYING", "reason": "证据已经够用"}'
# 校验通过后的合法后继：`VERIFYING` 的合法后继是 SYNTHESIZING / REVISE / DEGRADED，
# 写回 "VERIFYING" 会被路由器驳回两次后强制降级（见 supervisor.MAX_ROUTE_ATTEMPTS）。
# 本文件里「happy path」的 fixture 必须用这条，否则测的是降级路。
ROUTE_SYNTHESIZE_JSON = '{"next_state": "SYNTHESIZING", "reason": "证据已经够用"}'
# 校验后由路由器自己判「证据不足以支撑结论」：这条边是转移表里的正常收敛边
# （VERIFYING → DEGRADED），与「预算触顶」是两件完全不同的事。
ROUTE_DEGRADED_JSON = '{"next_state": "DEGRADED", "reason": "证据不足以支撑结论"}'
SUMMARY_JSON = (
    '{"findings": ["甲公司 2024 年营业收入为 12.3 亿元"],'
    ' "chunk_ids": ["doc1#chunk-03"], "confidence": 0.8, "open_questions": []}'
)
VERIFY_OK_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "supported",'
    ' "reason": "与原文一致", "chunk_ids": ["doc1#chunk-03"]}]}'
)
VERIFY_BAD_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "unsupported",'
    ' "reason": "原文写的是 1.23 亿元", "chunk_ids": []}]}'
)
SYNTHESIZE_JSON = (
    '{"text": "甲公司 2024 年营业收入为 12.3 亿元。", "chunk_ids": ["doc1#chunk-03"]}'
)

class FakeClient:
    """按脚本顺序返回响应；脚本里放异常对象就抛出来，用来模拟中途崩溃。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append({"system": system, "mode": mode, "messages": messages})
        if not self._script:
            raise AssertionError(f"脚本已用完，收到第 {len(self.calls)} 次模型调用")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResult(
            text=item,
            reasoning="",
            usage=Usage(hit_tokens=0, miss_tokens=1_000, output_tokens=200),
            mode=mode,
            tool_calls=[],
        )


class FakeGateway:
    """离线工具网关，返回形状与 tests/agents/test_retriever.py 里的 fake 保持一致。

    `two_stage_docs` 只是签名对齐（`runner._retrieve` 会显式传 3）：离线 fake 没有
    第二段检索，把它原样记下来，供「编排层确实开了两段式」这类断言用。
    """

    def __init__(self):
        self.two_stage_docs = None
        self.expand_neighbors = None
        self.queries = []

    def search_documents(
        self, query, top_k=8, doc_filters=None, two_stage_docs=0, expand_neighbors=0
    ):
        self.queries.append(query)
        self.two_stage_docs = two_stage_docs
        self.expand_neighbors = expand_neighbors
        return {
            "hits": [
                {
                    "chunk_id": "doc1#chunk-03",
                    "doc_id": "doc1",
                    "source": "年报2024.pdf",
                    "text": "2024 年公司实现营业收入 12.3 亿元",
                    "score": 0.9,
                }
            ]
        }

    def fetch_chunk(self, chunk_id):
        return {
            "chunk_id": chunk_id,
            "doc_id": "doc1",
            "source": "年报2024.pdf",
            "text": "2024 年公司实现营业收入 12.3 亿元，同比增长 8%",
            "page": 1,
        }

    def walk_graph(self, entity, hops=2, relation_types=None):
        return {"paths": []}

    def resolve_entity(self, mention, context=""):
        return {"candidates": []}

    def list_documents(self):
        return {"documents": [{"doc_id": "doc1", "source": "年报2024.pdf", "chunks": 1}]}

    def close(self):
        return None


class SplitUsageClient(FakeClient):
    """脚本与 `FakeClient` 一致，只把用量换成"命中 + 未命中 + 输出"都有值的一笔。

    `FakeClient` 的用量是 `hit=0`，那样 `tokens_cached` 列是不是真的落对了根本看不出来。
    """

    USAGE = Usage(hit_tokens=300, miss_tokens=700, output_tokens=150)

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        result = super().chat(
            system=system, messages=messages, mode=mode, tools=tools, json_object=json_object
        )
        return LLMResult(
            text=result.text,
            reasoning=result.reasoning,
            usage=self.USAGE,
            mode=mode,
            tool_calls=result.tool_calls,
        )


@pytest.fixture()
def env(tmp_path):
    """SQLite + artifact store 的最小环境；三张 schema 都由测试自己初始化。"""
    conn = connect(tmp_path / "akr.sqlite")
    init_schema(conn)
    init_ledger_schema(conn)
    init_trace_schema(conn)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    artifacts.put(
        Artifact(
            chunk_id="doc1#chunk-03",
            doc_id="doc1",
            text="2024 年公司实现营业收入 12.3 亿元，同比增长 8%",
            char_range=(0, 30),
            page=1,
        )
    )
    return conn, artifacts


def _recorded_states(conn, run_id):
    return [e.state for e in TraceRecorder(conn, run_id=run_id).events(run_id) if e.kind == "state"]


def _assert_legal_path(states):
    """把 trace 里的状态序列重放一遍：任何一步不在转移表里都会在这里失败。"""
    prev = State.PLANNING.value
    for current in states:
        assert State(current) in LEGAL_TRANSITIONS[State(prev)], f"{prev} -> {current} 不在转移表里"
        prev = current


def _run(conn, artifacts, client, **kwargs):
    kwargs.setdefault("gateway", FakeGateway())
    return run_once(
        question=QUESTION,
        client=client,
        conn=conn,
        artifacts=artifacts,
        **kwargs,
    )


def test_run_once_happy_path_reaches_done(env):
    conn, artifacts = env
    client = FakeClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON]
    )

    result = _run(conn, artifacts, client)

    assert isinstance(result, RunResult)
    assert result.state == State.DONE.value
    assert result.degraded is False
    assert "12.3" in result.answer
    assert [c.chunk_id for c in result.citations] == ["doc1#chunk-03"]
    assert len(result.run_id) == RUN_ID_BYTES * 2
    assert result.trace_path is None
    _assert_legal_path(_recorded_states(conn, result.run_id))


def test_run_result_metrics_come_from_trace(env):
    conn, artifacts = env
    client = FakeClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON]
    )

    result = _run(conn, artifacts, client)
    summary = summarize_run(conn, result.run_id)

    assert result.steps == summary.steps == len(_recorded_states(conn, result.run_id))
    assert result.cost_cny == summary.cost_cny > 0


def test_worker_llm_event_records_token_split(env):
    """worker 的事件必须把明细落进各自的列（偏差日志第三十二节）。

    2026-09-25 之前这里把该子任务的**总量**写进 `tokens_out`、输入两列留 0：worker 的
    调用因此既丢了输入明细、又把输入算成了输出，spec §7.5 的缓存命中率无从计算。
    """

    conn, artifacts = env
    client = SplitUsageClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON]
    )

    result = _run(conn, artifacts, client)
    events = TraceRecorder(conn, run_id=result.run_id).events(result.run_id)
    workers = [event for event in events if event.kind == "llm" and event.name.startswith("worker")]

    assert len(workers) == 1
    worker = workers[0]
    assert (worker.tokens_in, worker.tokens_cached, worker.tokens_out) == (700, 300, 150)
    assert worker.total_tokens == SplitUsageClient.USAGE.total_tokens
    # 明细齐了就不该有兜底 note（那条 note 只给「总量还在、明细没有」的旧工件）
    assert [event.name for event in events if event.kind == "note" and event.name == "token_split_missing"] == []


def test_retrieval_query_is_the_original_question_not_the_subtask_goal(env):
    """检索用的是原问句 + 两段式（偏差日志第三十一节）。

    规划器把 `QUESTION` 改写成了子任务目标「查甲公司营业收入」，而实测这个改写版本
    当查询更差。这条断言把「编排层确实换成了原问句、并且开了两段式」钉死在代码上：
    两者都是系统行为的一部分，不能靠读注释确认。
    """
    conn, artifacts = env
    gateway = FakeGateway()
    client = FakeClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON]
    )

    _run(conn, artifacts, client, gateway=gateway)

    assert gateway.queries == [QUESTION]
    assert gateway.two_stage_docs == 3


def test_revise_round_keeps_the_feedback_goal_as_its_query(env):
    """纠错环不能也被换成原问句：它的子任务目标里带着 Verifier 的反馈。

    换掉等于让补证据白跑一遍同样的检索——这是「原问句」这条规则的边界，必须显式测。
    """
    conn, artifacts = env
    gateway = FakeGateway()
    client = FakeClient(
        [
            PLAN_JSON,
            ROUTE_JSON,
            SUMMARY_JSON,
            VERIFY_BAD_JSON,
            SUMMARY_JSON,
            ROUTE_JSON,
            VERIFY_OK_JSON,
            ROUTE_SYNTHESIZE_JSON,
            SYNTHESIZE_JSON,
        ]
    )

    _run(conn, artifacts, client, gateway=gateway)

    assert gateway.queries[0] == QUESTION
    assert len(gateway.queries) == 2
    assert gateway.queries[1].startswith("补证据（第 1 轮）：")
    # 两段式对两轮都开：它改的是候选排序，与「查什么」是两件事
    assert gateway.two_stage_docs == 3


def test_run_once_selfchecks_transition_table_on_start(env, monkeypatch):
    """启动自检必须在每次 run 的最前面跑：转移表不良构时后面所有结论都不成立。"""
    import akr.orchestration.runner as runner_module

    conn, artifacts = env
    calls = []
    real = runner_module.check_all_transitions
    monkeypatch.setattr(
        runner_module, "check_all_transitions", lambda: (calls.append(1), real())[1]
    )
    client = FakeClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON]
    )

    _run(conn, artifacts, client)

    assert calls == [1]


def test_rerun_with_same_run_id_is_a_noop(env):
    """同一个 run_id 再跑一次不能重复烧钱：终态 + 检查点让循环体一次都不进。"""
    conn, artifacts = env
    first = _run(
        conn,
        artifacts,
        FakeClient([PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON]),
        run_id="deadbeefcafe",
    )
    client_again = FakeClient([])

    again = _run(conn, artifacts, client_again, run_id="deadbeefcafe")

    assert client_again.calls == []
    assert again.answer == first.answer
    assert again.steps == first.steps


def test_resume_run_recovers_from_checkpoints(env):
    """中途崩在 verify：恢复时 plan / route / worker 都必须命中检查点，不再重放。"""
    conn, artifacts = env
    run_id = "deadbeefcafe"
    broken = FakeClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, RuntimeError("verify 阶段进程被杀")]
    )

    with pytest.raises(RuntimeError):
        _run(conn, artifacts, broken, run_id=run_id)

    assert load_ledger(conn, run_id) is not None
    healthy = FakeClient([VERIFY_OK_JSON, ROUTE_SYNTHESIZE_JSON, SYNTHESIZE_JSON])
    result = resume_run(
        run_id,
        gateway=FakeGateway(),
        client=healthy,
        conn=conn,
        artifacts=artifacts,
    )

    assert result.state == State.DONE.value
    assert len(healthy.calls) == 3  # 只剩 verify / route / synthesize 三次真实调用
    notes = [e.name for e in TraceRecorder(conn, run_id=run_id).events(run_id) if e.kind == "note"]
    assert notes.count("checkpoint_hit") >= 3
    _assert_legal_path(_recorded_states(conn, run_id))


def test_budget_exhaustion_converges_through_legal_transitions(env):
    """预算在规划后触顶：一次 LLM 都不再发，且只能走转移表里存在的收敛路径。"""
    conn, artifacts = env
    client = FakeClient([PLAN_JSON])

    result = _run(conn, artifacts, client, budget=Budget(max_cost_cny=1e-9))

    assert result.degraded is True
    assert result.state == State.DONE.value
    assert len(client.calls) == 1  # 触顶之后没有任何模型调用
    assert DEGRADED_NOTICE in result.answer
    states = _recorded_states(conn, result.run_id)
    assert states[-1] == State.DONE.value
    assert State.DEGRADED.value in states
    _assert_legal_path(states)
    errors = [e for e in TraceRecorder(conn, run_id=result.run_id).events(result.run_id)
              if e.kind == "error"]
    assert [e.name for e in errors] == ["budget_exhausted"]


def test_router_chosen_degraded_is_flagged_and_not_blamed_on_budget(env):
    """路由器判 DEGRADED（证据不足以支撑结论）与预算触顶是两件事，报数必须分得开。

    2026-09-25 主实验实测：`ours` 30 条里有 21 条最终答案是「降级草稿」，其中 7 条的
    trace 里**根本没有** `budget_exhausted` 事件——它们是 `VERIFYING → DEGRADED` 这条
    正常收敛边走出来的。而 `RunResult.degraded` 只在这条边之外被置位，于是这 7 条
    （连同预算触顶但标志同样漏掉的 4 条）在记录里显示成「正常跑完」，答案却带着
    「预算已耗尽」的免责声明：标志漏报 + 归因说错，两处都是错。
    """
    conn, artifacts = env
    client = FakeClient(
        [PLAN_JSON, ROUTE_JSON, SUMMARY_JSON, VERIFY_OK_JSON, ROUTE_DEGRADED_JSON]
    )

    result = _run(conn, artifacts, client)

    assert result.state == State.DONE.value
    assert result.degraded is True
    # 归因必须说对：这条路上一个预算维度都没触顶，不能报「预算已耗尽」
    assert DEGRADED_NOTICE not in result.answer
    assert UNVERIFIED_NOTICE in result.answer
    errors = [
        e for e in TraceRecorder(conn, run_id=result.run_id).events(result.run_id)
        if e.kind == "error"
    ]
    assert [e.name for e in errors] == []
    _assert_legal_path(_recorded_states(conn, result.run_id))


def test_unsupported_verdict_adds_verify_subtask_with_feedback(env):
    """纠错环必须以子任务形式落地：反馈进 goal，下一轮 WORKING 才会真的去补证据。"""
    conn, artifacts = env
    client = FakeClient(
        [
            PLAN_JSON,          # plan
            ROUTE_JSON,         # route-working-r0 -> VERIFYING
            SUMMARY_JSON,       # worker t1
            VERIFY_BAD_JSON,    # verify-r0 -> unsupported
                SUMMARY_JSON,       # worker t-revise-1（kind="verify"）
                ROUTE_JSON,         # route-working-r1 -> VERIFYING
                VERIFY_OK_JSON,     # verify-r1 -> supported
                ROUTE_SYNTHESIZE_JSON,  # route-verifying-r1 -> SYNTHESIZING
                SYNTHESIZE_JSON,    # synthesize
        ]
    )

    result = _run(conn, artifacts, client)

    ledger = load_ledger(conn, result.run_id)
    revise_tasks = [t for t in ledger.subtasks.values() if t.kind == "verify"]
    assert len(revise_tasks) == 1
    assert "原文写的是 1.23 亿元" in revise_tasks[0].goal
    assert revise_tasks[0].status == "done"
    assert ledger.revise_round == 1
    assert result.degraded is False
    assert result.state == State.DONE.value
    _assert_legal_path(_recorded_states(conn, result.run_id))


# --- P1：worker 的否定式 finding 不进校验草稿 ------------------------------------
#
# 草稿是「待校验的候选答案」。worker 报告里「检索片段中未出现……金额」这类句子是
# **检索范围的说明**，不是答案结论，却被 `_provisional_draft` 原样拼进草稿，
# 于是被 `split_claims` 切成断言、必然判 unsupported、必然触发一轮 REVISE：
# 这是 verify 均值 2.47 次/题、吃掉 70% LLM 花费的主要来源（见内部偏差日志
# 第四十二节）。注意过滤器只作用于**校验输入**：账本与降级答案里的原话一字不改。

POSITIVE_FINDING = "甲公司 2024 年营业收入为 12.3 亿元"
NEGATIVE_FINDING = "检索片段中未出现甲公司 2024 年归属于上市公司股东的净利润金额"


def _ledger_and_reports(*findings):
    ledger = TaskLedger(question=QUESTION)
    ledger.add(SubTask(task_id="t1", goal="查甲公司营业收入", kind="retrieve"))
    ledger.mark("t1", "done")
    report = WorkerReport(
        findings=list(findings),
        citations=[Citation(doc_id="doc1", chunk_id="doc1#chunk-03", source="年报2024.pdf")],
    )
    return ledger, {"t1": report}


def test_provisional_draft_drops_negative_findings_before_verifying():
    ledger, reports = _ledger_and_reports(POSITIVE_FINDING, NEGATIVE_FINDING)
    draft = _provisional_draft(ledger, reports, drop_negative_findings=True)
    assert POSITIVE_FINDING in draft.text
    assert NEGATIVE_FINDING not in draft.text
    # 引用不能被「顺带」删掉：证据集合是后续取原文的依据。
    assert draft.chunk_ids == ["doc1#chunk-03"]


def test_provisional_draft_keeps_negatives_when_every_finding_is_negative():
    """全是否定式时不能滤成空草稿：空草稿会被读成「一条证据都没取到」而去降级。"""

    ledger, reports = _ledger_and_reports(NEGATIVE_FINDING)
    draft = _provisional_draft(ledger, reports, drop_negative_findings=True)
    assert NEGATIVE_FINDING in draft.text


def test_provisional_draft_keeps_negatives_by_default():
    """默认不滤：只有校验路径显式要求时才滤，调用方漏传不会静默改语义。"""

    ledger, reports = _ledger_and_reports(POSITIVE_FINDING, NEGATIVE_FINDING)
    assert NEGATIVE_FINDING in _provisional_draft(ledger, reports).text


def test_degraded_draft_keeps_negative_findings():
    """降级答案是给人看的：worker 原话里的「没查到什么」必须保留，否则就是不诚实。"""

    ledger, reports = _ledger_and_reports(POSITIVE_FINDING, NEGATIVE_FINDING)
    assert NEGATIVE_FINDING in _degraded_draft(ledger, reports).text
