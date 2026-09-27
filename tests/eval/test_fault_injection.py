import pytest
import subprocess
import sys
from pathlib import Path

from akr.agents.tool_gateway import ToolCallError
from eval.runners.fault_injection import (
    FAULT_RATES,
    FAULT_SEED,
    FaultConfig,
    FaultInjectionLog,
    FaultRunLog,
    FaultyGateway,
    InjectedFault,
    InjectedTimeout,
    load_fault_logs,
    run_fault_experiment,
    save_fault_logs,
    summarize_faults,
)
from eval.runners.run_experiment import (
    BudgetExceeded,
    BudgetGuard,
    ExperimentError,
    RecordingGateway,
    RunRecord,
    ScoreResult,
    SystemOutput,
)
from eval.runners.run_ledger import RunCountExceeded, RunLedger
from eval.taskset.schema import EvalTask


class FakeGateway:
    """最小网关：五个工具都成功，并记录自己被调了哪些次。"""

    def __init__(self):
        self.calls = []

    def search_documents(
        self, query, top_k=8, doc_filters=None, two_stage_docs=0, expand_neighbors=0
    ):
        self.calls.append("search_documents")
        return {
            "hits": [
                {
                    "chunk_id": "d1#c1",
                    "doc_id": "d1",
                    "source": "甲公司2023年报.pdf",
                    "text": "营业收入 100 亿元",
                    "score": 1.0,
                    "retriever": "vector",
                }
            ],
            "fusion": "rrf",
            "retriever_stats": {},
        }

    def walk_graph(self, entity, hops=2, relation_types=None):
        self.calls.append("walk_graph")
        return {"paths": []}

    def resolve_entity(self, mention, context=""):
        self.calls.append("resolve_entity")
        return {"candidates": []}

    def fetch_chunk(self, chunk_id):
        self.calls.append("fetch_chunk")
        return {
            "chunk_id": chunk_id,
            "doc_id": "d1",
            "source": "甲公司2023年报.pdf",
            "text": "营业收入 100 亿元",
            "char_range": [0, 5],
        }

    def list_documents(self):
        self.calls.append("list_documents")
        return {"documents": []}

    def close(self):
        pass


def make_task(task_id="single_hop-001"):
    return EvalTask(
        task_id=task_id,
        type="single_hop",
        difficulty="easy",
        question="甲公司 2023 年营业收入是多少？",
        gold_answer="100 亿元",
        gold_chunk_ids=["d1#c1"],
        answerable=True,
        notes="",
    )


class FakeScorer:
    """固定判分：这道题的分数由构造参数决定，故障注入实验不关心 judge 的实现。"""

    def __init__(self, judge_score=2.0):
        self.judge_score = judge_score

    def score(self, *, task, answer, citations, fetch_evidence):
        return ScoreResult(
            judge_score=self.judge_score,
            supports=(True,),
            cost_cny=0.001,
            calls=2,
        )


def make_run_task(*, degrade=False):
    """桩系统：调一次检索，然后照抄一条答案。

    `degrade=False`：工具报错就往上抛（模拟没有降级路径的系统）；
    `degrade=True`：工具报错仍给出降级答案（模拟"故障下也答得出来"）。
    """

    def run_task(task, gateway):
        try:
            hits = gateway.search_documents(task.question)["hits"]
        except ToolCallError:
            if not degrade:
                raise
            return SystemOutput(
                answer="工具不可用，已降级回答。",
                citations=(),
                cost_cny=0.01,
                cost_source="metered",
                steps=2,
                illegal_transitions=0,
                latency_ms=10.0,
                degraded=True,
                state="degraded",
            )
        return SystemOutput(
            answer=f"营业收入 100 亿元，见 {hits[0]['chunk_id']}。",
            citations=(hits[0]["chunk_id"],),
            cost_cny=0.02,
            cost_source="metered",
            steps=3,
            illegal_transitions=0,
            latency_ms=20.0,
            degraded=False,
            state="done",
        )

    return run_task


def drain(gateway, times=1, query="甲公司"):
    """反复调检索，返回"每次是否被注入故障"的序列。"""
    outcomes = []
    for _ in range(times):
        try:
            gateway.search_documents(query)
        except ToolCallError:
            outcomes.append(False)
        else:
            outcomes.append(True)
    return outcomes


def test_fault_config_defaults_to_search_documents_and_bounds_the_rate():
    config = FaultConfig(rate=0.2)
    assert config.tools == ("search_documents",)   # 只碰检索：降级路径就长在这条工具上
    assert config.mode == "error" and config.seed == FAULT_SEED
    with pytest.raises(ValueError):
        FaultConfig(rate=-0.01)
    with pytest.raises(ValueError):
        FaultConfig(rate=1.01)


def test_fault_config_rejects_unknown_mode_and_tools():
    with pytest.raises(ValueError):
        FaultConfig(rate=0.2, mode="explode")
    with pytest.raises(ValueError):
        FaultConfig(rate=0.2, tools=())
    with pytest.raises(ValueError):
        FaultConfig(rate=0.2, tools=("search_document",))   # 拼错的工具名不许静默失效


def test_fault_config_can_target_another_tool():
    gateway = FaultyGateway(FakeGateway(), FaultConfig(rate=1.0, tools=("fetch_chunk",)))
    assert drain(gateway, times=2) == [True, True]   # 检索不在射程内，照常成功
    with pytest.raises(InjectedFault):
        gateway.fetch_chunk("d1#c1")
    assert gateway.log().total_calls == 1            # 注入率的分母只数射程内的工具


def test_faulty_gateway_never_injects_at_zero_rate():
    gateway = FaultyGateway(FakeGateway(), FaultConfig(rate=0.0))
    assert drain(gateway, times=20) == [True] * 20
    log = gateway.log()
    assert (log.total_calls, log.injected, log.injected_rate) == (20, 0, 0.0)


def test_faulty_gateway_always_injects_at_full_rate():
    gateway = FaultyGateway(FakeGateway(), FaultConfig(rate=1.0))
    with pytest.raises(InjectedFault):
        gateway.search_documents("甲公司")
    log = gateway.log()
    assert (log.total_calls, log.injected, log.injected_rate) == (1, 1, 1.0)


def test_injected_fault_shapes_match_the_real_tool_error():
    assert issubclass(InjectedFault, ToolCallError)   # 形状与真实工具失败一致
    assert issubclass(InjectedTimeout, InjectedFault)
    gateway = FaultyGateway(FakeGateway(), FaultConfig(rate=1.0, mode="timeout"))
    with pytest.raises(InjectedTimeout, match="超时"):
        gateway.search_documents("甲公司")


def test_injection_sequence_is_reproducible_for_the_same_seed():
    config = FaultConfig(rate=0.3, seed=7)
    first = drain(FaultyGateway(FakeGateway(), config), times=30)
    second = drain(FaultyGateway(FakeGateway(), config), times=30)
    assert first == second                           # 换机器也一样：故障位置不靠运气
    assert False in first and True in first          # 30 次里既有注入也有放行，不是常数


def test_faulty_gateway_leaves_other_tools_alone():
    gateway = FaultyGateway(FakeGateway(), FaultConfig(rate=1.0))
    assert gateway.walk_graph("甲公司")["paths"] == []
    assert gateway.resolve_entity("甲公司")["candidates"] == []
    assert gateway.list_documents()["documents"] == []
    assert gateway.fetch_chunk("d1#c1")["chunk_id"] == "d1#c1"
    assert gateway.log() == FaultInjectionLog(total_calls=0, injected=0, injected_rate=0.0)


def test_faulty_gateway_passes_the_payload_through_untouched():
    gateway = FaultyGateway(FakeGateway(), FaultConfig(rate=0.0))
    payload = gateway.search_documents("甲公司", top_k=3, doc_filters={"doc_id": "d1"})
    assert payload["fusion"] == "rrf"
    assert payload["hits"][0]["chunk_id"] == "d1#c1"
    assert gateway.log().total_calls == 1


def test_injected_failures_enter_the_gateway_call_ledger():
    recording = RecordingGateway(FaultyGateway(FakeGateway(), FaultConfig(rate=1.0)))
    mark = recording.mark()
    with pytest.raises(InjectedFault):
        recording.search_documents("甲公司")
    chunk_ids, successes = recording.since(mark)
    assert chunk_ids == ()
    assert successes == (False,)   # 注入的失败与真实失败在调用账上一模一样


def _fault_records(*, degrade, rates=(0.0, 1.0), modes=("error",), repeats=1, guard=None,
                   ledger=None, sink=None, log_sink=None):
    """跑一小场故障实验：两题、给定档位与模式，方便逐条断言。"""
    return run_fault_experiment(
        tasks=[make_task("single_hop-001"), make_task("multi_hop-001")],
        run_task=make_run_task(degrade=degrade),
        scorer=FakeScorer(),
        fetch_evidence=lambda chunk_id: "营业收入 100 亿元",
        gateway=FakeGateway(),
        rates=rates,
        modes=modes,
        repeats=repeats,
        guard=guard,
        ledger=ledger,
        sink=sink,
        log_sink=log_sink,
    )


def test_run_fault_experiment_emits_one_record_per_condition():
    records, reports = _fault_records(degrade=False, rates=(0.0, 1.0))
    assert len(records) == 4                     # 2 档 × 1 模式 × 1 次 × 2 题
    assert len({record.run_id for record in records}) == 4   # run_id 带档位，不会撞车
    assert all(record.system == "ours" for record in records)
    assert all(record.phase.startswith("fault") for record in records)
    healthy = [record for record in records if "0.0" in record.phase]
    broken = [record for record in records if "1.0" in record.phase]
    assert [record.error for record in healthy] == [None, None]
    assert [record.judge_score for record in healthy] == [2.0, 2.0]
    assert all(record.error and "InjectedFault" in record.error for record in broken)
    assert all(record.judge_score is None and record.state == "error" for record in broken)
    assert [report.fault_rate for report in reports] == [0.0, 1.0]


def test_recovery_rate_tells_degrading_apart_from_broken():
    _, reports = _fault_records(degrade=True, rates=(0.0, 1.0))
    by_rate = {report.fault_rate: report for report in reports}
    zero, full = by_rate[0.0], by_rate[1.0]
    assert (zero.injected_calls, zero.injected_runs, zero.recovered) == (0, 0, 0)
    assert zero.recovery_rate is None        # 没有故障可恢复：写 None，不写 0 也不写 1
    assert zero.final_success_rate == 1.0
    assert (full.injected_calls, full.injected_runs) == (2, 2)
    assert full.recovered == 2 and full.recovery_rate == pytest.approx(1.0)
    assert full.final_success_rate == pytest.approx(1.0)
    assert full.degraded_rate == pytest.approx(1.0)


def test_a_stub_without_a_degraded_path_recovers_less():
    _, strict = _fault_records(degrade=False, rates=(1.0,))
    _, degrading = _fault_records(degrade=True, rates=(1.0,))
    assert (strict[0].recovered, strict[0].recovery_rate) == (0, 0.0)
    assert strict[0].final_success_rate == 0.0
    assert degrading[0].recovery_rate > strict[0].recovery_rate
    assert degrading[0].degraded_rate > strict[0].degraded_rate


def test_summarize_faults_reports_every_rate_including_zero():
    _, reports = _fault_records(degrade=False, rates=FAULT_RATES, repeats=2)
    assert [report.fault_rate for report in reports] == list(FAULT_RATES)
    assert {report.runs for report in reports} == {4}     # 2 题 × 2 次
    assert reports[0].injected_calls == 0                 # 0% 档是整条曲线的参照点
    assert reports[-1].final_success_rate == 0.0


def test_summarize_faults_requires_a_record_for_every_log():
    orphan = FaultRunLog(
        mode="error",
        rate=0.2,
        repeat=1,
        task_id="single_hop-001",
        run_id="fault-error-0.2-ours-single_hop-001-r1",
        log=FaultInjectionLog(total_calls=3, injected=1, injected_rate=1 / 3),
    )
    with pytest.raises(ExperimentError, match="找不到"):
        summarize_faults([], [orphan])


def test_fault_logs_round_trip_and_reproduce_the_report(tmp_path):
    collected, logs = [], []
    _, reports = _fault_records(degrade=True, rates=(0.0, 1.0), sink=collected, log_sink=logs)
    loaded = load_fault_logs(save_fault_logs(tmp_path / "fault-logs.jsonl", logs))
    assert loaded == logs                    # 故障条件与注入账一字不差地回来
    assert [entry.run_id for entry in loaded] == [record.run_id for record in collected]
    assert summarize_faults(collected, loaded) == reports   # 报告可以离线重算（Task 37）


def test_run_fault_experiment_stops_when_the_guard_is_exhausted():
    collected = []
    with pytest.raises(BudgetExceeded):
        _fault_records(
            degrade=False,
            rates=(0.0,),
            repeats=2,
            guard=BudgetGuard(0.03),   # 每次运行 0.02：第一题通过，第二题触顶
            sink=collected,
        )
    assert len(collected) == 1         # 中断时已跑完的记录仍要能被调用方拿到


def test_the_fault_matrix_stops_at_the_run_ledger_cap(tmp_path):
    """故障矩阵默认 108 次，装不进 60 次的账：触顶即停，记录与注入账都不留半条。"""
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=3)
    collected, logs = [], []

    with pytest.raises(RunCountExceeded, match="3/3"):
        _fault_records(
            degrade=True,
            rates=(0.0, 1.0),
            repeats=2,
            ledger=ledger,
            sink=collected,
            log_sink=logs,
        )

    assert len(collected) == len(logs) == 3   # 一一对应：恢复率的分母不会出现孤儿
    assert ledger.used == 3
    assert [entry.run_id for entry in logs] == [record.run_id for record in collected]


def test_the_fault_matrix_shares_one_book_with_the_main_phase(tmp_path):
    """与 Task 35 共用一本账：主实验花掉的次数，故障矩阵那边看得见。"""
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=6)
    for index in range(5):
        ledger.claim(phase="main", system="ours", task_id=f"single_hop-{index:03d}")
    collected = []

    with pytest.raises(RunCountExceeded):
        _fault_records(degrade=True, rates=(1.0,), ledger=ledger, sink=collected)

    assert len(collected) == 1         # 只剩一次额度：两题的矩阵跑不完
    assert ledger.used == 6


def test_cli_help_runs_without_pytest_pythonpath():
    """同 `run_experiment.py`：脚本直跑时 `eval/` 不在 sys.path 里，`--help` 必须能起来。"""

    root = Path(__file__).resolve().parents[2]
    script = root / "eval" / "runners" / "fault_injection.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(root),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "工具故障注入实验" in result.stdout
