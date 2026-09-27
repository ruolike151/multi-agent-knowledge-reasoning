import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from akr.agents.tool_gateway import ToolCallError
from akr.model.client import LLMResult
from akr.model.cost import Usage
from eval.baselines.react_agent import MAX_WALL_SECONDS
from eval.runners.run_experiment import (
    MAIN_BUDGET_CNY,
    MAIN_WALL_SECONDS,
    SYSTEMS,
    VARIANCE_REPEATS,
    VARIANCE_SUBSET_SIZE,
    BudgetExceeded,
    BudgetGuard,
    completed_run_keys,
    ExperimentError,
    RecordingGateway,
    RunRecord,
    ScoreResult,
    SystemOutput,
    build_runners,
    execute_phase,
    is_refusal_text,
    load_records,
    main,
    resolve_budget_limits,
    run_system,
    save_records,
    select_variance_subset,
    summarize_records,
    tool_context_from_env,
)
from eval.runners.run_ledger import RunCountExceeded, RunLedger
from eval.taskset.schema import EvalTask


class FakeClient:
    """judge 侧的最小客户端：按顺序吐出预置 JSON 文本。"""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append({"system": system, "messages": messages, "mode": mode})
        return LLMResult(
            text=self._payloads.pop(0),
            reasoning="",
            usage=Usage(hit_tokens=10, miss_tokens=20, output_tokens=5),
            mode=mode,
            tool_calls=[],
        )


class FakeGateway:
    """按调用次序吐出预置 hits 的网关；每次调用只返回一批。"""

    def __init__(self, batches=None):
        # batches 是"每次 search_documents 返回的 chunk_id 列表"的列表
        self.batches = list(batches) if batches else [["d1#c1", "d1#c2"]]
        self.index = 0

    def _hits(self):
        batch = self.batches[min(self.index, len(self.batches) - 1)]
        self.index += 1
        return [
            {
                "chunk_id": chunk_id,
                "doc_id": chunk_id.split("#")[0],
                "source": "2023年报.pdf",
                "text": f"{chunk_id} 的正文",
                "score": 1.0,
                "retriever": "vector",
            }
            for chunk_id in batch
        ]

    def search_documents(
        self, query, top_k=8, doc_filters=None, two_stage_docs=0, expand_neighbors=0
    ):
        return {"hits": self._hits(), "fusion": "rrf", "retriever_stats": {}}

    def walk_graph(self, entity, hops=2, relation_types=None):
        return {"paths": []}

    def resolve_entity(self, mention, context=""):
        return {"candidates": []}

    def fetch_chunk(self, chunk_id):
        return {
            "chunk_id": chunk_id,
            "doc_id": chunk_id.split("#")[0],
            "source": "2023年报.pdf",
            "text": f"{chunk_id} 的正文",
            "char_range": [0, 5],
        }

    def list_documents(self):
        return {"documents": []}

    def close(self):
        pass


class ExplodingGateway(FakeGateway):
    def search_documents(
        self, query, top_k=8, doc_filters=None, two_stage_docs=0, expand_neighbors=0
    ):
        raise ToolCallError("搜索引擎超时")


class FakeScorer:
    """固定判分结果，让 runner 的测试不依赖 judge 的具体实现。"""

    def __init__(self, *, judge_score=2.0, supports=(True, True), cost_cny=0.001):
        self.judge_score = judge_score
        self.supports = supports
        self.cost_cny = cost_cny
        self.calls = []

    def score(self, *, task, answer, citations, fetch_evidence):
        self.calls.append({"task_id": task.task_id, "answer": answer, "citations": tuple(citations)})
        return ScoreResult(
            judge_score=self.judge_score,
            supports=tuple(self.supports),
            cost_cny=self.cost_cny,
            calls=1 + len(self.supports),
        )


def make_task(task_id="single_hop-001", **overrides):
    payload = {
        "task_id": task_id,
        "type": "single_hop",
        "difficulty": "easy",
        "question": "甲公司 2023 年营业收入是多少？",
        "gold_answer": "100 亿元",
        "gold_chunk_ids": ["d1#c1"],
        "answerable": True,
        "notes": "",
    }
    payload.update(overrides)
    return EvalTask(**payload)


def make_system_output(answer="营业收入 100 亿元。", citations=("d1#c1",), **overrides):
    payload = {
        "answer": answer,
        "citations": tuple(citations),
        "cost_cny": 0.02,
        "cost_source": "metered",
        "steps": 4,
        "illegal_transitions": 0,
        "latency_ms": 1234.0,
        "degraded": False,
        "state": "done",
    }
    payload.update(overrides)
    return SystemOutput(**payload)


CORE_MIX = (("single_hop", 8), ("multi_hop", 10), ("multi_doc", 7), ("refusal", 5))


def make_taskset():
    """按 spec §9.1 的核心集配额造 30 条任务（8 / 10 / 7 / 5）。"""
    tasks = []
    for kind, count in CORE_MIX:
        for index in range(count):
            refusal = kind == "refusal"
            tasks.append(
                make_task(
                    task_id=f"{kind}-{index:03d}",
                    type=kind,
                    answerable=not refusal,
                    gold_chunk_ids=[] if refusal else [f"d{index}#c1"],
                )
            )
    return tasks


def run_one(system="ours", *, output=None, task=None, scorer=None, guard=None,
            gateway=None, phase="main", repeat=1):
    """跑一次 run_system，默认参数就是一条"理想记录"。

    传入 gateway 时，桩系统会真的走一次网关检索——这样检索指标才落在网关账上，
    而不是由系统自己汇报（设计 2）。
    """

    def run_task(current):
        if gateway is not None:
            gateway.search_documents(current.question)
        return output or make_system_output()

    return run_system(
        system,
        task or make_task(),
        phase=phase,
        repeat=repeat,
        run_task=run_task,
        scorer=scorer or FakeScorer(),
        fetch_evidence=lambda chunk_id: f"{chunk_id} 的正文",
        gateway=gateway,
        guard=guard,
    )


def test_recording_gateway_records_only_the_window_since_mark():
    """指标必须归属于"这一次运行"，所以用 mark/since 划窗口，而不是读累计量。"""
    gateway = RecordingGateway(FakeGateway([["d1#c1", "d1#c2"], ["d2#c9", "d1#c1"]]))

    gateway.search_documents("甲公司")
    mark = gateway.mark()
    gateway.search_documents("乙公司")

    retrieved, successes = gateway.since(mark)

    assert retrieved == ("d2#c9", "d1#c1")     # 只含第二次调用，且去重保序
    assert successes == (True,)
    assert len(gateway.calls) == 2


def test_recording_gateway_since_calls_keeps_each_search_call_separate():
    """逐次口径：不跨调用去重，只记 `search_documents`，别的工具不算一次检索。"""
    gateway = RecordingGateway(FakeGateway([["d1#c1", "d1#c2"], ["d2#c9", "d1#c1"]]))

    gateway.search_documents("甲公司")
    mark = gateway.mark()
    gateway.search_documents("乙公司")
    gateway.fetch_chunk("d1#c1")          # 不是检索：不许混进逐次口径

    assert gateway.since_calls(mark) == (("d2#c9", "d1#c1"),)
    assert gateway.since(mark)[0] == ("d2#c9", "d1#c1")   # 主表口径不受影响
    # 窗口里一次 search_documents 都没有 → 空元组 = 不可测，不是 ((),)
    assert gateway.since_calls(gateway.mark()) == ()


def test_run_system_keeps_both_retrieval_views_and_summary_shows_per_call_recall():
    """一次运行同时留两份检索视图：主表口径（去重截断）+ 逐次调用口径（不截断）。"""
    gateway = RecordingGateway(FakeGateway([["x1", "x2"], ["d1#c1"]]))

    def run_task(current):
        gateway.search_documents(current.question)   # 首查没中
        gateway.search_documents(current.question)   # 二查才补回 gold
        return make_system_output()

    record = run_system(
        "ours",
        make_task(),                                  # gold_chunk_ids = ["d1#c1"]
        phase="main",
        repeat=1,
        run_task=run_task,
        scorer=FakeScorer(),
        fetch_evidence=lambda chunk_id: f"{chunk_id} 的正文",
        gateway=gateway,
    )

    assert record.retrieved_ids == ("x1", "x2", "d1#c1")
    assert record.retrieved_per_call == (("x1", "x2"), ("d1#c1",))

    summary = summarize_records([record])["ours"]
    assert summary["recall@8"] == 1.0                 # 主表口径不变
    assert summary["per_call_recall@8"] == pytest.approx(0.5)


def test_run_record_from_old_jsonl_stays_unmeasurable_for_per_call():
    """旧 JSONL 没有 `retrieved_per_call` 字段：回读必须成功，且逐次口径判为不可测。"""
    legacy = run_one("ours", gateway=RecordingGateway(FakeGateway())).model_dump()
    legacy.pop("retrieved_per_call")

    restored = RunRecord.model_validate(legacy)

    assert restored.retrieved_per_call == ()
    assert "per_call_recall@8" not in summarize_records([restored])["ours"]


def test_recording_gateway_marks_failed_call_and_reraises():
    """失败要记账，但异常必须继续往上抛——记录不等于吞掉。"""
    gateway = RecordingGateway(ExplodingGateway())

    with pytest.raises(ToolCallError, match="超时"):
        gateway.search_documents("甲公司")

    assert [call.success for call in gateway.calls] == [False]
    assert gateway.since(0) == ((), (False,))


def test_is_refusal_text_matches_only_the_shared_markers():
    """拒答判定对两组一视同仁：只看文本标记，不看目标系统独有的状态名。"""
    assert is_refusal_text("无法从知识库确定该公司的营业收入。") is True
    assert is_refusal_text("知识库中未找到相关记录。") is True
    assert is_refusal_text("甲公司 2023 年营业收入为 100 亿元。") is False


def test_select_variance_subset_is_stratified_and_reproducible():
    tasks = make_taskset()

    first = select_variance_subset(tasks)
    second = select_variance_subset(tasks)

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert len(first) == VARIANCE_SUBSET_SIZE == 12
    assert len({task.type for task in first}) == 4      # 四类按比例都抽到
    assert VARIANCE_REPEATS == 3
    assert {task.task_id for task in first} <= {task.task_id for task in tasks}


def test_select_variance_subset_rejects_oversized_request():
    with pytest.raises(ExperimentError, match="无法抽出"):
        select_variance_subset(make_taskset()[:5], size=12)


def test_budget_guard_raises_and_does_not_book_the_overspending_charge():
    guard = BudgetGuard(MAIN_BUDGET_CNY)

    guard.charge(2.0)
    assert guard.spent_cny == pytest.approx(2.0)
    assert guard.remaining_cny == pytest.approx(0.5)

    with pytest.raises(BudgetExceeded, match="2.5"):
        guard.charge(0.6)

    assert guard.spent_cny == pytest.approx(2.0)        # 超支的那笔不落账
    assert SYSTEMS == ("baseline_a", "ours")


def test_run_system_takes_retrieval_metrics_from_the_gateway_window():
    gateway = RecordingGateway(FakeGateway([["d1#c1", "d1#c2", "d1#c1"]]))

    record = run_one("ours", gateway=gateway)

    assert record.retrieved_ids == ("d1#c1", "d1#c2")    # 去重保序
    assert record.tool_call_success == 1.0
    assert record.cost_source == "metered"


def test_run_system_leaves_retrieval_unmeasurable_without_a_gateway():
    record = run_one("ours")

    assert record.retrieved_ids == ()
    assert record.tool_call_success is None      # None ≠ 0.0：不可测不是"全部失败"


def test_run_record_round_trips_through_jsonl(tmp_path):
    record = run_one("ours", phase="variance", repeat=2)
    path = tmp_path / "runs.jsonl"

    save_records(path, [record, record])

    assert load_records(path) == [record, record]
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["phase"] == "variance"
    assert line["repeat"] == 2


def test_run_record_to_metric_input_maps_every_metric_field():
    scorer = FakeScorer(judge_score=1.0, supports=(True, False), cost_cny=0.004)
    record = run_one("ours", scorer=scorer, gateway=RecordingGateway(FakeGateway()))

    metric = record.to_metric_input()

    assert metric.task_id == record.task_id == "single_hop-001"
    assert metric.type == "single_hop"
    assert metric.answerable is True
    assert metric.retrieved_ids == ("d1#c1", "d1#c2")
    assert metric.gold_ids == ("d1#c1",)
    assert metric.supports == (True, False)
    assert metric.answered is True
    assert metric.judge_score == 1.0
    assert metric.latency_ms == record.latency_ms
    assert metric.cost_cny == pytest.approx(0.024)       # 系统 0.02 + judge 0.004
    assert metric.steps == 4
    assert metric.illegal_transitions == 0


def test_run_system_records_error_without_faking_a_score():
    """系统崩了就是崩了：error 落盘、judge_score 留空，绝不写成 0 分。"""

    def boom(_task):
        raise RuntimeError("工具全部不可用")

    record = run_system(
        "ours",
        make_task(),
        phase="main",
        repeat=1,
        run_task=boom,
        scorer=FakeScorer(),
        fetch_evidence=lambda chunk_id: "",
    )

    assert record.error is not None and "工具全部不可用" in record.error
    assert record.judge_score is None
    assert record.supports == ()
    assert record.answered is False
    assert record.answer == ""


def test_run_system_marks_refusal_through_the_shared_text_marker():
    task = make_task(task_id="refusal-000", type="refusal", answerable=False, gold_chunk_ids=[])
    output = make_system_output(
        answer="无法从知识库确定该公司的营业收入。",
        citations=(),
        state="refuse",
    )

    record = run_one("ours", output=output, task=task, scorer=FakeScorer(judge_score=2.0, supports=()))

    assert record.answerable is False
    assert record.answered is False
    assert is_refusal_text(record.answer) is True
    assert record.supports == ()


def test_build_runners_mounts_baseline_a_and_ours():
    """两组都装成同一个签名；没有第三组，也不允许后来者偷偷加回一组。"""
    gateway = RecordingGateway(FakeGateway())

    runners = build_runners(
        client=object(),
        gateway=gateway,
        artifacts=object(),
        conn=object(),
    )

    assert set(runners) == set(SYSTEMS) == {"baseline_a", "ours"}


def test_wall_seconds_defaults_do_not_touch_either_side(monkeypatch):
    """不传 `--wall-seconds` 时两组都走各自原来的默认值（库里的 120 / 基线的 120）。"""
    import eval.runners.run_experiment as runner_module
    from eval.runners.run_experiment import MAIN_WALL_SECONDS

    seen: dict[str, object] = {}

    def fake_run_once(**kwargs):
        seen["budget"] = kwargs["budget"]
        return SimpleNamespace(
            run_id="r1", answer="答案", citations=(), cost_cny=0.0,
            steps=1, degraded=False, state="done",
        )

    class FakeAgent:
        def __init__(self, _client, _gateway, *, max_wall_seconds=None):
            seen["baseline_wall"] = max_wall_seconds

        def run(self, _question):
            return SimpleNamespace(
                answer="答案", citations=(), cost_cny=0.0, steps=1,
                latency_ms=0.0, degraded=False,
            )

    monkeypatch.setattr(runner_module, "run_once", fake_run_once)
    monkeypatch.setattr(
        runner_module, "summarize_run",
        lambda _conn, _run_id: SimpleNamespace(illegal_transitions=0),
    )
    monkeypatch.setattr(runner_module, "ReActAgent", FakeAgent)
    runners = build_runners(
        client=object(), gateway=RecordingGateway(FakeGateway()),
        artifacts=object(), conn=object(),
    )

    runners["ours"](make_task())
    runners["baseline_a"](make_task())

    assert seen["budget"] is None          # 库里 `Budget()` 的默认值原样生效
    assert seen["baseline_wall"] == MAX_WALL_SECONDS == MAIN_WALL_SECONDS


def test_wall_seconds_reaches_both_systems(monkeypatch):
    """改闸必须同时落到两组：目标系统走 `Budget`，基线走 Agent 的构造参数。

    为什么这条要有单测：2026-09-25 主实验里两组都写着"120 秒"，但基线当时**根本没有**
    时间闸门，于是"ours 慢"既可能是架构代价、也可能只有一组被截断——资源盒子不对称的
    数据只能定性看。改闸的入口因此必须在测试里钉住"一个值、两组同吃"。
    """
    import eval.runners.run_experiment as runner_module

    seen: dict[str, object] = {}

    def fake_run_once(**kwargs):
        seen["budget"] = kwargs["budget"]
        return SimpleNamespace(
            run_id="r1", answer="答案", citations=(), cost_cny=0.0,
            steps=1, degraded=False, state="done",
        )

    class FakeAgent:
        def __init__(self, _client, _gateway, *, max_wall_seconds=None):
            seen["baseline_wall"] = max_wall_seconds

        def run(self, _question):
            return SimpleNamespace(
                answer="答案", citations=(), cost_cny=0.0, steps=1,
                latency_ms=0.0, degraded=False,
            )

    monkeypatch.setattr(runner_module, "run_once", fake_run_once)
    monkeypatch.setattr(
        runner_module, "summarize_run",
        lambda _conn, _run_id: SimpleNamespace(illegal_transitions=0),
    )
    monkeypatch.setattr(runner_module, "ReActAgent", FakeAgent)
    runners = build_runners(
        client=object(), gateway=RecordingGateway(FakeGateway()),
        artifacts=object(), conn=object(), wall_seconds=300.0,
    )

    runners["ours"](make_task())
    runners["baseline_a"](make_task())

    assert seen["budget"].max_wall_seconds == 300
    assert seen["baseline_wall"] == 300.0
    # `--wall-seconds` 只改墙钟这一个维度：其余四个维度必须继承 `Budget` 的**当前**
    # 默认值（2026-09-26 重调后是 200_000 tokens / ¥1.0），否则改一次墙钟会把
    # token 与钱盒子悄悄退回 spec 原值，而这两条正是慢题被掐死的直接原因。
    assert seen["budget"].max_tokens == 200_000
    assert seen["budget"].max_cost_cny == 1.0
    assert seen["budget"].max_steps == 24


def _phase(*, tasks, repeats=1, systems=SYSTEMS, ledger=None, runners=None, scorer=None,
           gateway=None, sink=None, skip=None, guard=None):
    """跑一小段阶段循环：默认两个系统都吐同一条理想输出。"""

    def run_task(_task):
        return make_system_output()

    return execute_phase(
        phase="main",
        systems=systems,
        tasks=tasks,
        repeats=repeats,
        runners=runners or {system: run_task for system in systems},
        scorer=scorer or FakeScorer(),
        fetch_evidence=lambda chunk_id: "正文",
        gateway=gateway,
        ledger=ledger,
        skip=skip,
        guard=guard,
        sink=sink,
    )


def test_execute_phase_stops_at_the_run_ledger_cap_and_keeps_what_it_ran(tmp_path):
    """次数闸门是硬停：第 N+1 次运行根本不会发生，已跑完的记录仍交给调用方。"""
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=3)
    collected: list = []
    tasks = [make_task(task_id=f"single_hop-{index:03d}") for index in (1, 2, 3)]

    with pytest.raises(RunCountExceeded, match="3/3"):
        _phase(tasks=tasks, ledger=ledger, sink=collected)

    assert [record.run_id for record in collected] == [
        "main-baseline_a-single_hop-001-r1",
        "main-baseline_a-single_hop-002-r1",
        "main-baseline_a-single_hop-003-r1",
    ]
    assert ledger.used == 3            # 被拦下的那一次没有落账
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run-ledger.json"]


def test_execute_phase_without_a_ledger_runs_the_whole_grid():
    tasks = [make_task(task_id=f"single_hop-{index:03d}") for index in (1, 2)]

    records = _phase(tasks=tasks, repeats=2)

    assert len(records) == 8           # 2 系统 × 2 题 × 2 轮
    assert [record.run_id for record in records[:2]] == [
        "main-baseline_a-single_hop-001-r1",
        "main-baseline_a-single_hop-002-r1",
    ]


def test_a_crashed_run_still_consumes_its_claim(tmp_path):
    """系统崩了也是跑过一次（钱花了、时间占了）：次数照样记，不重试也不退额度。"""
    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=5)

    def boom(_task):
        raise RuntimeError("工具全部不可用")

    records = _phase(
        tasks=[make_task()],
        systems=("ours",),
        runners={"ours": boom},
        ledger=ledger,
    )

    assert len(records) == 1 and records[0].error is not None
    assert ledger.used == 1


def test_cli_help_runs_without_pytest_pythonpath():
    """脚本直跑时 `eval/` 不在 sys.path 里（`.venv` 只把 `src` 装成可编辑包）——这条守住那个补丁。

    2026-09-25 实跑复现过：`python eval\\runners\\run_experiment.py` 在真正开跑前就
    `ModuleNotFoundError: No module named 'eval'`。`--help` 不需要模型与账本，因此这条
    用例零成本、也绝不会碰到 API。
    """

    root = Path(__file__).resolve().parents[2]
    script = root / "eval" / "runners" / "run_experiment.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(root),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "两组系统的同口径对照实验 runner" in result.stdout


def test_tool_context_gets_real_index_paths(monkeypatch):
    """`build_tool_context` 的 `chroma_path` / `bm25_path` 传 `None` = **不建**索引（不是用默认值）。

    2026-09-25 冒烟实测：runner 省略了这两个参数，`search_documents` 立刻在
    `ctx.vectors.query(...)` 上抛 `AttributeError: 'NoneType' object has no attribute 'query'`，
    两种系统 4/4 全部倒在第一步。这条守住"runner 必须传非空路径"。
    """

    captured: dict[str, object] = {}

    def fake_build_tool_context(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("akr.mcp_server.server.build_tool_context", fake_build_tool_context)
    tool_context_from_env(db_path="db.sqlite", artifacts_root="art", neo4j_uri="bolt://x:1")

    assert captured["db_path"] == "db.sqlite"
    assert captured["artifacts_root"] == "art"
    assert captured["neo4j_uri"] == "bolt://x:1"
    assert captured["chroma_path"]
    assert captured["bm25_path"]


# ---- 两道钱闸门的 CLI 旋钮（2026-09-25：用户的控闸口径是次数，不是钱） -------------


def test_resolve_budget_limits_returns_both_limits():
    """合法上限原样返回（且转成 float）：CLI 里的 `2.5` / `1.0` 就是这两道闸。"""

    assert resolve_budget_limits(max_cost_cny=4, judge_cost_cny=2.0) == (4.0, 2.0)


# ---- 续跑（--resume）与"触顶那一笔不能丢记录"（2026-09-25 主实验中场中断后补的） --------


def test_execute_phase_resume_skips_done_cells_without_charging_the_ledger(tmp_path):
    """续跑跳过的格子**不发请求、也不扣次数**：次数是硬闸门，不能因重跑同一格被重复扣。"""

    ledger = RunLedger(tmp_path / "run-ledger.json", max_runs=10)
    seen: list[str] = []

    def run_task(task):
        seen.append(task.task_id)
        return make_system_output()

    records = _phase(
        tasks=[make_task(task_id=f"single_hop-{index:03d}") for index in (1, 2, 3)],
        systems=("baseline_a",),
        runners={"baseline_a": run_task},
        ledger=ledger,
        skip={("main", "baseline_a", "single_hop-002", 1)},
    )

    assert seen == ["single_hop-001", "single_hop-003"]
    assert [record.task_id for record in records] == ["single_hop-001", "single_hop-003"]
    assert ledger.used == 2       # 跳过的那个格子没有被扣次数


def test_completed_run_keys_survive_save_and_load(tmp_path):
    """续跑读回来的跳表键必须与写下去的一致（`--resume` 就是这么读的）。"""

    rows = _phase(
        tasks=[make_task(task_id="single_hop-001")],
        systems=("ours",),
        runners={"ours": lambda _task: make_system_output()},
    )
    path = save_records(tmp_path / "main.jsonl", rows)

    assert completed_run_keys(rows) == {("main", "ours", "single_hop-001", 1)}
    assert completed_run_keys(load_records(path)) == completed_run_keys(rows)


def test_budget_stop_hands_back_the_record_that_spent_the_money(tmp_path):
    """钱闸门顶掉的那一笔**仍然是一条记录**：钱已花、次数已扣，丢掉记录等于账目对不上。

    2026-09-25 主实验实测：闸门顶在第 41 次，账本记 41 次而 `eval/runs/main.jsonl` 只有 40 条。
    修复后记录随异常交出、先落进 sink，再中断整场实验。
    """

    collected: list = []
    guard = BudgetGuard(0.001)      # make_system_output 的成本是 0.02，必定顶掉

    with pytest.raises(BudgetExceeded, match="预算触顶"):
        _phase(
            tasks=[make_task(task_id="single_hop-001")],
            systems=("ours",),
            runners={"ours": lambda _task: make_system_output()},
            guard=guard,
            sink=collected,
        )

    assert len(collected) == 1
    assert collected[0].system_cost_cny == 0.02
    assert "budget:" in (collected[0].error or "")


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_resolve_budget_limits_rejects_non_positive(bad):
    """0 / 负数 / NaN / inf 一律拒掉：非法上限绝不能被解释成"不设闸门"。"""

    with pytest.raises(ExperimentError):
        resolve_budget_limits(max_cost_cny=bad, judge_cost_cny=1.0)
    with pytest.raises(ExperimentError):
        resolve_budget_limits(max_cost_cny=2.5, judge_cost_cny=bad)


def test_cli_rejects_non_positive_budget_before_any_request(tmp_path, capsys):
    """非法上限以退出码 2 结束，且**在开跑之前**：账本一个字节都不该落。"""

    ledger_path = tmp_path / "run-ledger.json"
    with pytest.raises(SystemExit) as excinfo:
        main(["--ledger", str(ledger_path), "--max-cost-cny", "0"])

    assert excinfo.value.code == 2
    assert "--max-cost-cny 必须是正数" in capsys.readouterr().err
    assert not ledger_path.exists()


def test_cli_warns_when_money_gate_is_changed(tmp_path, capsys, monkeypatch):
    """显式改钱闸门必须打得出来（与 `--max-runs` 同一套"改闸要看得见"的口径）。

    这里把 `Settings.from_env` 打成哨兵异常：只要它被调到，就说明已经走到了"检查完参数、
    准备连模型"这一步——参数与警告都过了，而**没有任何请求发出去**。
    """

    class _Stop(RuntimeError):
        pass

    from akr.config import Settings

    def boom(cls):  # noqa: ANN001 - 只是为了在 warning 之后立刻掐断
        raise _Stop("到此为止，不要再往下连模型")

    monkeypatch.setattr(Settings, "from_env", classmethod(boom))
    with pytest.raises(_Stop):
        main(
            [
                "--ledger",
                str(tmp_path / "run-ledger.json"),
                "--max-cost-cny",
                "4.0",
                "--judge-cost-cny",
                "2.0",
            ]
        )

    err = capsys.readouterr().err
    assert "钱闸门被显式改为 系统侧 ¥4.00 / judge ¥2.00" in err


@pytest.mark.parametrize("bad", ["0", "-30", "nan", "inf"])
def test_cli_rejects_non_positive_wall_seconds_before_any_request(
    tmp_path, capsys, bad
):
    """墙钟同理：0 不能解释成"不设闸门"（`elapsed() >= 0` 恒真 = 每一步都触顶）。"""

    ledger_path = tmp_path / "run-ledger.json"
    with pytest.raises(SystemExit) as excinfo:
        main(["--ledger", str(ledger_path), "--wall-seconds", bad])

    assert excinfo.value.code == 2
    assert "--wall-seconds 必须是正数" in capsys.readouterr().err
    assert not ledger_path.exists()


def test_cli_warns_when_wall_gate_is_changed(tmp_path, capsys, monkeypatch):
    """改墙钟与改钱同一套口径：打到 stderr，说清"两组同值生效"。"""

    class _Stop(RuntimeError):
        pass

    from akr.config import Settings

    monkeypatch.setattr(
        Settings,
        "from_env",
        classmethod(lambda cls: (_ for _ in ()).throw(_Stop("到此为止，不要再往下连模型"))),
    )
    with pytest.raises(_Stop):
        main(["--ledger", str(tmp_path / "run-ledger.json"), "--wall-seconds", "300"])

    err = capsys.readouterr().err
    assert "墙钟闸门被显式改为 300 秒（默认 120 秒）" in err
