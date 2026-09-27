import json

import pytest

from akr.orchestration.budget import BudgetState
from akr.orchestration.ledger import (
    SUBTASK_KINDS,
    SUBTASK_STATUSES,
    Citation,
    CostRecord,
    LedgerFormatError,
    SubTask,
    TaskLedger,
    WorkerReport,
    init_ledger_schema,
    load_ledger,
    load_node_result,
    load_worker_report,
    record_node_result,
    result_ref_for,
    save_ledger,
    save_worker_report,
)
from akr.storage.artifact_store import ArtifactStore
from akr.storage.sqlite_store import connect

RUN_ID = "run-2026-09-23-001"


@pytest.fixture()
def conn(tmp_path):
    connection = connect(str(tmp_path / "akr.sqlite"))
    init_ledger_schema(connection)
    yield connection
    connection.close()


def make_report() -> WorkerReport:
    return WorkerReport(
        findings=["甲公司 2024 年营业收入为 12.3 亿元"],
        citations=[Citation(doc_id="doc1", chunk_id="doc1#chunk-03", source="甲公司2024年报.pdf")],
        confidence=0.8,
        open_questions=["未找到 2023 年同口径数据"],
        cost=CostRecord(steps=1, tokens=1_234, cost_cny=0.0123, latency_ms=2_400),
    )


def make_ledger() -> TaskLedger:
    ledger = TaskLedger(question="甲公司的营业收入是多少？")
    ledger.add(SubTask(task_id="t1", goal="检索甲公司营业收入", kind="retrieve"))
    ledger.add(SubTask(task_id="t2", goal="查子公司关系", kind="graph", depends_on=["t1"]))
    ledger.budget.charge(steps=1, llm_calls=1, tokens=500, cost_cny=0.005)
    return ledger


def test_init_ledger_schema_is_idempotent(conn):
    init_ledger_schema(conn)  # 再调一次不应报错
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert {"ledgers", "node_results"} <= names


def test_subtask_defaults_and_enum_membership():
    subtask = SubTask(task_id="t1", goal="检索", kind="retrieve")
    assert subtask.status == "pending"
    assert subtask.depends_on == []
    assert subtask.attempt == 0
    assert subtask.result_ref is None
    assert SUBTASK_KINDS == ("retrieve", "graph", "verify")
    assert SUBTASK_STATUSES == ("pending", "running", "done", "failed")


def test_add_rejects_duplicate_task_id():
    ledger = TaskLedger(question="q")
    ledger.add(SubTask(task_id="t1", goal="检索", kind="retrieve"))
    with pytest.raises(ValueError, match="task_id 必须在一次运行内唯一"):
        ledger.add(SubTask(task_id="t1", goal="又一个", kind="graph"))


def test_mark_updates_status_result_ref_and_attempt():
    ledger = make_ledger()
    subtask = ledger.mark("t1", "done", result_ref="run#t1")
    assert subtask.status == "done"
    assert subtask.result_ref == "run#t1"
    assert subtask.attempt == 0

    retried = ledger.mark("t1", "running", bump_attempt=True)
    assert retried.attempt == 1


def test_mark_rejects_unknown_task_and_status():
    ledger = make_ledger()
    with pytest.raises(KeyError):
        ledger.mark("t9", "done")
    with pytest.raises(ValueError, match="未知子任务状态"):
        ledger.mark("t1", "finished")


def test_unfinished_ids_lists_pending_and_running_only():
    ledger = make_ledger()
    assert ledger.unfinished_ids() == ["t1", "t2"]
    ledger.mark("t1", "done")
    ledger.mark("t2", "running")
    assert ledger.unfinished_ids() == ["t2"]
    ledger.mark("t2", "failed")
    assert ledger.unfinished_ids() == []


def test_ledger_round_trip_through_sqlite(conn):
    ledger = make_ledger()
    ledger.mark("t1", "done", result_ref=result_ref_for(run_id=RUN_ID, task_id="t1"))
    ledger.revise_round = 1
    ledger.answerable = False
    save_ledger(conn, ledger, run_id=RUN_ID)

    restored = load_ledger(conn, RUN_ID)
    assert restored is not None
    assert restored.question == ledger.question
    assert restored.answerable is False
    assert restored.revise_round == 1
    assert set(restored.subtasks) == {"t1", "t2"}
    assert restored.subtasks["t1"].status == "done"
    assert restored.subtasks["t1"].result_ref == f"{RUN_ID}#t1"
    assert restored.subtasks["t2"].depends_on == ["t1"]
    # 预算已用量必须一起恢复，否则断点恢复会白送一次预算
    assert restored.budget.used_of("steps") == 1
    assert restored.budget.used_of("cost_cny") == pytest.approx(0.005)
    assert restored.budget.limit_of("tokens") == 200_000


def test_load_ledger_returns_none_when_absent(conn):
    assert load_ledger(conn, "run-does-not-exist") is None


def test_save_ledger_twice_keeps_single_row(conn):
    ledger = make_ledger()
    save_ledger(conn, ledger, run_id=RUN_ID)
    ledger.mark("t1", "done")
    save_ledger(conn, ledger, run_id=RUN_ID)

    rows = conn.execute("SELECT COUNT(*) FROM ledgers WHERE run_id = ?", (RUN_ID,)).fetchone()
    assert rows[0] == 1
    assert load_ledger(conn, RUN_ID).subtasks["t1"].status == "done"


def test_corrupted_ledger_payload_raises_instead_of_returning_empty(conn):
    save_ledger(conn, make_ledger(), run_id=RUN_ID)
    conn.execute("UPDATE ledgers SET payload = ? WHERE run_id = ?", ("{不是 JSON", RUN_ID))
    conn.commit()
    with pytest.raises(LedgerFormatError, match=RUN_ID):
        load_ledger(conn, RUN_ID)


def test_ledger_with_unknown_kind_is_rejected(conn):
    save_ledger(conn, make_ledger(), run_id=RUN_ID)
    payload = json.loads(
        conn.execute("SELECT payload FROM ledgers WHERE run_id = ?", (RUN_ID,)).fetchone()[0]
    )
    payload["subtasks"][0]["kind"] = "browse"
    conn.execute(
        "UPDATE ledgers SET payload = ? WHERE run_id = ?",
        (json.dumps(payload, ensure_ascii=False), RUN_ID),
    )
    conn.commit()
    with pytest.raises(LedgerFormatError, match="未知子任务类型"):
        load_ledger(conn, RUN_ID)


def test_record_node_result_returns_true_then_false_without_overwrite(conn):
    assert record_node_result(
        conn, run_id=RUN_ID, task_id="t1", node="retrieve", payload={"findings": ["a"]}
    ) is True
    # 第二次写入同一键：幂等，返回 False，且旧结果不被覆盖
    assert record_node_result(
        conn, run_id=RUN_ID, task_id="t1", node="retrieve", payload={"findings": ["b"]}
    ) is False
    assert load_node_result(conn, run_id=RUN_ID, task_id="t1", node="retrieve") == {"findings": ["a"]}


def test_node_result_scope_and_unknown_key(conn):
    record_node_result(conn, run_id=RUN_ID, task_id="t1", node="retrieve", payload={"n": 1})
    # 同一 task/node 换 run_id 是另一条记录
    assert record_node_result(
        conn, run_id="run-002", task_id="t1", node="retrieve", payload={"n": 2}
    ) is True
    assert load_node_result(conn, run_id="run-002", task_id="t1", node="retrieve") == {"n": 2}
    assert load_node_result(conn, run_id=RUN_ID, task_id="t2", node="retrieve") is None


def test_node_result_payload_must_be_json_serialisable(conn):
    with pytest.raises(TypeError):
        record_node_result(
            conn, run_id=RUN_ID, task_id="t1", node="retrieve", payload={"bad": object()}
        )


def test_worker_report_json_round_trip():
    report = make_report()
    restored = WorkerReport.from_json(report.to_json())
    assert restored == report
    assert restored.citations[0].chunk_id == "doc1#chunk-03"
    assert restored.cost.latency_ms == 2_400
    json.loads(report.to_json())  # 必须是合法 JSON（会写进 artifact store 与 trace）


def test_result_ref_for_rejects_separator_in_run_id():
    assert result_ref_for(run_id="run-1", task_id="t1") == "run-1#t1"
    with pytest.raises(ValueError, match="run_id 不得包含"):
        result_ref_for(run_id="run#1", task_id="t1")


def test_save_and_load_worker_report_through_artifact_store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    report = make_report()
    ref = save_worker_report(store, run_id=RUN_ID, task_id="t1", report=report)
    assert ref == f"{RUN_ID}#t1"
    assert store.exists(RUN_ID, "t1")
    assert load_worker_report(store, ref) == report


def test_load_worker_report_rejects_malformed_ref(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(LedgerFormatError, match="result_ref"):
        load_worker_report(store, "no-separator")


def test_cost_record_round_trip_keeps_token_split():
    """明细三列的序列化必须闭环；没有这三列的老工件也要能读回来（补 0）。

    老工件走的是「总量还在、明细不可用」这条路：`runner._dispatch_subtask` 会把总量记进
    `tokens_in` 并留一条 `token_split_missing` note，所以 `tokens` 不能被这次改动动到。
    """

    record = CostRecord(
        steps=2,
        tokens=1_150,
        cost_cny=0.0123,
        latency_ms=800,
        hit_tokens=300,
        miss_tokens=700,
        output_tokens=150,
    )
    assert CostRecord.from_dict(record.to_dict()) == record

    legacy = CostRecord.from_dict(
        {"steps": 2, "tokens": 1_150, "cost_cny": 0.0123, "latency_ms": 800}
    )
    assert legacy.tokens == 1_150
    assert (legacy.hit_tokens, legacy.miss_tokens, legacy.output_tokens) == (0, 0, 0)
