import json

import pytest

from akr.observability.trace import TraceRecorder, init_trace_schema, summarize_run
from akr.storage import sqlite_store


def make_conn(tmp_path):
    conn = sqlite_store.connect(tmp_path / "akr.sqlite")
    init_trace_schema(conn)
    return conn


def make_recorder(tmp_path, conn, *, run_id="run-1", jsonl_path=None):
    return TraceRecorder(
        conn, run_id=run_id, jsonl_path=jsonl_path or tmp_path / "trace.jsonl"
    )


def test_init_trace_schema_is_idempotent(tmp_path):
    conn = make_conn(tmp_path)
    init_trace_schema(conn)  # 第二次调用不应抛异常
    conn.execute("SELECT run_id, seq, kind FROM trace_events")


def test_record_returns_event_with_incrementing_seq(tmp_path):
    conn = make_conn(tmp_path)
    recorder = make_recorder(tmp_path, conn)
    first = recorder.record("state", "PLANNING", state="PLANNING")
    second = recorder.record(
        "state", "DISPATCHING", prev_state="PLANNING", state="DISPATCHING"
    )
    assert (first.seq, second.seq) == (1, 2)
    assert second.prev_state == "PLANNING"
    assert first.ts > 0


def test_record_rejects_unknown_kind(tmp_path):
    recorder = make_recorder(tmp_path, make_conn(tmp_path))
    with pytest.raises(ValueError, match="kind"):
        recorder.record("telemetry", "x")


def test_record_truncates_detail_and_marks_it(tmp_path):
    recorder = make_recorder(tmp_path, make_conn(tmp_path))
    event = recorder.record("tool", "fetch_chunk", detail="长" * 900)
    assert event.detail.endswith("[truncated]")
    assert len(event.detail) <= recorder.detail_limit + len("…[truncated]")


def test_record_appends_one_jsonl_line_per_event(tmp_path):
    jsonl = tmp_path / "trace.jsonl"
    recorder = make_recorder(tmp_path, make_conn(tmp_path), jsonl_path=jsonl)
    recorder.record("state", "PLANNING", state="PLANNING")
    recorder.record("llm", "plan", model="deepseek-flash", mode="thinking", tokens_in=10)
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["name"] == "plan"


def test_events_returns_events_in_seq_order(tmp_path):
    conn = make_conn(tmp_path)
    recorder = make_recorder(tmp_path, conn)
    for name in ("PLANNING", "DISPATCHING", "WORKING"):
        recorder.record("state", name, state=name)
    assert [event.name for event in recorder.events()] == [
        "PLANNING",
        "DISPATCHING",
        "WORKING",
    ]
    assert [event.seq for event in recorder.events()] == [1, 2, 3]


def test_events_are_isolated_per_run_id(tmp_path):
    conn = make_conn(tmp_path)
    make_recorder(tmp_path, conn, run_id="run-1").record("state", "PLANNING")
    other = make_recorder(tmp_path, conn, run_id="run-2")
    other.record("state", "REFUSE")
    assert [event.name for event in other.events()] == ["REFUSE"]
    assert [event.name for event in other.events("run-1")] == ["PLANNING"]


def test_seq_continues_after_resume(tmp_path):
    conn = make_conn(tmp_path)
    make_recorder(tmp_path, conn).record("state", "PLANNING")
    resumed = make_recorder(tmp_path, conn)  # 同一个 run_id 重新打开
    assert resumed.record("state", "DISPATCHING").seq == 2


def test_summarize_run_aggregates_counts_tokens_and_cost(tmp_path):
    conn = make_conn(tmp_path)
    recorder = make_recorder(tmp_path, conn)
    recorder.record("state", "PLANNING", state="PLANNING", duration_ms=10)
    recorder.record(
        "state", "DISPATCHING", prev_state="PLANNING", state="DISPATCHING", duration_ms=20
    )
    recorder.record(
        "llm",
        "plan",
        model="deepseek-flash",
        mode="thinking",
        tokens_in=100,
        tokens_cached=50,
        tokens_out=30,
        cost_cny=0.01,
        duration_ms=900,
    )
    recorder.record(
        "llm", "synthesize", mode="non-thinking", tokens_in=20, tokens_out=10, cost_cny=0.002
    )
    summary = summarize_run(conn, "run-1")
    assert summary.steps == 2
    assert summary.llm_calls == 2
    assert summary.tokens == 100 + 50 + 30 + 20 + 10
    assert summary.cost_cny == pytest.approx(0.012)


def test_summarize_run_counts_rejections_and_errors(tmp_path):
    conn = make_conn(tmp_path)
    recorder = make_recorder(tmp_path, conn)
    recorder.record("state", "PLANNING", status="rejected", detail="非法转移：PLANNING→DONE")
    recorder.record("tool", "search_documents", status="error", detail="Neo4j 不可达")
    recorder.record("error", "run", status="error", detail="预算耗尽")
    summary = summarize_run(conn, "run-1")
    assert summary.illegal_transitions == 1
    assert summary.error_count == 2


def test_summarize_run_percentiles_ignore_zero_durations(tmp_path):
    conn = make_conn(tmp_path)
    recorder = make_recorder(tmp_path, conn)
    for duration in (100, 100, 100, 100, 500):
        recorder.record("tool", "fetch_chunk", duration_ms=duration)
    recorder.record("note", "noop")  # duration_ms=0，不参与分位数
    summary = summarize_run(conn, "run-1")
    assert summary.p50_ms == 100
    assert summary.p95_ms == 500


def test_summarize_run_on_unknown_run_returns_zeros(tmp_path):
    summary = summarize_run(make_conn(tmp_path), "missing-run")
    assert summary.steps == 0
    assert summary.cost_cny == 0.0
    assert summary.p95_ms == 0
