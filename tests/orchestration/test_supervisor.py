import pytest

from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.orchestration.ledger import Citation, CostRecord, SubTask, TaskLedger, WorkerReport
from akr.orchestration.states import State
from akr.orchestration.supervisor import (
    COMPRESS_TOKEN_THRESHOLD,
    MAX_ROUTE_ATTEMPTS,
    PLAN_SYSTEM,
    SYNTHESIZE_SYSTEM,
    build_context,
    build_subtasks,
    collect_citations,
    compress_findings,
    decide_next,
    estimate_tokens,
    plan_question,
    synthesize,
)

PLAN_JSON = (
    '{"answerable": true, "subtasks": ['
    '{"task_id": "t1", "goal": "甲公司的营业收入是多少？", "kind": "retrieve", "depends_on": []},'
    '{"task_id": "t2", "goal": "甲公司与丙公司的供应关系是什么？",'
    ' "kind": "graph", "depends_on": ["t1"]}'
    '], "next_state": "DISPATCHING", "reason": "先检索再走图"}'
)
UNANSWERABLE_JSON = (
    '{"answerable": false, "subtasks": [], "next_state": "REFUSE", "reason": "问题与语料无关"}'
)
ROUTE_JSON = '{"next_state": "VERIFYING", "reason": "已有足够证据"}'
ILLEGAL_ROUTE_JSON = '{"next_state": "DONE", "reason": "想直接结束"}'
SYNTHESIS_JSON = (
    '{"text": "甲公司 2024 年营业收入为 12.3 亿元，其子公司乙公司向丙公司供应原材料。",'
    ' "chunk_ids": ["doc1#chunk-03", "doc2#chunk-01"]}'
)
HALLUCINATED_SYNTHESIS_JSON = (
    '{"text": "甲公司 2024 年营业收入为 12.3 亿元，另有未验证结论。",'
    ' "chunk_ids": ["doc1#chunk-03", "doc9#chunk-99"]}'
)

LONG_FINDING = "甲公司在报告期内实现营业收入 12.3 亿元。" + "，其中主营业务收入占比 98%" * 12


class FakeClient:
    """假的 LLMClient：按调用顺序返回预置文本，记录每次调用的 system 与 messages。"""

    def __init__(self, texts, usage=None):
        self.texts = list(texts)
        self.usage = (
            usage if usage is not None else Usage(hit_tokens=0, miss_tokens=1_000, output_tokens=200)
        )
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append(
            {"system": system, "messages": messages, "mode": mode, "json_object": json_object}
        )
        index = min(len(self.calls) - 1, len(self.texts) - 1)
        return LLMResult(
            text=self.texts[index], reasoning="", usage=self.usage, mode=mode, tool_calls=[]
        )


def make_report(findings, chunk_ids, *, confidence=0.7, open_questions=None):
    citations = [
        Citation(doc_id=chunk_id.split("#")[0], chunk_id=chunk_id, source=f"{chunk_id}.pdf")
        for chunk_id in chunk_ids
    ]
    return WorkerReport(
        findings=list(findings),
        citations=citations,
        confidence=confidence,
        open_questions=list(open_questions or []),
        cost=CostRecord(steps=2, tokens=100, cost_cny=0.001, latency_ms=10),
    )


def make_ledger(*, failed=()):
    """t1 已完成（有报告），t2 已完成，failed 里的子任务故意留成 failed。"""

    ledger = TaskLedger(question="甲公司与丙公司的供应关系是什么？")
    ledger.add(SubTask(task_id="t1", goal="甲公司的营业收入是多少？", kind="retrieve"))
    ledger.add(SubTask(task_id="t2", goal="甲公司与丙公司的供应关系？", kind="graph"))
    ledger.mark("t1", "done")
    ledger.mark("t2", "done")
    for task_id in failed:
        ledger.add(SubTask(task_id=task_id, goal=f"{task_id} 的目标", kind="retrieve"))
        ledger.mark(task_id, "failed")
    return ledger


def test_plan_question_returns_outcome_and_uses_thinking():
    client = FakeClient([PLAN_JSON])
    outcome, result = plan_question(client, "甲公司与丙公司的供应关系是什么？")
    assert client.calls[0]["system"] == PLAN_SYSTEM
    assert client.calls[0]["mode"] == "thinking"
    assert isinstance(result, LLMResult)
    assert outcome.answerable is True
    assert [subtask.task_id for subtask in outcome.subtasks] == ["t1", "t2"]
    assert outcome.subtasks[1].depends_on == ["t1"]


def test_plan_question_prompt_carries_only_the_question():
    client = FakeClient([PLAN_JSON])
    plan_question(client, "甲公司与丙公司的供应关系是什么？")
    call = client.calls[0]
    assert len(call["messages"]) == 1  # 没有对话历史、没有其他 agent 的上下文
    assert call["messages"][0]["content"].strip() == "甲公司与丙公司的供应关系是什么？"


def test_build_subtasks_maps_plan_to_ledger_entries():
    outcome, _ = plan_question(FakeClient([PLAN_JSON]), "甲公司与丙公司的供应关系是什么？")
    subtasks = build_subtasks(outcome)
    assert [subtask.task_id for subtask in subtasks] == ["t1", "t2"]
    assert [subtask.kind for subtask in subtasks] == ["retrieve", "graph"]
    assert [subtask.status for subtask in subtasks] == ["pending", "pending"]
    assert subtasks[1].depends_on == ["t1"]


def test_build_subtasks_rejects_unknown_kind():
    outcome, _ = plan_question(
        FakeClient(['{"answerable": true, "subtasks": ['
                    '{"task_id": "t1", "goal": "查一下", "kind": "search"}],'
                    ' "next_state": "DISPATCHING", "reason": ""}']),
        "问题",
    )
    with pytest.raises(ValueError, match="kind"):
        build_subtasks(outcome)


def test_build_subtasks_rejects_duplicate_task_id():
    outcome, _ = plan_question(
        FakeClient(['{"answerable": true, "subtasks": ['
                    '{"task_id": "t1", "goal": "查 A", "kind": "retrieve"},'
                    '{"task_id": "t1", "goal": "查 B", "kind": "retrieve"}],'
                    ' "next_state": "DISPATCHING", "reason": ""}']),
        "问题",
    )
    with pytest.raises(ValueError, match="唯一"):
        build_subtasks(outcome)


def test_build_subtasks_rejects_dependency_on_unknown_task():
    outcome, _ = plan_question(
        FakeClient(['{"answerable": true, "subtasks": ['
                    '{"task_id": "t1", "goal": "查 A", "kind": "retrieve", "depends_on": ["t9"]}],'
                    ' "next_state": "DISPATCHING", "reason": ""}']),
        "问题",
    )
    with pytest.raises(ValueError, match="depends_on"):
        build_subtasks(outcome)


def test_build_subtasks_returns_empty_for_unanswerable_plan():
    outcome, _ = plan_question(FakeClient([UNANSWERABLE_JSON]), "今天天气如何？")
    assert outcome.answerable is False
    assert outcome.next_state == "REFUSE"
    assert build_subtasks(outcome) == []


def sample_reports():
    return {
        "t1": make_report(["甲公司 2024 年营业收入为 12.3 亿元"], ["doc1#chunk-03"]),
        "t2": make_report(["乙公司向丙公司供应原材料"], ["doc2#chunk-01"]),
    }


def test_decide_next_accepts_legal_target():
    client = FakeClient([ROUTE_JSON])
    decision, results = decide_next(client, make_ledger(), sample_reports(), current=State.WORKING)
    assert decision.next_state == "VERIFYING"
    assert len(results) == 1
    assert client.calls[0]["mode"] == "thinking"


def test_decide_next_retries_illegal_target_and_lists_legal_ones():
    client = FakeClient([ILLEGAL_ROUTE_JSON, ROUTE_JSON])
    decision, results = decide_next(client, make_ledger(), sample_reports(), current=State.WORKING)
    assert decision.next_state == "VERIFYING"
    assert len(results) == 2  # 非法答复被驳回，再问一次
    retry_prompt = client.calls[1]["messages"][0]["content"]
    assert "DONE" in retry_prompt
    assert "WORKING" in retry_prompt and "VERIFYING" in retry_prompt


def test_decide_next_degrades_after_max_route_attempts():
    client = FakeClient([ILLEGAL_ROUTE_JSON])
    decision, results = decide_next(client, make_ledger(), sample_reports(), current=State.WORKING)
    assert decision.next_state == State.DEGRADED.value
    assert len(results) == MAX_ROUTE_ATTEMPTS == 2
    assert len(client.calls) == MAX_ROUTE_ATTEMPTS  # 有上限，不会无限重试
    assert "非法" in decision.reason


def test_decide_next_rejects_unknown_state_name():
    client = FakeClient(['{"next_state": "Plan", "reason": "大小写写错了"}'])
    decision, results = decide_next(client, make_ledger(), sample_reports(), current=State.WORKING)
    assert decision.next_state == State.DEGRADED.value
    assert len(results) == MAX_ROUTE_ATTEMPTS
    assert "Plan" in decision.reason


def test_decide_next_carries_verifier_feedback():
    client = FakeClient(['{"next_state": "REVISE", "reason": "补证据"}'])
    decision, _ = decide_next(
        client,
        make_ledger(),
        sample_reports(),
        current=State.VERIFYING,
        verifier_feedback="第二条断言缺少原文支撑",
    )
    assert decision.next_state == State.REVISE.value
    assert "第二条断言缺少原文支撑" in client.calls[0]["messages"][0]["content"]


def test_estimate_tokens_grows_with_text():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 150) == 100  # 150 / CHARS_PER_TOKEN(1.5)
    assert estimate_tokens("a" * 1500) > estimate_tokens("a" * 150)


def test_compress_findings_keeps_first_sentence_and_chunk_ids():
    compressed = compress_findings(
        "甲公司 2024 年营业收入为 12.3 亿元。其中主营业务收入占比 98%。",
        chunk_ids=["doc1#chunk-03"],
    )
    assert compressed.startswith("甲公司 2024 年营业收入为 12.3 亿元。")
    assert "doc1#chunk-03" in compressed
    assert "其中主营业务收入占比" not in compressed


def test_build_context_under_threshold_keeps_full_findings():
    reports = sample_reports()
    reports["t1"] = make_report([LONG_FINDING], ["doc1#chunk-03"])
    context, compressed = build_context(make_ledger(), reports)
    assert compressed is False
    assert LONG_FINDING in context
    assert "doc1#chunk-03" in context


def test_build_context_compresses_finished_subtasks_over_threshold():
    reports = sample_reports()
    reports["t1"] = make_report([LONG_FINDING], ["doc1#chunk-03"])
    context, compressed = build_context(make_ledger(), reports, token_budget=1)
    assert compressed is True
    assert LONG_FINDING not in context
    assert "甲公司在报告期内实现营业收入 12.3 亿元。" in context
    assert "doc1#chunk-03" in context


def test_build_context_keeps_open_questions_and_failed_subtasks_intact():
    long_question = "需要确认关系方向：" + "细节" * 100
    reports = sample_reports()
    reports["t1"] = make_report([LONG_FINDING], ["doc1#chunk-03"], open_questions=[long_question])
    reports["t3"] = make_report([LONG_FINDING], ["doc3#chunk-01"])
    context, compressed = build_context(make_ledger(failed=("t3",)), reports, token_budget=1)
    assert compressed is True
    assert long_question in context  # open_questions 永不压缩
    assert LONG_FINDING in context  # 失败子任务的信息也不压缩


def test_collect_citations_dedupes_across_reports():
    reports = {
        "t1": make_report(["a"], ["doc1#chunk-03", "doc2#chunk-01"]),
        "t2": make_report(["b"], ["doc2#chunk-01"]),
    }
    citations = collect_citations(make_ledger(), reports)
    assert [citation.chunk_id for citation in citations] == ["doc1#chunk-03", "doc2#chunk-01"]
    assert all(citation.source for citation in citations)


def test_synthesize_returns_draft_with_ledger_citations():
    client = FakeClient([SYNTHESIS_JSON])
    draft, result = synthesize(
        client, "甲公司与丙公司的供应关系是什么？", make_ledger(), sample_reports()
    )
    assert client.calls[0]["system"] == SYNTHESIZE_SYSTEM
    assert client.calls[0]["mode"] == "non-thinking"
    assert isinstance(result, LLMResult)
    assert draft.chunk_ids == ["doc1#chunk-03", "doc2#chunk-01"]
    assert draft.dropped_chunk_ids == []
    assert "营业收入" in draft.text


def test_synthesize_drops_hallucinated_chunk_ids():
    client = FakeClient([HALLUCINATED_SYNTHESIS_JSON])
    draft, _ = synthesize(client, "问题", make_ledger(), sample_reports())
    assert draft.chunk_ids == ["doc1#chunk-03"]
    assert draft.dropped_chunk_ids == ["doc9#chunk-99"]


def test_synthesize_degraded_prompt_marks_uncertainty():
    reports = sample_reports()
    reports["t1"] = make_report(
        ["甲公司 2024 年营业收入为 12.3 亿元"],
        ["doc1#chunk-03"],
        open_questions=["丙公司年报未下载"],
    )
    client = FakeClient([SYNTHESIS_JSON])
    synthesize(client, "问题", make_ledger(), reports, degraded=True)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "降级" in prompt
    assert "丙公司年报未下载" in prompt


# --- P2：口径差异与「检索范围说明」不写成不确定项 --------------------------------
#
# 依据（15 格 v3 重跑 trace；`tmp/probe_p2_hedge.py` 摊开的两题 worker 报告原文）：
# 判分 0 的答案里反复出现「未解决：片段中未出现『归属于上市公司股东的净利润』这一完整
# 表述对应的金额，仅出现『归属于母公司股东的净利润』口径的 -9,263,788,838.65 元，
# 无法确认两者是否为同一口径」——这句话一字不差来自 worker 的 open_questions，被
# Supervisor 渲染进上下文后模型照抄成「不确定事项」。这不是检索没召回到证据，而是
# 同一份年报的两个并列披露口径被读成了「证据缺失」。所以修的是渲染与合成规则，
# 不是再加一轮检索。

CALIBER_FINDING_A = "TCL中环 2025 年归属于上市公司股东的净利润为 -92.64 亿元"
CALIBER_FINDING_B = "TCL中环 2025 年归属于母公司股东的净利润为 -92.64 亿元"


def test_build_context_flags_caliber_differences_within_one_report():
    reports = sample_reports()
    reports["t1"] = make_report([CALIBER_FINDING_A], ["doc1#chunk-03"])
    reports["t2"] = make_report([CALIBER_FINDING_B], ["doc2#chunk-01"])
    context, _ = build_context(make_ledger(), reports)
    assert "归属于上市公司股东的净利润" in context
    assert "归属于母公司股东的净利润" in context
    assert "口径不同不等于证据缺失" in context


def test_build_context_has_no_caliber_note_for_a_single_caliber():
    context, _ = build_context(make_ledger(), sample_reports())
    assert "口径" not in context


def test_build_context_marks_retrieval_scope_notes():
    reports = sample_reports()
    reports["t1"] = make_report(
        ["甲公司 2024 年营业收入为 12.3 亿元"],
        ["doc1#chunk-03"],
        open_questions=["片段中未出现研发投入金额"],
    )
    reports["t2"] = make_report(["检索片段中未出现乙公司的注册资本"], ["doc2#chunk-01"])
    context, _ = build_context(make_ledger(), reports)
    assert context.count("检索范围说明") == 2
    assert "片段中未出现研发投入金额" in context
    assert "检索片段中未出现乙公司的注册资本" in context


def test_build_context_keeps_ordinary_findings_unmarked():
    """关于公司本身的否定事实是正经结论，不能被贴上「检索范围说明」的标签。"""

    reports = sample_reports()
    reports["t1"] = make_report(["甲公司报告期内未发生重大资产重组"], ["doc1#chunk-03"])
    context, _ = build_context(make_ledger(), reports)
    assert "甲公司报告期内未发生重大资产重组" in context
    assert "检索范围说明" not in context


def test_synthesize_system_forbids_hedging_on_caliber_gap():
    assert "口径" in SYNTHESIZE_SYSTEM
    assert "不是证据缺失" in SYNTHESIZE_SYSTEM
    assert "检索范围说明" in SYNTHESIZE_SYSTEM
