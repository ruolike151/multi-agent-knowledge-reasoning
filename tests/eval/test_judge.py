import json

import pytest

from akr.model.client import LLMResult
from akr.model.cost import Usage
from eval.judges import judge as judge_module
from eval.judges.judge import (
    CALIBRATION_SIZE,
    JUDGE_PASS_THRESHOLD,
    SCORE_LEVELS,
    JudgeError,
    build_judge_system,
    calibrate,
    entailment_support,
    judge_answer,
    load_rubric,
)
from eval.taskset.schema import EvalTask


class FakeClient:
    """按顺序吐出预置 payload，并把每次调用的参数记下来；测试里永不触网。"""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "mode": mode,
                "tools": tools,
                "json_object": json_object,
            }
        )
        return LLMResult(
            text=self._payloads.pop(0),
            reasoning="",
            usage=Usage(hit_tokens=10, miss_tokens=20, output_tokens=5),
            mode=mode,
            tool_calls=[],
        )


def make_task(task_id="single_hop-001", **overrides):
    payload = {
        "task_id": task_id,
        "type": "single_hop",
        "difficulty": "easy",
        "question": "甲公司 2023 年营业收入是多少？",
        "gold_answer": "100 亿元",
        "gold_chunk_ids": ["doc1#chunk-03"],
        "answerable": True,
        "notes": "",
    }
    payload.update(overrides)
    return EvalTask(**payload)


def verdict_payload(score=2, supported=True, reason="与 gold 引用一致"):
    return json.dumps({"score": score, "supported": supported, "reason": reason})


def test_load_rubric_reads_disk_and_fails_loudly_when_missing(tmp_path, monkeypatch):
    """rubric 是实验条件的一部分，读不到就必须炸，不能静默用空 rubric 继续打分。"""
    rubric_path = tmp_path / "rubric.md"
    rubric_path.write_text("# 评分标准\n\n2 分：完整回答且全部断言有引用支撑。\n", encoding="utf-8")
    monkeypatch.setattr(judge_module, "RUBRIC_PATH", rubric_path)

    rubric = load_rubric()

    assert "完整回答且全部断言有引用支撑" in rubric

    rubric_path.unlink()
    with pytest.raises(JudgeError, match="rubric"):
        load_rubric()


def test_build_judge_system_embeds_rubric_verbatim():
    rubric = "# 独占标记 RUBRIC-MARKER-42\n只按此标准打分。\n"

    system = build_judge_system(rubric)

    assert "RUBRIC-MARKER-42" in system
    assert build_judge_system(rubric) == system          # 同 rubric → 同前缀（前缀缓存友好）
    assert build_judge_system("别的标准").index("别的标准") > 0


def test_judge_answer_sends_rubric_thinking_and_structured_flag():
    client = FakeClient([verdict_payload(score=2, supported=True)])
    task = make_task()

    verdict, result = judge_answer(
        client,
        task,
        "营业收入为 100 亿元。",
        evidence="[gold 答案] 100 亿元\n[引用 doc1#chunk-03] ……营业收入 100 亿元……",
    )

    (call,) = client.calls
    assert verdict.score == 2
    assert verdict.supported is True
    assert verdict.task_id == "single_hop-001"
    assert call["mode"] == "thinking"                 # 主观打分走 thinking
    assert call["json_object"] is True                # 结构化输出，便于校验与修复
    assert call["system"] == judge_module.JUDGE_SYSTEM
    assert "rubric" in call["system"].lower() or "评分标准" in call["system"]
    assert "营业收入为 100 亿元" in json.dumps(call["messages"], ensure_ascii=False)
    assert result.usage.total_tokens == 35


def test_judge_answer_rejects_out_of_range_score_instead_of_clamping():
    """越界分数必须抛错：clamp 会把"judge 看不懂题"悄悄变成"部分正确"。"""
    client = FakeClient([verdict_payload(score=3)])

    with pytest.raises(JudgeError, match="score"):
        judge_answer(client, make_task(), "任意答案", evidence="任意证据")

    assert SCORE_LEVELS == (0, 1, 2)
    assert JUDGE_PASS_THRESHOLD == 1


def test_judge_answer_rejects_malformed_json():
    client = FakeClient(["我觉得这条回答不错，给 2 分。"])

    with pytest.raises(JudgeError):
        judge_answer(client, make_task(), "任意答案", evidence="任意证据")


def test_judge_answer_truncates_oversized_evidence():
    """证据超长要截断到 MAX_EVIDENCE_CHARS，防止单次 judge 调用把预算吃光。

    探针字符用「囧」而不是「证」：rubric 与 system prompt 里本来就有"证据"这类词，
    用常见字去数出现次数会数到自己身上。
    """
    client = FakeClient([verdict_payload()])

    judge_answer(client, make_task(), "答案", evidence="囧" * 5000)

    sent = json.dumps(client.calls[0]["messages"], ensure_ascii=False)
    assert sent.count("囧") == judge_module.MAX_EVIDENCE_CHARS


def test_entailment_support_is_non_thinking_and_returns_bool():
    client = FakeClient([json.dumps({"supported": False})])

    supported, result = entailment_support(client, "乙公司是甲公司子公司。", "……甲公司持有乙公司 60% 股权。")

    assert supported is False
    assert client.calls[0]["mode"] == "non-thinking"   # 机械判断走 non-thinking
    assert result.text.startswith("{")


def test_entailment_support_rejects_missing_field():
    client = FakeClient([json.dumps({"verdict": "yes"})])

    with pytest.raises(JudgeError):
        entailment_support(client, "断言", "原文")


def calibration_inputs(tasks, *, humans):
    """校准集的三份输入：人工标签（前置写好）、系统答案、gold 证据。"""
    return {
        "human_labels": humans,
        "answers": {task.task_id: "营业收入 100 亿元。" for task in tasks},
        "evidence_by_task": {task.task_id: "……营业收入 100 亿元……" for task in tasks},
    }


def test_calibrate_reports_perfect_agreement_and_kappa():
    tasks = [make_task(task_id=f"single_hop-00{i}") for i in range(1, 5)]
    client = FakeClient([verdict_payload(score=2)] * 4)
    humans = {task.task_id: True for task in tasks}

    report = calibrate(client, tasks, **calibration_inputs(tasks, humans=humans))

    assert report.total == 4
    assert report.agreement_rate == 1.0
    assert report.kappa == 1.0              # 常量标注且完全一致 → κ 记 1.0（Task 33 口径）
    assert report.divergences == ()
    assert report.judge_pass_rate == 1.0
    assert report.human_pass_rate == 1.0
    assert report.cost_cny > 0             # 校准本身的成本也要入账（spec §7.7 预算含 judge）


def test_calibrate_kappa_matches_task33_formula_and_lists_divergences():
    tasks = [make_task(task_id=f"single_hop-00{i}") for i in range(1, 5)]
    # judge 分数 [2, 0, 2, 0] → 二值化 [通过, 不通过, 通过, 不通过]
    client = FakeClient(
        [
            verdict_payload(score=2),
            verdict_payload(score=0),
            verdict_payload(score=2),
            verdict_payload(score=0),
        ]
    )
    humans = {
        tasks[0].task_id: True,
        tasks[1].task_id: True,
        tasks[2].task_id: False,
        tasks[3].task_id: False,
    }

    report = calibrate(client, tasks, **calibration_inputs(tasks, humans=humans))

    # po = 0.5、两类各半 pe = 0.5 → κ = 0；与 Task 33 的 cohen_kappa 是同一套算法
    assert report.agreement_rate == pytest.approx(0.5)
    assert report.kappa == pytest.approx(0.0)
    assert [item.task_id for item in report.divergences] == [
        tasks[1].task_id,
        tasks[2].task_id,
    ]
    assert report.divergences[0].human_passed is True
    assert report.divergences[0].judge_passed is False
    assert report.divergences[0].score == 0          # 分数留着，才知道分歧是"差一档"还是"完全相反"


def test_calibrate_refuses_to_start_without_full_human_labels():
    """人工标签缺一条就必须在花任何模型调用之前停下（fail-closed）。"""
    tasks = [make_task(task_id=f"single_hop-00{i}") for i in range(1, 3)]
    client = FakeClient([verdict_payload()] * 2)
    inputs = calibration_inputs(tasks, humans={tasks[0].task_id: True})

    with pytest.raises(JudgeError, match="human_labels"):
        calibrate(client, tasks, **inputs)

    assert client.calls == []                        # 一条调用都没发出去


def test_calibration_size_matches_spec():
    """spec §9.5 写死 15 题校准集；样本量写进代码常量，报告层据此核对。"""
    assert CALIBRATION_SIZE == 15
