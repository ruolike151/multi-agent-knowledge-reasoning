import pytest

from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.orchestration.supervisor import DraftAnswer
from akr.orchestration.verifier import (
    EVIDENCE_CHARS,
    MAX_EVIDENCE_BLOCKS,
    MAX_REVISE_ROUNDS,
    REASON_CHARS,
    VERIFY_SYSTEM,
    ClaimVerdict,
    VerifyResult,
    VerifyOutput,
    should_revise,
    split_claims,
    verify_answer,
)

DRAFT = DraftAnswer(
    text="甲公司 2024 年营业收入为 12.3 亿元。公司的市占率超过 30%。",
    chunk_ids=["doc1#chunk-03", "doc2#chunk-01"],
)

EVIDENCE = {
    "doc1#chunk-03": "2024 年公司实现营业收入 12.3 亿元，同比增长 8%",
    "doc2#chunk-01": "本公司持有乙公司 100% 股权，乙公司纳入合并报表范围",
}

# 模型侧只回**序号**，不回抄断言原文（见 `VERIFY_SYSTEM`：回抄是纯输出浪费，
# 而且回抄一改字就再也对不回原断言）。断言正文由 `verify_answer` 按序号还原。
VERDICTS_JSON = (
    '{"verdicts": ['
    '{"index": 1, "verdict": "supported",'
    ' "reason": "原文明确写出营业收入 12.3 亿元", "chunk_ids": ["doc1#chunk-03"]},'
    '{"index": 2, "verdict": "unsupported",'
    ' "reason": "原文没有任何市占率数据", "chunk_ids": []}'
    "]}"
)
ALL_SUPPORTED_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "supported",'
    ' "reason": "原文明确", "chunk_ids": ["doc1#chunk-03"]}]}'
)
SUPPORTED_WITHOUT_EVIDENCE_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "supported",'
    ' "reason": "看起来合理", "chunk_ids": ["doc9#chunk-99"]}]}'
)
UNKNOWN_VERDICT_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "SUPPORTED",'
    ' "reason": "大小写写错了", "chunk_ids": []}]}'
)


class FakeClient:
    """假的 LLMClient：按调用顺序返回预置文本，绝不发真实请求。"""

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


def make_fetcher(evidence=None, calls=None):
    """假的原文取回函数：Task 31 会把它接到 artifact store 上。"""

    table = dict(EVIDENCE if evidence is None else evidence)

    def fetch(chunk_id):
        if calls is not None:
            calls.append(chunk_id)
        return table.get(chunk_id, "")

    return fetch


def test_split_claims_splits_on_chinese_punctuation():
    claims = split_claims("甲公司营业收入为 12.3 亿元。公司市占率超过 30%；研发投入 1.2 亿元")
    assert claims == ["甲公司营业收入为 12.3 亿元", "公司市占率超过 30%", "研发投入 1.2 亿元"]


def test_split_claims_keeps_decimal_points_intact():
    assert split_claims("营业收入 12.3 亿元,同比增长 8.5%") == ["营业收入 12.3 亿元,同比增长 8.5%"]


def test_split_claims_drops_short_fragments():
    assert split_claims("好。甲公司营业收入为 12.3 亿元。对") == ["甲公司营业收入为 12.3 亿元"]


def test_split_claims_handles_empty_input():
    assert split_claims("") == []
    assert split_claims("  \n   ") == []


def test_verify_answer_uses_thinking_and_isolated_prompt():
    client = FakeClient([VERDICTS_JSON])
    verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    call = client.calls[0]
    assert call["system"] == VERIFY_SYSTEM
    assert call["mode"] == "thinking"
    assert len(call["messages"]) == 1  # 看不到生成过程，也看不到 Supervisor 的推理
    content = call["messages"][0]["content"]
    assert "甲公司 2024 年营业收入为 12.3 亿元" in content  # 断言逐条列出
    assert "公司的市占率超过 30%" in content
    assert "2024 年公司实现营业收入 12.3 亿元，同比增长 8%" in content  # 喂原文正文
    assert "doc1#chunk-03" in content and "doc2#chunk-01" in content


def test_verify_answer_collects_verdicts():
    client = FakeClient([VERDICTS_JSON])
    result, raw = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert [verdict.verdict for verdict in result.verdicts] == ["supported", "unsupported"]
    assert result.has_unsupported is True
    assert isinstance(raw, LLMResult)


def test_verify_answer_fetches_original_text_by_chunk_id():
    calls = []
    client = FakeClient([VERDICTS_JSON])
    verify_answer(client, DRAFT, fetch_evidence=make_fetcher(calls=calls))
    assert calls == ["doc1#chunk-03", "doc2#chunk-01"]


def test_evidence_is_truncated_to_evidence_chars():
    long_text = "甲" * EVIDENCE_CHARS + "尾部标记"
    client = FakeClient([VERDICTS_JSON])
    verify_answer(
        client,
        DRAFT,
        fetch_evidence=make_fetcher({"doc1#chunk-03": long_text, "doc2#chunk-01": long_text}),
    )
    content = client.calls[0]["messages"][0]["content"]
    assert "尾部标记" not in content
    assert "甲" * 20 in content


def test_missing_evidence_is_marked_in_prompt():
    client = FakeClient([SUPPORTED_WITHOUT_EVIDENCE_JSON])
    verify_answer(
        client,
        DraftAnswer(text="公司的市占率超过 30%。", chunk_ids=["doc9#chunk-99"]),
        fetch_evidence=make_fetcher(),
    )
    content = client.calls[0]["messages"][0]["content"]
    assert "（未找到原文" in content


def test_missing_evidence_downgrades_supported_verdict():
    client = FakeClient([SUPPORTED_WITHOUT_EVIDENCE_JSON])
    result, _ = verify_answer(
        client,
        DraftAnswer(text="公司的市占率超过 30%。", chunk_ids=["doc9#chunk-99"]),
        fetch_evidence=make_fetcher(),
    )
    verdict = result.verdicts[0]
    assert verdict.verdict == "partially_supported"
    assert "证据缺失" in verdict.reason
    assert result.has_unsupported is False  # 降级不是 unsupported，不该凭它触发 REVISE


def test_unknown_verdict_is_treated_as_unsupported():
    client = FakeClient([UNKNOWN_VERDICT_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    verdict = result.verdicts[0]
    assert verdict.verdict == "unsupported"
    assert "SUPPORTED" in verdict.reason
    assert result.has_unsupported is True


def test_feedback_lists_only_unpassed_claims():
    client = FakeClient([VERDICTS_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert "公司的市占率超过 30%" in result.feedback
    assert "原文没有任何市占率数据" in result.feedback
    assert "甲公司 2024 年营业收入为 12.3 亿元" not in result.feedback


def test_feedback_is_empty_when_all_supported():
    client = FakeClient([ALL_SUPPORTED_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert result.has_unsupported is False
    assert result.feedback == ""


def test_should_revise_true_when_unsupported_within_rounds():
    client = FakeClient([VERDICTS_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert should_revise(result, revise_round=0) is True
    assert should_revise(result, revise_round=MAX_REVISE_ROUNDS - 1) is True


def test_should_revise_false_when_round_limit_reached():
    client = FakeClient([VERDICTS_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert should_revise(result, revise_round=MAX_REVISE_ROUNDS) is False


def test_should_revise_false_when_only_partially_supported():
    result = VerifyResult(
        verdicts=[ClaimVerdict(claim="公司的市占率超过 30%", verdict="partially_supported", reason="缺原文")],
        has_unsupported=False,
        feedback="公司的市占率超过 30%：缺原文（partially_supported）",
    )
    assert should_revise(result, revise_round=0) is False


# --- P1 瘦身：模型回序号、reason 限长、证据块封顶、纠错轮次封顶 -------------------
#
# 依据（15 格 v3 重跑 trace，`tmp/join-half15-v3rerun.json`）：verify 37 次调用吃掉
# 全部 LLM 花费的 70%，输入 130k、输出 325k token。输出的大头是 thinking 的推理
# token，本轮刻意不动（spec §7.2 要求逐断言 thinking）。这里改的是三处纯浪费
# 与一处过度回退：
#  1) 让模型原样抄回断言，等于拿输出 token 换一堆下游对不回原文的字符串；
#  2) reason 不限长，模型会写整段；
#  3) 证据块数量不设上限，而全库 6,348 个原文块只有 0.7% 超过 EVIDENCE_CHARS，
#     真正的量级来自「喂了几块」；
#  4) MAX_REVISE_ROUNDS=2 意味着单题最多 3 次 verify，而实测均值已是 2.47 次/题。


OUT_OF_RANGE_INDEX_JSON = (
    '{"verdicts": [{"index": 9, "verdict": "unsupported",'
    ' "reason": "编号根本不存在", "chunk_ids": []}]}'
)
DUPLICATE_INDEX_JSON = (
    '{"verdicts": ['
    '{"index": 1, "verdict": "unsupported", "reason": "先判不通过", "chunk_ids": []},'
    '{"index": 1, "verdict": "supported", "reason": "又改口说通过",'
    ' "chunk_ids": ["doc1#chunk-03"]}'
    "]}"
)
LONG_REASON_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "unsupported",'
    ' "reason": "' + "啰" * (REASON_CHARS * 2) + '", "chunk_ids": []}]}'
)
# 引用一个「草稿里存在、但被证据块上限挡在门外」的编号（第 13 个块）。
TAIL_CITED_JSON = (
    '{"verdicts": [{"index": 1, "verdict": "supported",'
    ' "reason": "文档里写了", "chunk_ids": ["doc13#chunk-01"]}]}'
)


def test_verified_claim_text_is_restored_from_index():
    """模型只回序号，断言正文由代码按序号还原，避免回抄一改字就对不回原断言。"""

    client = FakeClient([VERDICTS_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert [verdict.claim for verdict in result.verdicts] == [
        "甲公司 2024 年营业收入为 12.3 亿元",
        "公司的市占率超过 30%",
    ]


def test_out_of_range_index_is_dropped():
    """序号越界说明模型没在回答「哪条断言」：不能凭空造一条出来，直接丢弃。"""

    client = FakeClient([OUT_OF_RANGE_INDEX_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert result.verdicts == []
    assert result.has_unsupported is False
    assert result.feedback == ""


def test_duplicate_index_keeps_the_first_verdict():
    """同一序号回两条判定时取第一条：结果必须确定，不随模型输出顺序摆动。"""

    client = FakeClient([DUPLICATE_INDEX_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    assert len(result.verdicts) == 1
    assert result.verdicts[0].verdict == "unsupported"
    assert result.verdicts[0].claim == "甲公司 2024 年营业收入为 12.3 亿元"


def test_reason_is_truncated_to_reason_chars():
    client = FakeClient([LONG_REASON_JSON])
    result, _ = verify_answer(client, DRAFT, fetch_evidence=make_fetcher())
    reason = result.verdicts[0].reason
    assert reason.endswith("…")
    assert len(reason) == REASON_CHARS + 1


def test_verify_system_does_not_ask_for_claim_echo():
    assert "原样抄回" not in VERIFY_SYSTEM
    assert "index" in VERIFY_SYSTEM


def test_evidence_blocks_are_capped_and_the_tail_counts_as_missing():
    """证据块数量封顶；没喂进去的编号按「证据缺失」处理，不能拿来支撑断言。"""

    chunk_ids = [f"doc{i}#chunk-01" for i in range(MAX_EVIDENCE_BLOCKS + 3)]
    evidence = {chunk_id: f"{chunk_id} 的正文" for chunk_id in chunk_ids}
    client = FakeClient([TAIL_CITED_JSON])
    result, _ = verify_answer(
        client,
        DraftAnswer(text="甲公司的营业收入是 12.3 亿元，依据 doc13#chunk-01。", chunk_ids=chunk_ids),
        fetch_evidence=make_fetcher(evidence),
    )
    content = client.calls[0]["messages"][0]["content"]
    assert content.count("[doc") == MAX_EVIDENCE_BLOCKS
    assert chunk_ids[MAX_EVIDENCE_BLOCKS] not in content
    # 尾部的编号按「证据缺失」处理：模型若引用它们，不能算 supported。
    assert result.verdicts[0].verdict == "partially_supported"
    assert "证据缺失" in result.verdicts[0].reason


def test_verdict_citing_unknown_chunk_id_is_downgraded():
    """引用草稿里根本不存在的编号 = 编引用，不能算 supported（光靠 prompt 约束不住）。"""

    client = FakeClient([SUPPORTED_WITHOUT_EVIDENCE_JSON])
    result, _ = verify_answer(
        client,
        DraftAnswer(text="公司的市占率超过 30%。", chunk_ids=["doc2#chunk-01"]),
        fetch_evidence=make_fetcher(),
    )
    verdict = result.verdicts[0]
    assert verdict.verdict == "partially_supported"
    assert "doc9#chunk-99" in verdict.reason
    # 降级不是 unsupported，因此不会凭它触发 REVISE。
    assert result.has_unsupported is False


def test_max_revise_rounds_is_one():
    """单题最多 2 次 verify（首判 + 一轮纠错）：实测均值已 2.47 次/题，第二轮收益低。"""

    assert MAX_REVISE_ROUNDS == 1


def test_verify_output_schema_has_no_claim_field():
    """回包模型只描述「序号 → 判定」，不回抄断言正文。"""

    fields = VerifyOutput.model_fields
    assert set(fields) == {"verdicts"}
    verdict_fields = list(VerifyOutput.model_fields["verdicts"].annotation.__args__[0].model_fields)
    assert "claim" not in verdict_fields
    assert "index" in verdict_fields
