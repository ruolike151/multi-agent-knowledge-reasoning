import json
from types import SimpleNamespace

import pytest

from eval.taskset.build_candidates import (
    DRAFT_SYSTEM,
    draft_records,
    format_source_block,
    read_chunks,
)
from eval.taskset.schema import EvalTask, TasksetError


def make_chunk(chunk_id, text):
    doc_id = chunk_id.split("#")[0]
    return SimpleNamespace(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        text=text,
    )


class FakeClient:
    """假的模型客户端：记录调用参数，返回预置 JSON 文本。"""

    def __init__(self, payload):
        self._payload = payload
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
        return SimpleNamespace(
            text=json.dumps(self._payload, ensure_ascii=False),
            reasoning="",
            usage=None,
            mode=mode,
            tool_calls=[],
        )


DRAFT_PAYLOAD = {
    "tasks": [
        {
            "task_id": "draft-001",
            "type": "single_hop",
            "difficulty": "easy",
            "question": "甲公司 2024 年营业收入是多少？",
            "gold_answer": "120.5 亿元",
            "gold_chunk_ids": ["doc1#c0003"],
            "answerable": True,
            "notes": "抽取自营业收入表",
        },
        {
            "task_id": "draft-002",
            "type": "refusal",
            "difficulty": "medium",
            "question": "乙公司 2024 年的海外营收是多少？",
            "gold_answer": "无法从知识库确定",
            "gold_chunk_ids": [],
            "answerable": False,
            "notes": "实体替换：语料中没有该公司",
        },
    ]
}


def test_format_source_block_marks_chunk_ids(tmp_path):
    block = format_source_block(
        [make_chunk("doc1#c0003", "营业收入 120.5 亿元"), make_chunk("doc2#c0000", "净利润 8 亿元")]
    )
    assert "doc1#c0003" in block
    assert "营业收入 120.5 亿元" in block
    assert "doc2#c0000" in block


def test_format_source_block_truncates_and_flags_truncation():
    block = format_source_block([make_chunk("doc1#c0000", "长" * 500)], max_chars=100)
    assert len(block) < 400
    assert "截断" in block


def test_format_source_block_accepts_dicts_from_chunks_jsonl(tmp_path):
    """CLI 的真实路径：``read_chunks`` 吐出的是 SQLite 导出的 dict，不是对象。

    2026-09-25 实测：``main()`` 里 ``format_source_block(read_chunks(...))`` 直接抛
    ``AttributeError: 'dict' object has no attribute 'chunk_id'``——单测一直用
    ``SimpleNamespace``，所以这条路径从来没被覆盖过。
    """

    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "doc1#c0003",
                "doc_id": "doc1",
                "source": "doc1.pdf",
                "text": "营业收入 120.5 亿元",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    block = format_source_block(read_chunks(path))
    assert "doc1#c0003" in block
    assert "营业收入 120.5 亿元" in block


def test_draft_records_uses_extract_node_and_forbids_tools():
    client = FakeClient(DRAFT_PAYLOAD)
    draft_records(client, "block", kind="single_hop", limit=2)
    call = client.calls[0]
    assert call["mode"] == "non-thinking"
    assert call["tools"] is None
    assert call["json_object"] is True
    assert call["system"] == DRAFT_SYSTEM


def test_draft_records_system_prompt_is_stable_prefix():
    client = FakeClient(DRAFT_PAYLOAD)
    draft_records(client, "block-1", kind="single_hop", limit=1)
    draft_records(client, "block-2", kind="multi_hop", limit=1)
    assert client.calls[0]["system"] == client.calls[1]["system"]


def test_draft_records_returns_validated_eval_tasks():
    tasks = draft_records(FakeClient(DRAFT_PAYLOAD), "block", kind="single_hop", limit=5)
    assert [task.task_id for task in tasks] == ["draft-001", "draft-002"]
    assert all(isinstance(task, EvalTask) for task in tasks)
    assert tasks[1].answerable is False


def test_draft_records_respects_limit():
    tasks = draft_records(FakeClient(DRAFT_PAYLOAD), "block", kind="single_hop", limit=1)
    assert len(tasks) == 1


def test_draft_records_rejects_invalid_record():
    payload = {"tasks": [dict(DRAFT_PAYLOAD["tasks"][0], type="unknown")]}
    with pytest.raises(TasksetError, match="draft"):
        draft_records(FakeClient(payload), "block", kind="single_hop", limit=5)


def test_draft_records_requires_tasks_key():
    with pytest.raises(TasksetError, match="tasks"):
        draft_records(FakeClient({"items": []}), "block", kind="single_hop", limit=5)


def test_draft_records_applies_task_id_prefix():
    tasks = draft_records(
        FakeClient(DRAFT_PAYLOAD), "block", kind="single_hop", limit=2, task_id_prefix="run1"
    )
    assert [task.task_id for task in tasks] == ["run1-001", "run1-002"]
