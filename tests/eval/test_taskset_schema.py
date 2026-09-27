import json

import pytest

from eval.taskset.schema import (
    CORE_QUOTAS,
    EXTENDED_QUOTAS,
    EvalTask,
    TasksetError,
    dump_taskset,
    load_taskset,
    validate_taskset,
)


def sample(**overrides):
    payload = {
        "task_id": "single_hop-001",
        "type": "single_hop",
        "difficulty": "easy",
        "question": "甲公司 2024 年的营业收入是多少？",
        "gold_answer": "人民币 120.5 亿元",
        "gold_chunk_ids": ["doc1#c0003"],
        "answerable": True,
        "notes": "",
    }
    payload.update(overrides)
    return payload


def write_jsonl(path, records):
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_tasks(quotas):
    tasks = []
    for task_type, count in quotas.items():
        for index in range(count):
            answerable = task_type != "refusal"
            tasks.append(
                EvalTask.model_validate(
                    sample(
                        task_id=f"{task_type}-{index:03d}",
                        type=task_type,
                        answerable=answerable,
                        gold_chunk_ids=["doc1#c0001"] if answerable else [],
                    )
                )
            )
    return tasks


def test_record_round_trips_fields_and_unicode(tmp_path):
    path = write_jsonl(tmp_path / "tasks.jsonl", [sample()])
    (task,) = load_taskset(path)
    assert task.question == "甲公司 2024 年的营业收入是多少？"
    assert task.gold_chunk_ids == ["doc1#c0003"]
    assert task.answerable is True


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(sample(), ensure_ascii=False) + "\n\n   \n",
        encoding="utf-8",
    )
    assert len(load_taskset(path)) == 1


def test_invalid_json_reports_file_and_line_number(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(sample(), ensure_ascii=False) + "\n{not json}\n",
        encoding="utf-8",
    )
    with pytest.raises(TasksetError, match=r"tasks\.jsonl:2"):
        load_taskset(path)


def test_missing_field_reports_file_and_line_number(tmp_path):
    payload = sample()
    payload.pop("question")
    path = write_jsonl(tmp_path / "tasks.jsonl", [payload])
    with pytest.raises(TasksetError, match=r"tasks\.jsonl:1"):
        load_taskset(path)


def test_unknown_type_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "tasks.jsonl", [sample(type="single-hop")])
    with pytest.raises(TasksetError, match="type"):
        load_taskset(path)


def test_extra_field_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "tasks.jsonl", [sample(gold_spans=["x"])])
    with pytest.raises(TasksetError):
        load_taskset(path)


def test_refusal_must_be_unanswerable_and_have_no_gold_chunks(tmp_path):
    path = write_jsonl(
        tmp_path / "tasks.jsonl",
        [sample(task_id="refusal-001", type="refusal", answerable=True)],
    )
    with pytest.raises(TasksetError, match="refusal"):
        load_taskset(path)


def test_answerable_task_requires_gold_chunk_ids(tmp_path):
    path = write_jsonl(tmp_path / "tasks.jsonl", [sample(gold_chunk_ids=[])])
    with pytest.raises(TasksetError, match="gold_chunk_ids"):
        load_taskset(path)


def test_duplicate_task_id_is_rejected(tmp_path):
    path = write_jsonl(tmp_path / "tasks.jsonl", [sample(), sample()])
    with pytest.raises(TasksetError, match="重复 task_id"):
        load_taskset(path)


def test_validate_core_accepts_exact_quota():
    validate_taskset(make_tasks(CORE_QUOTAS), expect="core")  # 不抛异常即为通过


def test_validate_core_rejects_wrong_total_and_reports_mix():
    tasks = make_tasks(CORE_QUOTAS)[:-1]
    with pytest.raises(TasksetError, match="30"):
        validate_taskset(tasks, expect="core")


def test_validate_core_rejects_wrong_mix():
    tasks = make_tasks(CORE_QUOTAS)
    tasks[-1] = EvalTask.model_validate(
        sample(task_id="single_hop-099", type="single_hop", answerable=True)
    )
    with pytest.raises(TasksetError, match="refusal"):
        validate_taskset(tasks, expect="core")


def test_validate_extended_requires_80_tasks():
    validate_taskset(make_tasks(EXTENDED_QUOTAS), expect="extended")
    with pytest.raises(TasksetError, match="80"):
        validate_taskset(make_tasks(CORE_QUOTAS), expect="extended")


def test_validate_rejects_unknown_expect():
    with pytest.raises(TasksetError, match="expect"):
        validate_taskset([], expect="smoke")


def test_dump_taskset_writes_one_compact_line_per_task(tmp_path):
    path = tmp_path / "out.jsonl"
    dump_taskset(make_tasks(CORE_QUOTAS), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    assert json.loads(lines[0])["task_id"] == "single_hop-000"
    assert "甲公司" in lines[0]  # ensure_ascii=False：中文不被转义，diff 可读
