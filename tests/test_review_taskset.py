"""`scripts/review_taskset.py` 的单测（人工复核 30 条任务集的那件工具）。

本文件的重点不是「打印得好不好看」，而是三条不许松的契约：

1. gold 引用找不到**必须报错退出**，不能静默跳过（否则复核过的题数会掺假）；
2. 题序必须与任务集文件顺序一致（人工按顺序复核，跳序会漏题）；
3. 直接跑 `python scripts\review_taskset.py` 不能因为 `eval/` 不在 `src/` 而 import 失败。

全部用例走临时 JSONL + `importlib` 加载，**零真实 API 调用、零 Neo4j 依赖**。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval.taskset.schema import TasksetError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "review_taskset.py"


def _load():
    spec = importlib.util.spec_from_file_location("review_taskset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # 不注册会踩 dataclass 的 sys.modules[A] 坑（见偏差日志第二十四节）
    spec.loader.exec_module(module)
    return module


review_taskset = _load()


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _chunk(chunk_id: str, text: str, *, doc_id: str = "annual-600438-2025", page: int = 12) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source": f"data/corpus/annual/{doc_id}.pdf",
        "ordinal": 1,
        "page_start": page,
        "page_end": page,
        "section_path": "第三节 管理层讨论与分析",
        "text": text,
    }


def _task(
    task_id: str,
    *,
    task_type: str = "single_hop",
    question: str = "2024 年营业收入是多少？",
    gold_answer: str = "1234 万元",
    gold_chunk_ids: list[str] | None = None,
    answerable: bool = True,
    notes: str = "",
    difficulty: str = "easy",
) -> dict:
    return {
        "task_id": task_id,
        "type": task_type,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": gold_answer,
        "gold_chunk_ids": ["c1"] if gold_chunk_ids is None else gold_chunk_ids,
        "answerable": answerable,
        "notes": notes,
    }


@pytest.fixture()
def inputs(tmp_path):
    """一份两 chunk 语料 + 一条可答任务，返回 (taskset, chunks) 路径工厂。"""

    chunks_path = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [
            _chunk("c1", "公司 2024 年营业收入 1234 万元，同比增长 12%。"),
            _chunk("c2", "公司 2025 年营业收入 1500 万元。", page=13),
        ],
    )

    def make(rows: list[dict], name: str = "tasks.jsonl") -> Path:
        return _write_jsonl(tmp_path / name, rows)

    return make, chunks_path


def test_report_pairs_each_gold_chunk_with_its_original_text(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("core-single-001")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)]) == 0
    out = capsys.readouterr()
    assert "core-single-001" in out.out
    assert "2024 年营业收入是多少？" in out.out
    assert "gold_answer：1234 万元" in out.out
    assert "公司 2024 年营业收入 1234 万元" in out.out  # 原文真的打出来了
    assert "gold 引用 c1：source=data/corpus/annual/annual-600438-2025.pdf" in out.out
    assert "页码=12-12" in out.out
    assert "第三节 管理层讨论与分析" in out.out
    assert "[done] 复核 1 条题目" in out.err


def test_review_order_follows_taskset_file_order(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("b-task"), _task("a-task"), _task("c-task")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)]) == 0
    out = capsys.readouterr().out
    assert out.index("[b-task]") < out.index("[a-task]") < out.index("[c-task]")


def test_missing_gold_reference_fails_fast_and_is_never_silent(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("core-single-002", gold_chunk_ids=["nope#0001"])])
    code = review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "[缺失] 不在 chunks 文件里" in captured.out
    assert "gold 引用找不到：core-single-002 → nope#0001" in captured.err
    assert "提前修" in captured.err or "先修再复核" in captured.err


def test_missing_reference_does_not_masquerade_as_number_suspicion(inputs, capsys):
    """缺引用时数字必然对不上——此时不许打 [疑点]，否则真问题被噪音盖住。"""

    make, chunks_path = inputs
    taskset = make([_task("core-single-003", gold_chunk_ids=["nope#0001"])])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)]) == 1
    assert "[疑点]" not in capsys.readouterr().out


def test_number_absent_from_gold_text_is_flagged_as_suspicion(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("core-single-004", gold_answer="9999 万元")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)]) == 0
    out = capsys.readouterr().out
    assert "[疑点] gold_answer 里的数字 ['9999']" in out


def test_number_present_in_gold_text_is_not_flagged(inputs, capsys):
    make, chunks_path = inputs
    assert review_taskset.main(["--taskset", str(make([_task("core-single-005")])), "--chunks", str(chunks_path)]) == 0
    assert "[疑点]" not in capsys.readouterr().out


def test_thousands_separator_does_not_create_false_alarm(inputs, capsys):
    """原文写 `1,234.56`、答案写 `1234.56` 是同一件事，别报假疑点。"""

    make, chunks_path = inputs
    chunks_path.write_text(
        json.dumps(_chunk("c1", "营业收入 1,234.56 万元"), ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    taskset = make([_task("core-single-006", gold_answer="1234.56 万元")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)]) == 0
    assert "[疑点]" not in capsys.readouterr().out


def test_refusal_task_prints_three_technique_hint(inputs, capsys):
    make, chunks_path = inputs
    row = _task(
        "core-refusal-001",
        task_type="refusal",
        question="公司 2030 年的光伏出货量是多少？",
        gold_answer="",
        gold_chunk_ids=[],
        answerable=False,
        notes="时间错位：语料只到 2025 年",
    )
    assert review_taskset.main(["--taskset", str(make([row])), "--chunks", str(chunks_path)]) == 0
    out = capsys.readouterr().out
    assert "gold 引用：无" in out
    assert "实体替换 / 时间错位 / 关系不存在" in out
    assert "时间错位：语料只到 2025 年" in out


def test_task_id_filter_and_unknown_id(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("t1"), _task("t2", gold_chunk_ids=["c2"])])
    assert (
        review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--task-id", "t2"]) == 0
    )
    out = capsys.readouterr().out
    assert "[t2]" in out and "[t1]" not in out

    assert (
        review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--task-id", "t9"]) == 1
    )
    assert "任务集里没有这些 task_id：['t9']" in capsys.readouterr().err


def test_limit_truncates_and_zero_is_rejected(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("t1"), _task("t2")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--limit", "1"]) == 0
    assert "[t2]" not in capsys.readouterr().out
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--limit", "0"]) == 1
    assert "--limit 必须 >= 1" in capsys.readouterr().err


def test_expect_core_enforces_quota(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("t1")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--expect", "core"]) == 1
    assert "core 任务集需要 30 条，实际 1 条" in capsys.readouterr().err


def test_out_writes_lf_only_report(inputs, tmp_path, capsys):
    make, chunks_path = inputs
    taskset = make([_task("t1")])
    target = tmp_path / "nested" / "review.txt"
    assert (
        review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--out", str(target)]) == 0
    )
    assert capsys.readouterr().out == ""  # 落了文件就不再往 stdout 喷
    raw = target.read_bytes()
    assert b"\r\n" not in raw
    assert "gold_answer：1234 万元".encode("utf-8") in raw


def test_snippet_truncation_is_announced(inputs, capsys):
    make, chunks_path = inputs
    taskset = make([_task("t1")])
    assert (
        review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path), "--snippet-chars", "10"]) == 0
    )
    out = capsys.readouterr().out
    assert "此处截断，要全文请调大 --snippet-chars" in out
    assert "原文共" in out


def test_bad_snippet_chars_fails_fast(inputs, capsys):
    make, chunks_path = inputs
    assert review_taskset.main(["--taskset", str(make([_task("t1")])), "--chunks", str(chunks_path), "--snippet-chars", "0"]) == 1
    assert "--snippet-chars 必须 >= 1" in capsys.readouterr().err


def test_missing_input_files_fail_fast(tmp_path, capsys):
    assert review_taskset.main(["--taskset", str(tmp_path / "nope.jsonl")]) == 1
    assert "任务集不存在" in capsys.readouterr().err

    taskset = _write_jsonl(tmp_path / "t.jsonl", [_task("t1")])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(tmp_path / "nope.jsonl")]) == 1
    assert "分块文件不存在" in capsys.readouterr().err


def test_duplicate_chunk_ids_are_rejected(tmp_path):
    path = _write_jsonl(tmp_path / "dup.jsonl", [_chunk("c1", "甲"), _chunk("c1", "乙")])
    with pytest.raises(TasksetError, match="chunk_id 重复"):
        review_taskset.index_chunks(path)


def test_empty_after_filter_is_reported(tmp_path, capsys):
    chunks_path = _write_jsonl(tmp_path / "chunks.jsonl", [_chunk("c1", "甲")])
    taskset = _write_jsonl(tmp_path / "empty.jsonl", [])
    assert review_taskset.main(["--taskset", str(taskset), "--chunks", str(chunks_path)]) == 1
    assert "过滤后没有可复核的题目" in capsys.readouterr().err


def test_cli_runs_without_pytest_pythonpath(tmp_path):
    """脚本直跑时 `eval/` 不在 sys.path 里（`.venv` 只把 `src` 装成可编辑包）——这条守住那个补丁。"""

    chunks_path = _write_jsonl(tmp_path / "chunks.jsonl", [_chunk("c1", "营业收入 1234 万元")])
    taskset = _write_jsonl(tmp_path / "tasks.jsonl", [_task("t1")])
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--taskset", str(taskset), "--chunks", str(chunks_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "[t1]" in result.stdout
    assert "[done] 复核 1 条题目" in result.stderr
