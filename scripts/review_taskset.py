r"""把每道题与它的 gold 原文并排打出来，供人工复核（交接文档 §7.2 第 4 条）。

人工复核 30 条核心任务集是评测可信度的根，但要复核首先得看得见东西：`EvalTask` 里只存
`gold_chunk_ids`，原文躺在 `data/chunks.jsonl` 的 6176 条 chunk 里。本脚本把两者拼起来，
一屏之内就能判断「这句 gold_answer 是不是真被引用的 chunk 撑着」。

用法：

    .\.venv\Scripts\python.exe scripts\review_taskset.py                  # 复核 core.jsonl 全部题目
    .\.venv\Scripts\python.exe scripts\review_taskset.py --task-id core-single-001
    .\.venv\Scripts\python.exe scripts\review_taskset.py --out tmp/r.md   # 落成文件慢慢看
    .\.venv\Scripts\python.exe scripts\review_taskset.py --expect core    # 顺带校验 30 条与类型配比

三道护栏（都是为了「复核过的题数不掺假」）：

1. gold_chunk_id 在 chunks 文件里找不到 → 该条打 `[缺失]` 并在 stderr 汇总，退出码 1。
   **绝不静默跳过**：漏掉一条引用而照常退出 0，人工就以为那题已经复核过了。
2. `--expect core|extended` 会调用 `validate_taskset` 校验条数与类型配比，配比不对直接报错。
3. gold_answer 里的数字若一条都没出现在任何 gold 原文里 → 打 `[疑点]`。这只是提示不是判定：
   数字可能是换算、汇总或百分比换算出来的，由人工定夺。

退出码：0 = 选中的题目 gold 引用全部落到原文上；1 = 任务集 / 分块文件读不动、gold 引用找不到、
`--expect` 配比不符、过滤后没有题目可复核。

与 `build_candidates.py` 的关系：那边**出题**（从 chunk 反向构造草稿），这边**验题**。两边都不
调用被测系统——「自己出题自己答」是评测最常见的可信度漏洞。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# `eval/` 不在可编辑安装的 `src/` 里（`.venv` 只有 `__editable__.akr-0.1.0.pth` 指向 src），
# 所以直接跑脚本时 sys.path[0] 是 `scripts/`，`import eval.taskset...` 会 ModuleNotFoundError
# （2026-09-25 实测）。把仓库根补进 sys.path，才能复用 runner 读任务集与 chunk 的同一套实现。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.taskset.build_candidates import read_chunks  # noqa: E402
from eval.taskset.schema import (  # noqa: E402
    QUOTA_TABLES,
    EvalTask,
    TasksetError,
    load_taskset,
    validate_taskset,
)

DEFAULT_TASKSET = ROOT / "eval" / "taskset" / "core.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "chunks.jsonl"
DEFAULT_SNIPPET_CHARS = 400
RULE = "=" * 78
NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def index_chunks(path: str | Path) -> dict[str, dict[str, Any]]:
    """读 chunk JSONL，建 `chunk_id -> 记录` 索引；重复 chunk_id 直接报错（不许后写覆盖前写）。"""

    index: dict[str, dict[str, Any]] = {}
    for record in read_chunks(path):
        chunk_id = record["chunk_id"]
        if chunk_id in index:
            raise TasksetError(f"{path} 里 chunk_id 重复：{chunk_id}")
        index[chunk_id] = record
    return index


def gold_numbers(answer: str) -> list[str]:
    """抽出 gold_answer 里的数字（保留原样，供人工辨认）；去重且保持出现顺序。"""

    seen: dict[str, None] = {}
    for match in NUMBER_PATTERN.finditer(answer):
        seen.setdefault(match.group(0), None)
    return list(seen)


def snippet(text: str, limit: int) -> str:
    """压平空白后截断；截断必须**显式标注**（禁止让人以为这就是全文）。"""

    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]}……（原文共 {len(flat)} 字，此处截断，要全文请调大 --snippet-chars）"


def format_task(
    task: EvalTask, index: dict[str, dict[str, Any]], *, snippet_chars: int
) -> tuple[str, list[str]]:
    """拼一道题的复核块，返回 (文本, 这题找不到的 gold 引用列表)。"""

    lines = [
        RULE,
        f"[{task.task_id}] type={task.type} difficulty={task.difficulty}"
        f" answerable={str(task.answerable).lower()}",
        f"问题：{task.question}",
        f"gold_answer：{task.gold_answer}",
    ]
    if task.notes:
        lines.append(f"notes：{task.notes}")

    if not task.gold_chunk_ids:
        lines.append(
            "gold 引用：无（拒答题：需人工确认题面属「实体替换 / 时间错位 / 关系不存在」三种手法之一）"
        )

    missing: list[str] = []
    texts: list[str] = []
    for chunk_id in task.gold_chunk_ids:
        record = index.get(chunk_id)
        if record is None:
            missing.append(f"{task.task_id} → {chunk_id}")
            lines.append(f"gold 引用 {chunk_id}：[缺失] 不在 chunks 文件里，无法复核")
            continue
        texts.append(record["text"])
        lines.append(
            f"gold 引用 {chunk_id}：source={record['source']}｜doc_id={record['doc_id']}"
            f"｜页码={record.get('page_start')}-{record.get('page_end')}"
            f"｜章节={record.get('section_path') or '（无）'}"
        )
        lines.append(f"  原文：{snippet(record['text'], snippet_chars)}")

    # 数字对不上只在「引用齐全」时才算疑点：缺引用时对不上是必然的，报出来只会掩盖真问题。
    if task.answerable and not missing:
        haystack = "".join(texts).replace(",", "")
        absent = [number for number in gold_numbers(task.gold_answer) if number.replace(",", "") not in haystack]
        if absent:
            lines.append(
                f"[疑点] gold_answer 里的数字 {absent} 在 gold 原文里找不到"
                f"——人工确认是换算 / 汇总，还是引用错了片段"
            )
    return "\n".join(lines), missing


def build_report(
    tasks: Sequence[EvalTask], index: dict[str, dict[str, Any]], *, snippet_chars: int
) -> tuple[str, list[str]]:
    """整份复核报告（题序 = 任务集文件顺序）+ 全部缺失引用。"""

    blocks: list[str] = []
    missing: list[str] = []
    resolved = 0
    for task in tasks:
        block, task_missing = format_task(task, index, snippet_chars=snippet_chars)
        blocks.append(block)
        missing.extend(task_missing)
        resolved += len(task.gold_chunk_ids) - len(task_missing)

    counts = Counter(task.type for task in tasks)
    blocks.append(RULE)
    blocks.append(f"合计 {len(tasks)} 条题目：{dict(sorted(counts.items()))}")
    blocks.append(
        f"gold 引用 {resolved + len(missing)} 条：{resolved} 条落到原文，{len(missing)} 条找不到"
    )
    for item in missing:
        blocks.append(f"[缺失] {item}")
    blocks.append("人工复核清单（逐条打勾；任一条不过就改题或改引用，别放过）:")
    blocks.extend(
        f"  □ {item}"
        for item in (
            "gold 原文是否真支撑 gold_answer（数字口径、年份、单位）",
            "题面是否只靠 gold 引用即可回答，不需要别的片段",
            "多跳 / 多文档题是否真的必须跨片段或跨文档",
            "拒答题是否属实体替换 / 时间错位 / 关系不存在之一，且语料里确实没有答案",
            "难度标注与题目实际难度是否一致",
        )
    )
    return "\n".join(blocks), missing


def main(argv: Sequence[str] | None = None, *, out: Any = None) -> int:
    parser = argparse.ArgumentParser(description="把任务集与它的 gold 原文并排打出来供人工复核")
    parser.add_argument("--taskset", default=str(DEFAULT_TASKSET), help="任务集 JSONL，默认 eval/taskset/core.jsonl")
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS), help="chunk JSONL，默认 data/chunks.jsonl")
    parser.add_argument("--task-id", action="append", default=[], help="只复核这些 task_id（可重复）")
    parser.add_argument("--limit", type=int, default=None, help="最多复核前 N 条")
    parser.add_argument("--snippet-chars", type=int, default=DEFAULT_SNIPPET_CHARS, help="每条原文最多打多少字")
    parser.add_argument(
        "--expect",
        default="none",
        # 选项直接取 `QUOTA_TABLES` 的键：加一档任务集（如 core50）不必再改这一行，
        # 「新档位在 CLI 上不可达」这种漂移从此不可能发生。
        choices=[*sorted(QUOTA_TABLES), "none"],
        help="顺带校验条数与类型配比",
    )
    parser.add_argument("--out", default="-", help="报告输出路径；'-' 表示 stdout")
    args = parser.parse_args(argv)

    if args.snippet_chars < 1:
        print("[FAIL] --snippet-chars 必须 >= 1", file=sys.stderr)
        return 1
    if args.limit is not None and args.limit < 1:
        print("[FAIL] --limit 必须 >= 1", file=sys.stderr)
        return 1

    for label, path in (("任务集", args.taskset), ("分块文件", args.chunks)):
        if not Path(path).is_file():
            print(f"[FAIL] {label}不存在：{path}", file=sys.stderr)
            return 1

    try:
        tasks = load_taskset(args.taskset)
        if args.expect != "none":
            validate_taskset(tasks, expect=args.expect)
        index = index_chunks(args.chunks)
    except TasksetError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if args.task_id:
        wanted = set(args.task_id)
        unknown = sorted(wanted - {task.task_id for task in tasks})
        if unknown:
            print(f"[FAIL] 任务集里没有这些 task_id：{unknown}", file=sys.stderr)
            return 1
        tasks = [task for task in tasks if task.task_id in wanted]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("[FAIL] 过滤后没有可复核的题目", file=sys.stderr)
        return 1

    report, missing = build_report(tasks, index, snippet_chars=args.snippet_chars)

    if args.out == "-":
        print(report, file=out if out is not None else sys.stdout)
    else:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(report + "\n")
        print(f"[ok] 报告已写入 {target}", file=sys.stderr)

    if missing:
        for item in missing:
            print(f"[FAIL] gold 引用找不到：{item}", file=sys.stderr)
        print(
            f"[FAIL] 共 {len(missing)} 条 gold 引用在 {args.chunks} 里找不到——"
            f"题集与语料对不上，先修再复核",
            file=sys.stderr,
        )
        return 1

    print(
        f"[done] 复核 {len(tasks)} 条题目：gold 引用全部落到 {args.chunks} 的原文上",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
