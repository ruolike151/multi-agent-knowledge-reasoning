"""从语料 chunk 反向构造候选评测任务（spec §9.1）。

脚本产出的是**草稿**：``gold_chunk_ids`` 直接取自输入 chunk，人工全文复核后才可进
``eval/taskset/tasks.jsonl``。它不调用被测系统——「自己出题自己答」是评测最常见的可信度漏洞。

用法：

    python -m eval.taskset.build_candidates --chunks data/chunks.jsonl --kind multi_hop --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from akr.model.routing import mode_for
from akr.model.structured import StructuredOutputError, parse_with_repair
from eval.taskset.schema import EvalTask, TasksetError, dump_taskset

DRAFT_SYSTEM = (
    "你是企业知识库评测集的出题助手。你会收到若干带 chunk_id 的原文片段。\n"
    "只输出 JSON，不要任何解释文字，格式：\n"
    '{"tasks": [{"task_id": str, "type": "single_hop|multi_hop|multi_doc|refusal",\n'
    '  "difficulty": "easy|medium|hard", "question": str, "gold_answer": str,\n'
    '  "gold_chunk_ids": [str], "answerable": true|false, "notes": str}]}\n'
    "硬性规则：\n"
    "1) 每条题目的 gold_chunk_ids 必须精确到给定片段的 chunk_id，不允许写「见年报」。\n"
    "2) type=refusal 的题目：answerable 必须为 false、gold_chunk_ids 必须为空数组，\n"
    "   构造手法只能是以下三种之一并在 notes 里注明：实体替换 / 时间错位 / 关系不存在。\n"
    "3) 其他类型的题目 answerable 必须为 true，且至少一个 gold_chunk_ids。\n"
)

KIND_HINTS: dict[str, str] = {
    "single_hop": "抽取事实与关键数字，造单跳事实题。",
    "multi_hop": "找需要跨片段或跨文档连接才能回答的关系，造多跳推理题。",
    "multi_doc": "找需要汇总对比的指标，造多文档综合题。",
    "refusal": "用实体替换 / 时间错位 / 关系不存在三种手法造应拒答题。",
}


class DraftPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[EvalTask]


def _chunk_field(chunk: Any, name: str) -> Any:
    """取 chunk 上的一个字段。

    CLI 路径传进来的是 ``read_chunks`` 从 JSONL 读出的 **dict**（SQLite 导出的 8 字段），
    单测与库内调用传的是带属性的对象（``SimpleNamespace`` / ``Chunk``）。两种都要支持：
    2026-09-25 实测只写属性访问会让 ``main()`` 在发请求之前就 ``AttributeError``。
    """

    if isinstance(chunk, Mapping):
        return chunk[name]
    return getattr(chunk, name)


def format_source_block(chunks: Sequence[Any], *, max_chars: int = 6000) -> str:
    """把 chunk 拼成带 chunk_id 标记的原文块；超长时截断并显式标注（不静默丢内容）。"""

    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks):
        header = f"### {_chunk_field(chunk, 'chunk_id')} ({_chunk_field(chunk, 'source')})"
        if used + len(header) >= max_chars:
            blocks.append(
                f"[后续 {len(chunks) - index} 个片段因长度限制被截断，请用更小的输入块重跑]"
            )
            break
        remaining = max_chars - used - len(header)
        body = _chunk_field(chunk, "text")
        if len(body) > remaining:
            blocks.append(f"{header}\n{body[:remaining]}\n[本片段因长度限制被截断]")
            used = max_chars
            break
        blocks.append(f"{header}\n{body}")
        used += len(header) + len(body)
    return "\n\n".join(blocks)


def draft_records(
    client: Any,
    source_block: str,
    *,
    kind: str,
    limit: int,
    task_id_prefix: str = "draft",
    repair: Callable[[str], str] | None = None,
) -> list[EvalTask]:
    """调用一次模型生成候选任务草稿，逐条经 ``EvalTask`` 校验后返回。"""

    if kind not in KIND_HINTS:
        raise TasksetError(f"kind 只能是 {sorted(KIND_HINTS)} 之一，收到 {kind!r}")
    if limit < 1:
        raise TasksetError("limit 必须 >= 1")

    result = client.chat(
        system=DRAFT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"出题类型：{kind}（{KIND_HINTS[kind]}）\n最多 {limit} 条。\n\n"
                f"原文片段：\n{source_block}",
            }
        ],
        mode=mode_for("extract"),
        tools=None,
        json_object=True,
    )
    try:
        payload = parse_with_repair(result.text, DraftPayload, repair=repair)
    except StructuredOutputError as exc:
        raise TasksetError(
            f"候选任务草稿不合法（kind={kind}, task_id_prefix={task_id_prefix}）：{exc}"
        ) from exc

    selected = payload.tasks[:limit]
    return [
        task.model_copy(update={"task_id": f"{task_id_prefix}-{index:03d}"})
        for index, task in enumerate(selected, start=1)
    ]


def read_chunks(path: str | Path) -> list[dict[str, Any]]:
    """读 chunk JSONL（每行至少含 chunk_id / doc_id / source / text），坏行报行号。"""

    source = Path(path)
    records: list[dict[str, Any]] = []
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TasksetError(f"{source}:{lineno} 不是合法 JSON：{exc.msg}") from exc
        for field in ("chunk_id", "doc_id", "source", "text"):
            if field not in record:
                raise TasksetError(f"{source}:{lineno} 缺少字段 {field}")
        records.append(record)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 chunk 反向构造候选评测任务（草稿）")
    parser.add_argument("--chunks", required=True, help="chunk JSONL 路径")
    parser.add_argument("--kind", required=True, choices=sorted(KIND_HINTS))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--task-id-prefix", default="draft")
    parser.add_argument("--out", default="-", help="输出 JSONL 路径；'-' 表示 stdout")
    args = parser.parse_args(argv)

    from akr.config import Settings
    from akr.model.client import LLMClient
    from akr.model.structured import make_repair_fn

    client = LLMClient(Settings.from_env())
    tasks = draft_records(
        client,
        format_source_block(read_chunks(args.chunks)),
        kind=args.kind,
        limit=args.limit,
        task_id_prefix=args.task_id_prefix,
        repair=make_repair_fn(client, system=DRAFT_SYSTEM),
    )
    if args.out == "-":
        for task in tasks:
            print(json.dumps(task.to_json_dict(), ensure_ascii=False, sort_keys=True))
        return 0

    dump_taskset(tasks, args.out)
    print(
        f"已写入 {args.out}（{len(tasks)} 条草稿；人工复核 gold 引用后才可并入 tasks.jsonl）",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
