"""任务集 schema：全项目唯一一条验收链路的数据契约（spec §9.1）。

设计取舍：

- 类型配比写成常量表（``CORE_QUOTAS`` / ``MID_QUOTAS`` / ``EXTENDED_QUOTAS``），把「多少条
  怎么分」这件事从文档里的散文变成可执行断言——配比写错会在加载阶段就报错。
  ``core`` = 30 条（历史基线，全部历史表都以它为输入，**不再改动**）；
  ``core50`` = 50 条（在 core 之上按同比例扩到 50：8/10/7/5 → 13/17/12/8）。
- ``extra="forbid"`` + ``frozen=True``：任务集是人工与模型共同编辑的数据文件，
  多写的字段必须报错而不是被静默忽略。
- 坏行一律抛 ``TasksetError`` 并带 ``path:行号``，绝不跳过（Global Constraints：禁止静默吞异常）。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

TaskType = Literal["single_hop", "multi_hop", "multi_doc", "refusal"]
Difficulty = Literal["easy", "medium", "hard"]

TASK_TYPES: tuple[str, ...] = ("single_hop", "multi_hop", "multi_doc", "refusal")

CORE_QUOTAS: dict[str, int] = {"single_hop": 8, "multi_hop": 10, "multi_doc": 7, "refusal": 5}
MID_QUOTAS: dict[str, int] = {"single_hop": 13, "multi_hop": 17, "multi_doc": 12, "refusal": 8}
EXTENDED_QUOTAS: dict[str, int] = {
    "single_hop": 20,
    "multi_hop": 28,
    "multi_doc": 18,
    "refusal": 14,
}

QUOTA_TABLES: dict[str, dict[str, int]] = {
    "core": CORE_QUOTAS,
    "core50": MID_QUOTAS,
    "extended": EXTENDED_QUOTAS,
}


class TasksetError(ValueError):
    """任务集数据错误。"""


class EvalTask(BaseModel):
    """一行任务记录，字段与 spec §9.1 的任务记录格式一一对应。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    type: TaskType
    difficulty: Difficulty
    question: str = Field(min_length=1)
    gold_answer: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    answerable: bool
    notes: str = ""

    @field_validator("gold_chunk_ids")
    @classmethod
    def _reject_blank_chunk_ids(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            if not chunk_id.strip():
                raise ValueError("gold_chunk_ids 不能包含空字符串")
        return value

    @model_validator(mode="after")
    def _answerable_consistent_with_type(self) -> "EvalTask":
        is_refusal = self.type == "refusal"
        if is_refusal and self.answerable:
            raise ValueError("refusal 类任务的 answerable 必须为 false")
        if not is_refusal and not self.answerable:
            raise ValueError("只有 refusal 类任务允许 answerable=false")
        if self.answerable and not self.gold_chunk_ids:
            raise ValueError("可答任务必须至少给出一个 gold_chunk_ids")
        if is_refusal and self.gold_chunk_ids:
            raise ValueError("refusal 类任务的 gold_chunk_ids 必须为空")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_taskset(path: str | Path) -> list[EvalTask]:
    """逐行读 JSONL：空行忽略，坏行抛 ``TasksetError``（带 ``path:行号``）。"""

    source = Path(path)
    try:
        raw_text = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TasksetError(f"任务集文件不存在：{source}") from exc

    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(raw_text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TasksetError(f"{source}:{lineno} 不是合法 JSON：{exc.msg}") from exc
        if not isinstance(payload, dict):
            raise TasksetError(f"{source}:{lineno} 每行必须是一个 JSON 对象")
        try:
            task = EvalTask.model_validate(payload)
        except ValidationError as exc:
            raise TasksetError(f"{source}:{lineno} 任务记录不合法：{exc}") from exc
        if task.task_id in seen:
            raise TasksetError(f"{source}:{lineno} 重复 task_id：{task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)
    return tasks


def dump_taskset(tasks: Iterable[EvalTask], path: str | Path) -> None:
    """写 JSONL：一行一条、``ensure_ascii=False``、键排序、末尾换行（便于 diff）。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(task.to_json_dict(), ensure_ascii=False, sort_keys=True) for task in tasks
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_taskset(tasks: Sequence[EvalTask], *, expect: str = "core") -> None:
    """校验规模与类型配比。``expect='none'`` 只查 task_id 唯一性（smoke 集用）。"""

    if expect != "none" and expect not in QUOTA_TABLES:
        raise TasksetError(
            f"expect 只能是 {sorted(QUOTA_TABLES) + ['none']} 之一，收到 {expect!r}"
        )

    counts_by_id = Counter(task.task_id for task in tasks)
    duplicated = sorted(task_id for task_id, count in counts_by_id.items() if count > 1)
    if duplicated:
        raise TasksetError(f"任务集存在重复 task_id：{duplicated}")

    if expect == "none":
        return

    quotas = QUOTA_TABLES[expect]
    expected_total = sum(quotas.values())
    if len(tasks) != expected_total:
        raise TasksetError(f"{expect} 任务集需要 {expected_total} 条，实际 {len(tasks)} 条")

    counts_by_type = Counter(task.type for task in tasks)
    for task_type, expected in quotas.items():
        actual = counts_by_type.get(task_type, 0)
        if actual != expected:
            raise TasksetError(
                f"{expect} 任务集中 {task_type} 需要 {expected} 条，实际 {actual} 条"
                f"（实际配比：{dict(sorted(counts_by_type.items()))}）"
            )
