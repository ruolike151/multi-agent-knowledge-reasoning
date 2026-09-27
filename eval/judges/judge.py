"""LLM-as-judge：把「答案对不对」变成可复核的三档分数，并产出这把尺子自己的可信度证据。

两个判断分两次调用、走两种模式：
- `judge_answer`（节点 judge_answer，thinking）：在"部分正确"和"完全正确"之间权衡，是主观判断；
- `entailment_support`（节点 judge_entailment，non-thinking）：逐条断言对原文做蕴含判断，机械且量大。

失败方向只有一个：抛 `JudgeError`。评测器一旦静默宽容，幻觉就会被洗成"正确"，
这比被测系统本身的 bug 危险得多。这条规则来自全局约束「禁止静默吞异常」。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from akr.model.client import LLMClient, LLMResult
from akr.model.routing import mode_for
from akr.model.structured import StructuredOutputError, parse_with_repair
from eval.metrics import cohen_kappa
from eval.taskset.schema import EvalTask

# rubric 是磁盘上的文件：改刻度等于改实验条件，必须能被 git diff 到
RUBRIC_PATH = Path(__file__).with_name("rubric.md")

# 三档分数写死；越界的分数不 clamp、不四舍五入，直接报错
SCORE_LEVELS = (0, 1, 2)
# 二值化阈值：人工标签是"通过/不通过"，judge 的 0/1/2 必须映射到同一个类别空间才谈得上 κ
JUDGE_PASS_THRESHOLD = 1
# spec §9.5：校准集 15 题。写进常量是为了让报告层能核对外推的样本量
CALIBRATION_SIZE = 15
# 单次判定的原文摘录上限，与 Verifier 的 EVIDENCE_CHARS 同量级：judge 的成本也要进预算
MAX_EVIDENCE_CHARS = 1600


class JudgeError(ValueError):
    """judge 的输入或输出不可信。判不清就报错，绝不默认通过。"""


def load_rubric() -> str:
    """读取 rubric 文件。文件缺失或为空都必须炸：空 rubric 打出来的分没有任何意义。"""
    try:
        text = RUBRIC_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise JudgeError(f"读不到 rubric 文件 {RUBRIC_PATH}：{exc}") from exc
    if not text.strip():
        raise JudgeError(f"rubric 文件是空的：{RUBRIC_PATH}")
    return text


_JUDGE_TEMPLATE = """你是严格的事实性评审员，负责按固定 rubric 给候选答案打分。

以下 rubric 是你**唯一**的评分标准，逐条适用，不得自行加严或放宽：

{rubric}

输出要求：只输出一个 JSON 对象，形如
{{"score": 0, "supported": true, "reason": "一句话理由"}}
- score 必须是整数 0、1、2 之一，不允许小数或其他取值；
- supported 表示候选答案的每一条事实断言是否都能被给定原文摘录支撑；
- 除这个 JSON 对象外不要输出任何文字。
"""


def build_judge_system(rubric: str) -> str:
    """把 rubric 拼成 system prompt。

    同一份 rubric 必须得到逐字节相同的字符串——这是 judge 调用的稳定前缀，
    决定了前缀缓存能不能命中（spec §7.5）。
    """
    return _JUDGE_TEMPLATE.format(rubric=rubric.strip())


# 模块导入时就把 rubric 读进来：rubric 有问题应该在启动时暴露，而不是在跑完钱以后
JUDGE_SYSTEM: str = build_judge_system(load_rubric())


class _VerdictPayload(BaseModel):
    """judge 的原始打分输出。`extra="forbid"`：多出字段说明模型没按格式走，宁可报错。"""

    model_config = ConfigDict(extra="forbid")

    score: int
    supported: bool
    reason: str


class _EntailmentPayload(BaseModel):
    """蕴含判定的原始输出：只要一个布尔值，不要理由——理由会把输出长度和成本抬上去。"""

    model_config = ConfigDict(extra="forbid")

    supported: bool


def _parse(raw: str, schema: type[BaseModel], *, repair: Callable[[str], str] | None) -> BaseModel:
    """把模型输出解析成结构化对象；解析不了就抛 JudgeError，不做"猜一个分数"的兜底。"""
    try:
        return parse_with_repair(raw, schema, repair=repair, max_repair=1)
    except (StructuredOutputError, ValidationError) as exc:
        raise JudgeError(f"judge 输出不是合法的 {schema.__name__}：{raw[:200]!r}") from exc


def _clip(text: str, limit: int) -> str:
    """原文摘录按字符截断：judge 的调用也要进 ¥5 预算，超长证据必须截。"""
    return text if len(text) <= limit else text[:limit]


def _format_gold(task: EvalTask) -> str:
    """把 gold 答案与 gold 引用拼成可读的两行；拒答题显式标出"无 gold 引用"。"""
    citations = "、".join(task.gold_chunk_ids) if task.gold_chunk_ids else "（无 gold 引用：本题按设计应拒答）"
    return f"答案：{task.gold_answer}\n引用：{citations}"


_VERDICT_TEMPLATE = """[问题]
{question}

[标准答案与 gold 引用]
{gold}

[候选答案]
{answer}

[原文摘录]
{evidence}
"""


@dataclass(frozen=True)
class JudgeVerdict:
    """一次打分的结论。score 是三档整数，supported 是"是否每条断言都有支撑"。"""

    task_id: str
    score: int
    supported: bool
    reason: str


def judge_answer(
    client: LLMClient,
    task: EvalTask,
    answer: str,
    *,
    evidence: str,
    repair: Callable[[str], str] | None = None,
) -> tuple[JudgeVerdict, LLMResult]:
    """按 rubric 给一条候选答案打分（thinking 模式）。

    返回 `(JudgeVerdict, LLMResult)`：第二个返回值是给成本计量用的——
    评测花的钱同样是这个项目的实验开销（spec §7.7）。
    """
    messages = [
        {
            "role": "user",
            "content": _VERDICT_TEMPLATE.format(
                question=task.question,
                gold=_format_gold(task),
                answer=answer,
                evidence=_clip(evidence, MAX_EVIDENCE_CHARS),
            ),
        }
    ]
    result = client.chat(
        system=JUDGE_SYSTEM,
        messages=messages,
        mode=mode_for("judge_answer"),
        json_object=True,
    )
    payload = _parse(result.text, _VerdictPayload, repair=repair)
    if payload.score not in SCORE_LEVELS:
        raise JudgeError(
            f"judge 返回的 score={payload.score} 不在 {SCORE_LEVELS} 内（任务 {task.task_id}）"
        )
    return (
        JudgeVerdict(
            task_id=task.task_id,
            score=payload.score,
            supported=payload.supported,
            reason=payload.reason,
        ),
        result,
    )


# 蕴含判定用独立的短 system：它不需要 rubric 的一整套打分细则，
# 而它是调用量最大的一类 judge（每条断言一次），少发的每个 token 都是省下的钱
ENTAILMENT_SYSTEM = """你是事实核查员。给你一条断言和一段原文，判断断言能否由原文直接推出。

只输出 JSON：{"supported": true} 或 {"supported": false}。
只要原文缺少断言中的任一具体事实（数字、主体、时间、口径），一律输出 false。
不要输出 JSON 以外的任何文字。"""

_ENTAILMENT_TEMPLATE = """[断言]
{claim}

[原文]
{evidence}
"""


def entailment_support(
    client: LLMClient,
    claim: str,
    evidence: str,
    *,
    repair: Callable[[str], str] | None = None,
) -> tuple[bool, LLMResult]:
    """判断单条断言能否被原文支撑（non-thinking 模式）。

    这是引用支撑率与幻觉率的原子操作：Task 30 负责把答案切成断言，
    本函数负责逐条判定，Task 33 负责把布尔列表汇总成比率。
    """
    messages = [
        {
            "role": "user",
            "content": _ENTAILMENT_TEMPLATE.format(
                claim=claim,
                evidence=_clip(evidence, MAX_EVIDENCE_CHARS),
            ),
        }
    ]
    result = client.chat(
        system=ENTAILMENT_SYSTEM,
        messages=messages,
        mode=mode_for("judge_entailment"),
        json_object=True,
    )
    payload = _parse(result.text, _EntailmentPayload, repair=repair)
    return payload.supported, result


@dataclass(frozen=True)
class Divergence:
    """一条 judge 与人工不一致的样本：留着它才知道分歧是"差一档"还是"完全相反"。"""

    task_id: str
    human_passed: bool
    judge_passed: bool
    score: int


@dataclass(frozen=True)
class CalibrationReport:
    """校准结论。Task 37 会把这份报告（含分歧清单）原样写进实验文档。"""

    total: int
    agreement_rate: float
    kappa: float
    judge_pass_rate: float
    human_pass_rate: float
    divergences: tuple[Divergence, ...]
    cost_cny: float


def calibrate(
    client: LLMClient,
    tasks: Sequence[EvalTask],
    *,
    human_labels: Mapping[str, bool],
    answers: Mapping[str, str],
    evidence_by_task: Mapping[str, str],
    repair: Callable[[str], str] | None = None,
    peak: bool = False,
) -> CalibrationReport:
    """在人工标注的校准集上测量 judge 与人的一致性（spec §9.5）。

    人工标签必须是**跑实验之前**按 gold 答案与 gold 引用写好的独立文件；
    本函数的入参里没有"系统自评"这一项，就不存在"照着系统输出打标签"的捷径。
    """
    if not tasks:
        raise JudgeError("校准集为空：没有人工标注就没有一致性可言")

    for name, mapping in (
        ("human_labels", human_labels),
        ("answers", answers),
        ("evidence_by_task", evidence_by_task),
    ):
        missing = [task.task_id for task in tasks if task.task_id not in mapping]
        if missing:
            raise JudgeError(f"{name} 缺少 {len(missing)} 题的条目：{missing[:5]}")

    verdicts: list[JudgeVerdict] = []
    cost_cny = 0.0
    for task in tasks:
        verdict, result = judge_answer(
            client,
            task,
            answers[task.task_id],
            evidence=evidence_by_task[task.task_id],
            repair=repair,
        )
        verdicts.append(verdict)
        cost_cny += result.usage.cost_cny(peak=peak)

    human_binary = [bool(human_labels[task.task_id]) for task in tasks]
    judge_binary = [verdict.score >= JUDGE_PASS_THRESHOLD for verdict in verdicts]
    agreement_rate = sum(
        1 for human, machine in zip(human_binary, judge_binary, strict=True) if human == machine
    ) / len(tasks)

    divergences = tuple(
        Divergence(
            task_id=task.task_id,
            human_passed=human,
            judge_passed=machine,
            score=verdict.score,
        )
        for task, human, machine, verdict in zip(
            tasks, human_binary, judge_binary, verdicts, strict=True
        )
        if human != machine
    )

    total = len(tasks)
    return CalibrationReport(
        total=total,
        agreement_rate=agreement_rate,
        # κ 直接复用 Task 33：同一份代码算同一件事，报告里两个数才是一套口径
        kappa=cohen_kappa(human_binary, judge_binary),
        judge_pass_rate=sum(1 for flag in judge_binary if flag) / total,
        human_pass_rate=sum(1 for flag in human_binary if flag) / total,
        divergences=divergences,
        cost_cny=cost_cny,
    )


def _read_jsonl(path: str | Path) -> list[dict]:
    """读 JSONL：坏行必须报出行号，绝不跳过（全局约束：禁止静默吞异常）。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records: list[dict] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise JudgeError(f"{path}:{number} 不是合法 JSON：{exc}") from exc
    return records


def main(argv: list[str] | None = None) -> int:
    """CLI：在 15 题校准集上跑 judge，把校准报告打到 stdout（真实 API，仅集成测试调用）。

    用法：
        python -m eval.judges.judge --taskset eval/taskset/tasks.jsonl \\
            --labels eval/judges/human_labels.jsonl \\
            --answers eval/judges/calibration_answers.jsonl
    """
    import argparse
    import json as _json
    import sys

    from akr.config import Settings
    from akr.model.client import LLMClient
    from eval.taskset.schema import load_taskset

    parser = argparse.ArgumentParser(description="judge 校准：一致率与 Cohen's κ")
    parser.add_argument("--taskset", required=True, help="任务集 JSONL（Task 22 格式）")
    parser.add_argument("--labels", required=True, help="人工标签 JSONL：{task_id, passed}")
    parser.add_argument("--answers", required=True, help="候选答案 JSONL：{task_id, answer, evidence}")
    parser.add_argument("--peak", action="store_true", help="按高峰价计成本（默认低峰）")
    args = parser.parse_args(argv)

    tasks = load_taskset(args.taskset)
    human_labels = {row["task_id"]: bool(row["passed"]) for row in _read_jsonl(args.labels)}
    answer_rows = {row["task_id"]: row for row in _read_jsonl(args.answers)}
    answers = {task_id: row["answer"] for task_id, row in answer_rows.items()}
    evidence_by_task = {task_id: row["evidence"] for task_id, row in answer_rows.items()}

    client = LLMClient(Settings.from_env())
    report = calibrate(
        client,
        tasks,
        human_labels=human_labels,
        answers=answers,
        evidence_by_task=evidence_by_task,
        peak=args.peak,
    )

    print(
        _json.dumps(
            {
                "total": report.total,
                "agreement_rate": report.agreement_rate,
                "kappa": report.kappa,
                "judge_pass_rate": report.judge_pass_rate,
                "human_pass_rate": report.human_pass_rate,
                "cost_cny": report.cost_cny,
                "divergences": [
                    {
                        "task_id": item.task_id,
                        "human_passed": item.human_passed,
                        "judge_passed": item.judge_passed,
                        "score": item.score,
                    }
                    for item in report.divergences
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report.total != CALIBRATION_SIZE:
        # 样本量不符不是错误，但必须显式提醒：报告里要写清这批数字是在几道题上得到的
        print(
            f"[警告] 校准集是 {report.total} 题，spec §9.5 约定的样本量是 {CALIBRATION_SIZE} 题",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - 手动执行入口
    raise SystemExit(main())
