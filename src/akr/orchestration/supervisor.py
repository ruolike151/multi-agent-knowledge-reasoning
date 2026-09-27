"""Supervisor：规划 / 路由 / 合成（spec §5.1、§5.2、§5.4）。

三件事拆成三个函数而不是一个「大 agent」：拆解错是规划问题，路由错是控制流问题，
合成错是引用问题，混在一个 prompt 里既无法单独定位，也无法单独度量。

硬约束落点：
- 控制流由 `states.py` 的转移表决定，本模块只负责「让模型在合法集合内选 + 驳回非法值」；
- Supervisor context 只有账本与 `WorkerReport`，文档原文永不进入（spec §5.4 第 3 条）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from akr.model.routing import mode_for
from akr.model.structured import make_repair_fn, parse_with_repair
from akr.orchestration.ledger import (
    SUBTASK_KINDS,
    Citation,
    SubTask,
    TaskLedger,
    WorkerReport,
)
from akr.orchestration.states import (
    InvalidTransition,
    State,
    UnknownStateName,
    check_transition,
    legal_targets,
    state_from_name,
)

COMPRESS_TOKEN_THRESHOLD = 6_000
CHARS_PER_TOKEN = 1.5
FINDING_CHARS = 120
MAX_ROUTE_ATTEMPTS = 2

# 「这条 finding 是在说检索范围」的措辞白名单（P2，2026-09-26）。刻意收得很紧：
# 只认片段/检索本身缺东西的说法，不认「公司未发生重大资产重组」这类**关于公司本身
# 的否定事实**——后者是正经结论，贴上「检索范围说明」的标签就等于替 worker 改答案。
RETRIEVAL_SCOPE_MARKERS = (
    "未出现",
    "未找到",
    "未检索到",
    "没有找到",
    "没有出现",
    "检索片段",
    "片段中未",
    "片段里未",
    "无法从片段",
    "无法确认",
    "无法确定",
    "未解决",
    "证据缺失",
)
RETRIEVAL_SCOPE_MARK = "〔检索范围说明〕"

# 同一份年报里最常见的「并列披露口径」组。写成枚举而不是模糊匹配：把两个本该区分的
# 口径说成同一个，比漏报严重得多，所以宁可只覆盖已经在语料里确认过的这几组。
CALIBER_ALIAS_GROUPS = (
    ("归属于上市公司股东的净利润", "归属于母公司股东的净利润", "归母净利润"),
    ("营业收入", "主营业务收入", "营业总收入"),
    ("经营活动产生的现金流量净额", "经营活动现金流量净额"),
)
# 三个 prompt 都是模块级常量：静态前缀利于前缀缓存（spec §7.5）。
PLAN_SYSTEM = f"""你要把一个用户问题拆成可独立执行的子任务，只输出 JSON。

字段：
- answerable：这个问题能否用公司年报语料回答；语料之外的常识问题或闲聊给 false。
- subtasks：数组，每项含 task_id（从 t1 开始）、goal、kind、depends_on。
  - goal 必须自包含：执行这个子任务的 agent 看不到用户原问题和别的子任务，
    所以「上面那家公司」「刚才的结论」这类指代一律不允许，必须写全实体名与年份。
  - kind 只能是 {list(SUBTASK_KINDS)} 之一：retrieve（原文检索）、
    graph（实体关系多跳）、verify（需要额外一次交叉核对）。
  - depends_on 填需要先完成的 task_id，没有依赖就填空数组。
- next_state：DISPATCHING（需要派发子任务）、SYNTHESIZING（无需检索即可作答）、
  REFUSE（无法回答）。
- reason：一句话说明拆解依据。

只输出 JSON 对象，不要解释，不要 Markdown 代码块。"""

ROUTE_SYSTEM = """你是状态机路由。你只能在给定的「合法后继」集合里选一个状态，只输出 JSON。

字段：
- next_state：必须逐字等于合法后继里的某一个状态名，大小写敏感。
- reason：一句话说明为什么选它。

不要输出推理过程，不要输出集合外的状态名，不要 Markdown 代码块。"""

SYNTHESIZE_SYSTEM = """你要只依据任务账本与子任务报告写出最终答案，只输出 JSON。

字段：
- text：给用户看的答案正文，只写报告里有证据支持的结论。
- chunk_ids：答案引用的证据编号，只能从报告里出现过的 chunk_id 中选，禁止新增编号。

规则：
- 语料是公司年报，主体与年份必须写清楚，不要用指代不清的「该公司」。
- 直接回答用户问的那个口径，并把口径名写出来（例如「归属于上市公司股东的净利润」）。
- 同一份年报会并列披露多个口径（归属于上市公司股东的净利润 / 归属于母公司股东的
  净利润、营业收入 / 主营业务收入）。口径不同**不是证据缺失**：数值不同就按口径分别
  写清楚，不要因为口径不同就写成不确定。
- 标注「检索范围说明」的内容与 open_questions 记录的是「检索到的片段里没有」，
  不是答案结论。只有当用户问的那个数值在语料里确实查不到时，才把它写成不确定项；
  否则不要出现在答案里。
- 输入里出现降级标记时，明确说明哪些部分没能验证。

只输出 JSON 对象，不要解释，不要 Markdown 代码块。"""


class PlannedSubTask(BaseModel):
    task_id: str = ""
    goal: str = ""
    kind: str = "retrieve"
    depends_on: list[str] = Field(default_factory=list)


class PlanOutcome(BaseModel):
    answerable: bool = True
    subtasks: list[PlannedSubTask] = Field(default_factory=list)
    next_state: str = "DISPATCHING"
    reason: str = ""


class Decision(BaseModel):
    next_state: str = ""
    reason: str = ""


class DraftAnswer(BaseModel):
    text: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    dropped_chunk_ids: list[str] = Field(default_factory=list)


def _call_model(client: Any, *, system: str, user: str, schema, mode: str):
    """一次模型调用 + 结构化校验与自修复，返回 (解析结果, LLMResult)。

    用量由调用方累加：Task 31 需要把每次调用的 token 与成本记进 trace 与账本。
    """

    result = client.chat(
        system=system,
        messages=[{"role": "user", "content": user}],
        mode=mode,
        json_object=True,
    )
    parsed = parse_with_repair(
        result.text,
        schema,
        repair=make_repair_fn(client, system=system, mode=mode),
    )
    return parsed, result


def plan_question(
    client: Any,
    question: str,
    *,
    feedback: str | None = None,
) -> tuple[PlanOutcome, Any]:
    """规划节点（thinking）。`feedback` 用于纠错环里的重新规划。"""

    user = question if not feedback else f"{question}\n\n上一轮的驳回原因（务必修正）：{feedback}"
    return _call_model(
        client,
        system=PLAN_SYSTEM,
        user=user,
        schema=PlanOutcome,
        mode=mode_for("plan"),
    )


def build_subtasks(outcome: PlanOutcome) -> list[SubTask]:
    """把计划翻译成账本条目，并做三项校验：kind 白名单、task_id 唯一、依赖存在。

    校验放在这里而不是 `plan_question` 里：模型输出允许有瑕疵（会被校验挡下并记录），
    但账本一旦写进非法条目，后面的派发、预算、评测全都会带着脏数据跑下去。
    """

    if not outcome.answerable:
        return []

    subtasks: list[SubTask] = []
    seen: set[str] = set()
    for planned in outcome.subtasks:
        if planned.kind not in SUBTASK_KINDS:
            raise ValueError(
                f"计划里的子任务 kind 必须是 {list(SUBTASK_KINDS)} 之一，收到：{planned.kind!r}"
            )
        if not planned.task_id:
            raise ValueError("计划里的子任务缺少 task_id")
        if planned.task_id in seen:
            raise ValueError(f"计划里的 task_id {planned.task_id!r} 必须在一次运行内唯一")
        seen.add(planned.task_id)
        subtasks.append(
            SubTask(
                task_id=planned.task_id,
                goal=planned.goal,
                kind=planned.kind,
                depends_on=list(planned.depends_on),
            )
        )

    for subtask in subtasks:
        for dependency in subtask.depends_on:
            if dependency not in seen:
                raise ValueError(
                    f"子任务 {subtask.task_id} 的 depends_on 指向不存在的子任务：{dependency!r}"
                )
    return subtasks


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1.5 字符/token，仅用于决定「要不要压缩」。

    不参与成本核算（成本一律以 `Usage` 为准），所以宁可粗一点也不引入分词依赖。
    """

    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


def compress_findings(finding: str, *, chunk_ids: list[str], limit: int = FINDING_CHARS) -> str:
    """把一条 finding 压成「结论句 + chunk_id」。

    只留第一个句号之前的结论：后半段多是修饰与展开，对下游合成的边际价值最低；
    但 `chunk_id` 必须保留——压缩不能把可溯源性一起压掉。
    """

    stripped = finding.strip()
    cut = stripped.find("。")
    if cut == -1:
        head = stripped[:limit] + ("…" if len(stripped) > limit else "")
    else:
        head = stripped[: cut + 1]
        if len(head) > limit:
            head = head[:limit] + "…"
    if not chunk_ids:
        return head
    return f"{head} [{'、'.join(chunk_ids)}]"


def _render(
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    compress: bool,
    question: str | None,
) -> str:
    lines = [
        f"用户问题：{question or ledger.question}",
        f"子任务账本（{len(ledger.subtasks)} 个）：",
    ]
    for task_id, subtask in ledger.subtasks.items():
        lines.append(f"- [{task_id}] {subtask.kind} / {subtask.status}：{subtask.goal}")
        report = reports.get(task_id)
        if report is None:
            continue
        chunk_ids = [citation.chunk_id for citation in report.citations]
        # 只有「已完成」的子任务才允许压缩：失败子任务本身信息量就少，压完就没了。
        if compress and subtask.status == "done":
            lines.extend(
                f"    · {mark_retrieval_scope(compress_findings(finding, chunk_ids=chunk_ids))}"
                for finding in report.findings
            )
        else:
            lines.extend(
                f"    · {mark_retrieval_scope(finding)}" for finding in report.findings
            )
            if chunk_ids:
                lines.append(f"    （引用：{'、'.join(chunk_ids)}）")
        # open_questions 一律是「还没解决的问题」，统一打上标记，避免合成节点把它
        # 当成答案结论照抄成「不确定事项」（见 `SYNTHESIZE_SYSTEM` 第 4 条规则）。
        lines.extend(
            f"    ? {RETRIEVAL_SCOPE_MARK} {item}" for item in report.open_questions
        )
    lines.extend(_caliber_notes(reports))
    return "\n".join(lines)


def is_retrieval_scope_note(text: str) -> bool:
    """这句话是在说「检索到的片段里没有」，还是在给一条关于公司的结论。

    runner 的校验草稿过滤器与这里的渲染标记共用同一个口径，避免两处各写一份正则、
    日后改一处忘一处：同一个句子在合成上下文里被标成「检索范围说明」，却在校验草稿里
    被当成断言去判 unsupported，那是最难查的一类不一致。
    """

    return any(marker in (text or "") for marker in RETRIEVAL_SCOPE_MARKERS)


def mark_retrieval_scope(text: str) -> str:
    """给「检索范围说明」贴上标记；普通 finding 原样返回。"""

    if is_retrieval_scope_note(text):
        return f"{RETRIEVAL_SCOPE_MARK}{text}"
    return text


def _caliber_notes(reports: Mapping[str, WorkerReport]) -> list[str]:
    """把「同一份年报的并列披露口径」显式写进上下文。

    只在同一组别名里出现 **两个及以上** 不同写法时才写注：这是「同一份年报并列披露」
    的可靠信号。只出现一个口径时什么都不说——那种情况下「别把口径当成证据缺失」
    本来也不会被误读。
    """

    text = "\n".join(
        line
        for report in reports.values()
        for line in (*report.findings, *report.open_questions)
    )
    notes: list[str] = []
    for group in CALIBER_ALIAS_GROUPS:
        present = [alias for alias in group if alias in text]
        if len(present) < 2:
            continue
        joined = "」与「".join(present)
        notes.append(
            f"注：本次证据里同时出现「{joined}」——这是同一份年报并列披露的口径，"
            f"口径不同不等于证据缺失；引用数值时要写明用的是哪个口径。"
        )
    return notes


def build_context(
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    token_budget: int = COMPRESS_TOKEN_THRESHOLD,
    question: str | None = None,
) -> tuple[str, bool]:
    """渲染 Supervisor context，返回 (文本, 是否发生压缩)。

    压缩只发生在渲染期，不回写 artifact store：评测读到的 `WorkerReport` 必须与
    worker 当时写的完全一致，否则被压缩过的运行与原始运行指标不可比。
    """

    full = _render(ledger, reports, compress=False, question=question)
    if token_budget <= 0 or estimate_tokens(full) <= token_budget:
        return full, False
    return _render(ledger, reports, compress=True, question=question), True


def _route_prompt(
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    current: State,
    allowed: tuple[State, ...],
    feedback: str | None,
) -> str:
    lines = [
        f"当前状态：{current.value}",
        f"合法后继：{', '.join(state.value for state in allowed)}",
        "子任务账本与报告：",
        build_context(ledger, reports)[0],
    ]
    if feedback:
        lines.append(f"Verifier 反馈：{feedback}")
    return "\n".join(lines)


def decide_next(
    client: Any,
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    current: State,
    verifier_feedback: str | None = None,
) -> tuple[Decision, list[Any]]:
    """路由节点（thinking）：模型在合法后继集合里选，非法值驳回重选，超限降级。

    返回 (决策, 本轮所有 LLMResult)：驳回也必须留下用量，否则「非法转移」这条
    失败路径在成本上看起来是免费的，评测会低估多 Agent 的协调开销。
    """

    allowed = legal_targets(current)
    names = ", ".join(state.value for state in allowed)
    base = _route_prompt(
        ledger, reports, current=current, allowed=allowed, feedback=verifier_feedback
    )
    prompt = base
    results: list[Any] = []
    reason = ""

    for _ in range(MAX_ROUTE_ATTEMPTS):
        decision, result = _call_model(
            client,
            system=ROUTE_SYSTEM,
            user=prompt,
            schema=Decision,
            mode=mode_for("route"),
        )
        results.append(result)
        try:
            target = state_from_name(decision.next_state)
        except UnknownStateName as error:
            reason = f"非法路由：状态名 {decision.next_state!r} 无法识别（{error}）"
        else:
            try:
                check_transition(current, target)
            except InvalidTransition as error:
                # 与 legal_targets 同源，正常情况下不可达；留着是为了万一表被改坏时
                # 仍然走「驳回」而不是让非法转移直接生效。
                reason = f"非法路由：{current.value} → {target.value}（{error}）"
            else:
                return decision, results
        prompt = (
            f"{base}\n\n上一次答复被驳回：{reason}\n"
            f"只允许这些合法后继：{names}，请重选。"
        )

    return (
        Decision(
            next_state=State.DEGRADED.value,
            reason=f"连续 {MAX_ROUTE_ATTEMPTS} 次非法路由，已降级：{reason}",
        ),
        results,
    )


def collect_citations(
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
) -> list[Citation]:
    """账本内所有报告的引用并集，按 chunk_id 去重（保留首次出现顺序）。"""

    citations: list[Citation] = []
    seen: set[str] = set()
    for task_id in ledger.subtasks:
        report = reports.get(task_id)
        if report is None:
            continue
        for citation in report.citations:
            if citation.chunk_id in seen:
                continue
            seen.add(citation.chunk_id)
            citations.append(citation)
    return citations


def synthesize(
    client: Any,
    question: str,
    ledger: TaskLedger,
    reports: Mapping[str, WorkerReport],
    *,
    degraded: bool = False,
) -> tuple[DraftAnswer, Any]:
    """合成节点（non-thinking）：只依据账本与报告作答，并过滤编造的引用。

    `degraded=True` 时在 prompt 里显式写明降级：让模型把不确定项标出来，
    而不是把「没验证过的结论」写成和验证过的一样确定。
    """

    known = {citation.chunk_id for citation in collect_citations(ledger, reports)}
    context, compressed = build_context(ledger, reports, question=question)
    parts = [context]
    if compressed:
        parts.append("（本轮账本已压缩：已完成子任务的结论只保留结论句与 chunk_id。）")
    if degraded:
        parts.append(
            "降级模式：预算或纠错轮次已用尽。必须在答案里显式标注不确定项，"
            "不得补写没有证据的结论。"
        )
    draft, result = _call_model(
        client,
        system=SYNTHESIZE_SYSTEM,
        user="\n\n".join(parts),
        schema=DraftAnswer,
        mode=mode_for("synthesize"),
    )

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for chunk_id in draft.chunk_ids:
        if chunk_id in seen:
            continue
        if chunk_id in known:
            seen.add(chunk_id)
            kept.append(chunk_id)
        else:
            dropped.append(chunk_id)
    return (
        DraftAnswer(text=draft.text, chunk_ids=kept, dropped_chunk_ids=dropped),
        result,
    )
