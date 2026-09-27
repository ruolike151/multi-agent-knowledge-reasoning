# src/akr/orchestration/verifier.py
"""Verifier：逐断言校验与纠错反馈（spec §5.5）。

隔离要求：只看「断言 ↔ 原文正文」的对应关系，看不到生成过程、看不到 Supervisor 的推理。
因此本模块的输入只有 `DraftAnswer`（文本 + chunk_id）与一个取原文的注入函数。

三条代码兜底（不交给模型自觉）：
- 原文取不到时，任何断言都不能是 supported；
- 判定值不在白名单内时，一律按 unsupported 处理；
- 序号越界的判定直接丢弃：模型没回答「哪条断言」时，不能凭空造一条断言出来。

回包形状（2026-09-26 起）：模型只回 `index`，断言正文由代码按序号还原。理由见
`VERIFY_SYSTEM` 与文件末尾的「成本口径」注释。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from akr.model.routing import mode_for
from akr.model.structured import make_repair_fn, parse_with_repair
from akr.orchestration.supervisor import DraftAnswer

VERDICTS = ("supported", "partially_supported", "unsupported")
# 单题最多 2 次 verify（首判 + 一轮纠错）。15 格 v3 重跑实测 verify 均值 2.47 次/题，
# 第二轮纠错换来的支撑率提升很小、成本却接近翻倍，因此把纠错轮次收到 1 轮。
MAX_REVISE_ROUNDS = 1
MIN_CLAIM_CHARS = 4
EVIDENCE_CHARS = 1_600
# 证据块数量上限。全库 6,348 个原文块的长度中位数只有几百字，p90 也没到 EVIDENCE_CHARS，
# 所以「单块截断」几乎从不生效；真正的量级来自「喂了几块」。这里封住长尾，
# 被挡掉的编号按证据缺失处理。
MAX_EVIDENCE_BLOCKS = 12
# reason 限长：不限长时模型会把整段推理写进 reason，而下游只用到十几个字。
REASON_CHARS = 120

# 只按中文句末标点与换行切分：英文句点会把「12.3 亿元」切断，那是最典型的误伤。
CLAIM_SEPARATORS = re.compile(r"[。！？；;\n]+")

VERIFY_SYSTEM = f"""你要逐条校验答案里的断言是否有原文支撑，只输出 JSON。

字段：
- verdicts：数组，每项含 index、verdict、reason、chunk_ids。
  - index：这条判定对应「候选答案的原子断言」列表里的序号，只填数字。
  - verdict 只能是 {list(VERDICTS)} 之一，逐字小写。
  - supported：原文直接支持这条断言。
  - partially_supported：原文只支持一部分，或数字、年份有出入。
  - unsupported：原文没有支持，或与原文矛盾。
  - chunk_ids 填支撑这条断言的原文编号，只能从输入里出现过的编号中选。
- reason：一句话（{REASON_CHARS} 字以内）说明判定依据，必须引用原文里的具体说法。

判定规则：
- 每条断言都要给出判定；漏掉的断言不会被当成通过。
- 原文里标注「证据缺失」的编号不能用来支撑任何断言。
- 不要因为断言"看起来合理"就给 supported，只依据原文。
- 语料范围以外的东西宁可判 unsupported，也不要猜。

只输出 JSON 对象，不要解释，不要 Markdown 代码块。"""


class VerdictOut(BaseModel):
    """模型侧回包的一项：只描述「第几条断言 → 判定」。

    刻意不含 `claim` 字段：回抄断言是纯输出浪费，而且回抄时改一个字就再也对不回
    原断言，代码侧只能靠模糊匹配去猜，反而把「哪条断言被推翻」变得不确定。
    """

    index: int = 0
    verdict: str = ""
    reason: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class VerifyOutput(BaseModel):
    """模型侧回包的顶层形状（只用于解析，不是对外契约）。"""

    verdicts: list[VerdictOut] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim: str = ""
    verdict: str = ""
    reason: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class VerifyResult(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    has_unsupported: bool = False
    feedback: str = ""


def split_claims(text: str) -> list[str]:
    """把答案拆成原子断言。

    纯函数且被 Task 33 的引用支撑率复用：同一个切分口径下，「几条断言里有几条被引用支撑」
    才是可比数字。太短的碎片（`MIN_CLAIM_CHARS` 以下）直接丢弃，它们多是语气词或连接词，
    计入会让支撑率虚高。
    """

    claims: list[str] = []
    for fragment in CLAIM_SEPARATORS.split(text or ""):
        claim = fragment.strip().strip("。！？；;、,， ")
        if len(claim) >= MIN_CLAIM_CHARS:
            claims.append(claim)
    return claims


def _gather_evidence(
    draft: DraftAnswer,
    fetch_evidence: Callable[[str], str],
) -> tuple[list[str], set[str]]:
    """按 chunk_id 取原文正文，返回 (拼好的证据行, 不能用来支撑断言的编号集合)。

    两类编号进 `missing`：取不到正文的，以及被 `MAX_EVIDENCE_BLOCKS` 挡在门外的。
    后者虽然库里有正文，但模型这一轮根本没看到，让它拿这份没看过的证据去支撑断言
    等于把「有没有原文」这件事外包给了幻觉。
    """

    lines: list[str] = []
    missing: set[str] = set()
    for chunk_id in draft.chunk_ids[:MAX_EVIDENCE_BLOCKS]:
        text = (fetch_evidence(chunk_id) or "").strip()
        if not text:
            missing.add(chunk_id)
            lines.append(f"[{chunk_id}] （未找到原文，证据缺失）")
            continue
        lines.append(f"[{chunk_id}] {text[:EVIDENCE_CHARS]}")
    dropped = draft.chunk_ids[MAX_EVIDENCE_BLOCKS:]
    if dropped:
        missing.update(dropped)
        lines.append(
            f"（本次只列出前 {MAX_EVIDENCE_BLOCKS} 个原文块，其余编号未提供正文，不得引用。）"
        )
    return lines, missing


def _user_prompt(claims: Iterable[str], evidence_lines: list[str]) -> str:
    """动态内容全部放 user 消息，system prompt 保持静态（前缀缓存要求）。"""

    lines = ["候选答案的原子断言："]
    lines.extend(f"{index}. {claim}" for index, claim in enumerate(claims, start=1))
    lines.append("可用的原文正文：")
    lines.extend(evidence_lines or ["（没有任何原文可用）"])
    return "\n".join(lines)


def _normalize(
    verdict: ClaimVerdict,
    missing: set[str],
    allowed: set[str],
) -> ClaimVerdict:
    """代码兜底：非法判定值按 unsupported 处理；证据缺失的断言不能是 supported。

    reason 在这里统一截断：兜底前缀也要受同一个上限约束，否则「模型写长 reason」
    这条成本漏洞会从兜底分支漏回来。

    `missing` 是「这一轮没喂正文」的编号（取不到原文，或被证据块上限挡掉）；
    `allowed` 是草稿里真实存在过的编号。引用集合外的编号说明模型在编引用，
    同样不能算 supported——原先这层只靠 prompt 里的「只能从输入里选」自觉，
    结果是一条编造编号的 supported 能直接让答案过检。
    """

    value = verdict.verdict
    reason = verdict.reason
    if value not in VERDICTS:
        reason = (
            f"模型给出的判定值 {value!r} 不在 {list(VERDICTS)} 内，"
            f"按最保守的 unsupported 处理；原理由：{reason}"
        )
        value = "unsupported"
    elif value == "supported":
        missing_ids = [chunk_id for chunk_id in verdict.chunk_ids if chunk_id in missing]
        unknown_ids = [chunk_id for chunk_id in verdict.chunk_ids if chunk_id not in allowed]
        if missing_ids or unknown_ids or not verdict.chunk_ids:
            detail = missing_ids or "没有任何引用编号"
            if unknown_ids:
                detail = f"引用了没有提供的编号 {unknown_ids}"
            reason = (
                f"证据缺失（{detail}），"
                f"不能判定为 supported；原理由：{reason}"
            )
            value = "partially_supported"
    return ClaimVerdict(
        claim=verdict.claim,
        verdict=value,
        reason=_clip_reason(reason),
        chunk_ids=list(verdict.chunk_ids),
    )


def _clip_reason(reason: str) -> str:
    """reason 限长：超长就截断并标上省略号，让「话被砍过」这件事在结果里可见。"""

    text = (reason or "").strip()
    if len(text) <= REASON_CHARS:
        return text
    return text[:REASON_CHARS] + "…"


def _restore_claims(outputs: Iterable[VerdictOut], claims: list[str]) -> list[ClaimVerdict]:
    """把模型回的序号还原成断言正文。

    越界序号丢弃（不能凭空造断言，也说明模型在乱回）；同一序号回多条时取第一条，
    让结果不随模型输出顺序摆动。返回顺序按序号升序，反馈文本因此是稳定的。
    """

    first_by_index: dict[int, VerdictOut] = {}
    for item in outputs:
        index = item.index
        if not 1 <= index <= len(claims):
            continue
        if index in first_by_index:
            continue
        first_by_index[index] = item
    return [
        ClaimVerdict(
            claim=claims[index - 1],
            verdict=first_by_index[index].verdict,
            reason=first_by_index[index].reason,
            chunk_ids=list(first_by_index[index].chunk_ids),
        )
        for index in sorted(first_by_index)
    ]


def verify_answer(
    client: Any,
    draft: DraftAnswer,
    *,
    fetch_evidence: Callable[[str], str],
) -> tuple[VerifyResult, Any]:
    """执行逐断言校验，返回 (校验结果, LLMResult)。"""

    claims = split_claims(draft.text)
    evidence_lines, missing = _gather_evidence(draft, fetch_evidence)
    raw = client.chat(
        system=VERIFY_SYSTEM,
        messages=[{"role": "user", "content": _user_prompt(claims, evidence_lines)}],
        mode=mode_for("verify"),
        json_object=True,
    )
    parsed = parse_with_repair(
        raw.text,
        VerifyOutput,
        repair=make_repair_fn(client, system=VERIFY_SYSTEM, mode=mode_for("verify")),
    )
    verdicts = [
        _normalize(verdict, missing, allowed=set(draft.chunk_ids))
        for verdict in _restore_claims(parsed.verdicts, claims)
    ]
    # 反馈文本由代码拼装：模型「总结」反馈时最容易把必须重查的点润色掉。
    feedback = "\n".join(
        f"- {verdict.claim}：{verdict.reason}（{verdict.verdict}）"
        for verdict in verdicts
        if verdict.verdict != "supported"
    )
    return (
        VerifyResult(
            verdicts=verdicts,
            has_unsupported=any(verdict.verdict == "unsupported" for verdict in verdicts),
            feedback=feedback,
        ),
        raw,
    )


def should_revise(
    result: VerifyResult,
    *,
    revise_round: int,
    max_rounds: int = MAX_REVISE_ROUNDS,
) -> bool:
    """spec §5.5 的回退条件：存在 unsupported 且纠错轮次还没用完。

    `partially_supported` 不触发回退：证据不足但没被推翻的断言，回退一次也补不出原文，
    只会白烧预算；这类情况留给降级输出显式标注。
    """

    return bool(result.has_unsupported and revise_round < max_rounds)
