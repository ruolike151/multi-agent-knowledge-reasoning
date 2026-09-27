"""状态集合与合法转移表——控制流约束的**唯一定义处**（spec §5.2）。

核心主张：**控制流不允许由 LLM 自由决定**。模型只能在 `legal_targets(current)`
给出的集合里选，越界即驳回。把这张表写成模块级常量（而不是散落在 if/else 里的隐式规则），
换来三件事：

1. **可穷举测试**：9 × 9 = 81 种「状态 × 目标」组合由参数化测试全覆盖（spec §14 的风险缓解）。
2. **可审计**：`DECIDERS` 明示哪些状态由 LLM 决定、哪些由规则唯一确定。
3. **可统计**：每次转移（含驳回）都写 trace，`illegal_transitions` 由此可算（Task 23 / Task 33）。

参考项目正是在这里退化成硬编码流水线：它的 conditional edges 看似可配置，实际分支条件是
写死的字符串比较，既不可测也不可解释（spec §2.2）。
"""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    """九个编排状态。用 ``StrEnum``：写进 trace / JSONL / RunResult 时无需任何转换。"""

    PLANNING = "PLANNING"
    DISPATCHING = "DISPATCHING"
    WORKING = "WORKING"
    VERIFYING = "VERIFYING"
    REVISE = "REVISE"
    SYNTHESIZING = "SYNTHESIZING"
    REFUSE = "REFUSE"
    DEGRADED = "DEGRADED"
    DONE = "DONE"


INITIAL_STATE = State.PLANNING
TERMINAL_STATES: frozenset[State] = frozenset({State.DONE})

# 与 spec §5.2 的转移表逐行对应。改这里就等于改规范，必须同步改 spec 与本任务的测试字面量。
LEGAL_TRANSITIONS: dict[State, tuple[State, ...]] = {
    State.PLANNING: (State.DISPATCHING, State.SYNTHESIZING, State.REFUSE),
    State.DISPATCHING: (State.WORKING,),
    State.WORKING: (State.WORKING, State.VERIFYING),
    State.VERIFYING: (State.SYNTHESIZING, State.REVISE, State.DEGRADED),
    State.REVISE: (State.WORKING,),
    State.SYNTHESIZING: (State.DONE,),
    State.REFUSE: (State.DONE,),
    State.DEGRADED: (State.DONE,),
    State.DONE: (),
}

DECIDERS: dict[State, str] = {
    State.PLANNING: "llm",
    State.DISPATCHING: "deterministic",
    State.WORKING: "deterministic",
    State.VERIFYING: "llm",
    State.REVISE: "deterministic",
    State.SYNTHESIZING: "deterministic",
    State.REFUSE: "deterministic",
    State.DEGRADED: "deterministic",
    State.DONE: "deterministic",
}


class InvalidTransition(RuntimeError):
    """运行时非法转移：当前状态不接受该目标。"""

    def __init__(self, current: State, target: State) -> None:
        self.current = current
        self.target = target
        legal = LEGAL_TRANSITIONS[current]
        rendered = ", ".join(legal) if legal else "（无：终态）"
        super().__init__(f"非法转移：{current} → {target}；{current} 的合法后继：{rendered}")


class UnknownStateName(InvalidTransition):
    """模型返回了不在状态集合内的名字（含大小写与空格错误）。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.current = None
        self.target = None
        RuntimeError.__init__(
            self,
            f"{name!r} 不是合法状态；合法状态名：{', '.join(State)}",
        )


class StateTableError(RuntimeError):
    """转移表自身构造错误（不是一次运行时非法转移）。"""


def legal_targets(current: State) -> tuple[State, ...]:
    """当前状态的合法后继集合。未知状态抛 `KeyError`：表是封闭集合。"""

    return LEGAL_TRANSITIONS[current]


def check_transition(current: State, target: State) -> None:
    """非法转移抛 `InvalidTransition`。Supervisor 与 Runner 每次转移前都必须调用它。"""

    if target not in LEGAL_TRANSITIONS[current]:
        raise InvalidTransition(current, target)


def state_from_name(name: str) -> State:
    """把模型返回的字符串转成 `State`。

    严格要求大小写与空格：模型返回 ``"planning"``、``"PLANNING "`` 一律算非法。
    宽松归一化会把「模型不听话」静默修好，而驳回次数正是我们要度量并写进 trace 的行为
    （Task 29 的「驳回并重选，最多 2 次」）。
    """

    try:
        return State(name)
    except ValueError as exc:
        raise UnknownStateName(name) from exc


def check_all_transitions() -> None:
    """表自检：目标都是已声明状态、终态无后继、决策者只有两种且 LLM 只管两个状态。"""

    for source, targets in LEGAL_TRANSITIONS.items():
        if source in TERMINAL_STATES and targets:
            raise StateTableError(f"终态 {source} 不应有后继：{targets}")
        for target in targets:
            if not isinstance(target, State) or target not in LEGAL_TRANSITIONS:
                raise StateTableError(f"{source} 的后继 {target!r} 不是已声明状态")

    if set(DECIDERS) != set(LEGAL_TRANSITIONS):
        raise StateTableError("DECIDERS 的键必须与状态集合完全一致")
    if set(DECIDERS.values()) != {"llm", "deterministic"}:
        raise StateTableError(f"决策者只能是 llm / deterministic，实际：{set(DECIDERS.values())}")
    llm_states = {state for state, decider in DECIDERS.items() if decider == "llm"}
    if llm_states != {State.PLANNING, State.VERIFYING}:
        raise StateTableError(f"只有 PLANNING / VERIFYING 允许由 LLM 决策，实际：{llm_states}")
