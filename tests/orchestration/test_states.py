import pytest

from akr.orchestration.states import (
    DECIDERS,
    INITIAL_STATE,
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    InvalidTransition,
    State,
    check_all_transitions,
    check_transition,
    legal_targets,
    state_from_name,
)

# spec §5.2 的转移表逐行抄写。测试里必须是字面量：
# 如果直接引用 LEGAL_TRANSITIONS，代码写错时测试会跟着一起错。
SPEC_TABLE = {
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

ALL_STATES = tuple(State)
ALL_PAIRS = [(current, target) for current in ALL_STATES for target in ALL_STATES]


def test_table_covers_exactly_the_nine_documented_states():
    assert len(ALL_STATES) == 9
    assert set(LEGAL_TRANSITIONS) == set(ALL_STATES)
    assert LEGAL_TRANSITIONS == SPEC_TABLE


@pytest.mark.parametrize("current,target", ALL_PAIRS)
def test_every_state_pair_matches_spec_table(current, target):
    if target in SPEC_TABLE[current]:
        check_transition(current, target)
    else:
        with pytest.raises(InvalidTransition):
            check_transition(current, target)


def test_legal_targets_returns_table_row():
    assert legal_targets(State.PLANNING) == (State.DISPATCHING, State.SYNTHESIZING, State.REFUSE)
    assert legal_targets(State.DONE) == ()
    assert isinstance(legal_targets(State.WORKING), tuple)


def test_all_targets_are_declared_states():
    for targets in LEGAL_TRANSITIONS.values():
        assert all(isinstance(target, State) for target in targets)


def test_done_is_terminal():
    assert TERMINAL_STATES == frozenset({State.DONE})
    assert legal_targets(State.DONE) == ()
    assert INITIAL_STATE is State.PLANNING


def test_only_three_states_transition_into_terminal():
    sources = {state for state, targets in SPEC_TABLE.items() if State.DONE in targets}
    assert sources == {State.SYNTHESIZING, State.REFUSE, State.DEGRADED}


def test_only_planning_and_verifying_are_llm_decided():
    llm_states = {state for state, decider in DECIDERS.items() if decider == "llm"}
    assert llm_states == {State.PLANNING, State.VERIFYING}
    assert set(DECIDERS) == set(ALL_STATES)
    assert set(DECIDERS.values()) == {"llm", "deterministic"}


def test_self_loop_only_allowed_for_working():
    self_looping = [state for state, targets in SPEC_TABLE.items() if state in targets]
    assert self_looping == [State.WORKING]


def test_invalid_transition_message_lists_states_and_legal_targets():
    with pytest.raises(InvalidTransition) as excinfo:
        check_transition(State.DONE, State.PLANNING)
    message = str(excinfo.value)
    assert "DONE" in message
    assert "PLANNING" in message
    assert "REFUSE" not in message  # DONE 的合法后继为空


def test_state_from_name_accepts_canonical_names():
    assert state_from_name("PLANNING") is State.PLANNING
    assert state_from_name("DEGRADED") is State.DEGRADED


def test_state_from_name_rejects_unknown_and_wrong_case():
    with pytest.raises(InvalidTransition, match="不是合法状态"):
        state_from_name("PLANNING ")  # 尾随空格也必须算非法
    with pytest.raises(InvalidTransition, match="不是合法状态"):
        state_from_name("planning")
    with pytest.raises(InvalidTransition, match="不是合法状态"):
        state_from_name("REPORT")


def test_check_all_transitions_passes_for_committed_table():
    check_all_transitions()  # 表自检：目标必须都是合法状态、无例外
