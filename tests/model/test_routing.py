import pytest

from akr.model.routing import NODE_MODES, mode_for


def test_reasoning_nodes_use_thinking():
    assert mode_for("plan") == "thinking"
    assert mode_for("route") == "thinking"
    assert mode_for("verify") == "thinking"


def test_formatting_nodes_use_non_thinking():
    assert mode_for("summarize") == "non-thinking"
    assert mode_for("extract") == "non-thinking"
    assert mode_for("synthesize") == "non-thinking"


def test_judges_are_split_by_cost_sensitivity():
    assert mode_for("judge_answer") == "thinking"
    assert mode_for("judge_entailment") == "non-thinking"


def test_unknown_node_raises_instead_of_defaulting():
    with pytest.raises(KeyError):
        mode_for("不存在的节点")


def test_every_mode_is_one_of_two_values():
    assert set(NODE_MODES.values()) == {"thinking", "non-thinking"}
