from typing import Literal

from akr.model.client import Mode

NodeName = Literal[
    "plan",
    "route",
    "verify",
    "summarize",
    "extract",
    "synthesize",
    "judge_answer",
    "judge_entailment",
]

# 路由表即 spec §7.2 的实现，单点定义，测试与实验都读这一份
NODE_MODES: dict[str, Mode] = {
    "plan": "thinking",
    "route": "thinking",
    "verify": "thinking",
    "summarize": "non-thinking",
    "extract": "non-thinking",
    "synthesize": "non-thinking",
    "judge_answer": "thinking",
    "judge_entailment": "non-thinking",
}


def mode_for(node: str) -> Mode:
    return NODE_MODES[node]
