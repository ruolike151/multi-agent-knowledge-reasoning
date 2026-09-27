"""worker 层的共享错误类型。

放在单独模块而不是某个 worker 里：Task 27 与 Task 28 都要用同一个异常类型，
否则 Task 31 得 catch 两次，而且两个 worker 的错误语义会慢慢分叉。
"""

from __future__ import annotations

from typing import Any


class ToolResultError(RuntimeError):
    """工具调用失败，或工具返回的载荷不合约定的字段要求。

    worker 自己不决定重试还是降级：它带工具名与详情抛出去，由编排层（Task 31）
    按预算与子任务重试次数决定「重试 / 降级 / 上报」（spec §10 的错误处理行）。
    """

    def __init__(self, tool: str, detail: Any) -> None:
        self.tool = tool
        self.detail = detail
        super().__init__(f"工具 {tool} 调用失败：{detail}")
