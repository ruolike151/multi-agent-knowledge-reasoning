"""预算与熔断（spec §5.6）——编排层的硬闸门，不是统计报表。

参考项目用「最多重试 3 次」这类拍脑袋常数控制循环；本模块把「还能不能继续跑」
变成五个可计量维度的检查：步数、模型调用次数、token、人民币成本、墙钟时间。
任一维度越限，编排层必须收敛到 SYNTHESIZING 或 DEGRADED（spec §5.2 的转移表
已为此留出 REFUSE / DEGRADED 分支）。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

# 维度名 → Budget 的字段名。维度名是外部契约（写进 trace 的 detail 与 RunResult），
# 字段名是内部实现细节，两者刻意分开，避免以后改字段名改坏对外口径。
DIMENSION_FIELDS: dict[str, str] = {
    "steps": "max_steps",
    "llm_calls": "max_llm_calls",
    "tokens": "max_tokens",
    "cost_cny": "max_cost_cny",
    "wall_seconds": "max_wall_seconds",
}

# 多维度同时越限时的报告优先级。顺序是规范的一部分：先报告最硬的约束
# （步数、调用次数是确定性上限），再报告成本类，最后是时间。
EXHAUSTION_ORDER: tuple[str, ...] = ("steps", "llm_calls", "tokens", "cost_cny", "wall_seconds")


@dataclass(frozen=True)
class Budget:
    """一次运行的预算上限。

    **这是一次有记录的偏离**：spec §5.6 的原值是 `max_tokens=120_000`、
    `max_cost_cny=0.5`，2026-09-26 按 15 格端到端实测把这两个维度抬到了
    `200_000` / `1.0`（偏差日志第四十二节）。依据是「触顶即质量事故」：

    * 15 格里最贵的一题实测 123,852 tokens / ¥0.6166，**两个维度都超了原上限**，
      被闸门打断后判分 0；
    * 另一题 `core-mh-005` 在 ¥0.5435 处被 `cost_cny` 掐掉，同样判分 0。

    也就是说原上限不是「省钱的护栏」，而是「慢题的止损线」——它把「答得慢」直接
    换算成「答错」。抬盒子之前必须先看过用量分布，否则就是拿质量换账面上的便宜；
    `max_wall_seconds` 本轮**刻意不动**（用户口令：先不管墙钟闸门），它属于同一类
    决策，仍待拍板。
    """

    max_steps: int = 24
    max_llm_calls: int = 40
    max_tokens: int = 200_000
    max_cost_cny: float = 1.0
    max_wall_seconds: int = 120


@dataclass
class BudgetState:
    """预算的已用量，可序列化进检查点（Task 26 / Task 31）。

    `clock` 可注入：墙钟维度必须在单测里被确定性地驱动，不能靠真实 sleep。
    默认用 `time.monotonic`——它不受系统时间调整影响，保证 `elapsed_seconds()`
    单调不减，这是「触顶不可逆」的前提之一。

    口径提醒：`tokens` 维度累加时必须用 Task 23 `RunSummary.tokens` 的同一口径
    （tokens_in + tokens_cached + tokens_out），否则预算与 trace 的数字对不上。
    """

    budget: Budget = field(default_factory=Budget)
    used: dict[str, float] = field(default_factory=lambda: {d: 0.0 for d in EXHAUSTION_ORDER})
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    started_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.budget is None:
            # 本任务文档把构造签名写成 `BudgetState(*, budget=None, clock=...)`，
            # 即「显式传 None」= 用默认预算（Task 31 的 `run_once(budget=None)` 就是这么传的），
            # 而不是「没有上限」。不兜这一下，`limit_of` 会在 None 上取属性直接炸。
            self.budget = Budget()
        for dimension in EXHAUSTION_ORDER:
            self.used.setdefault(dimension, 0.0)
        self.started_at = float(self.clock())

    def charge(
        self,
        *,
        steps: int = 0,
        llm_calls: int = 0,
        tokens: int = 0,
        cost_cny: float = 0.0,
    ) -> None:
        """累加已用量。负增量抛 `ValueError`：预算只能被消耗，不能被还回去。"""

        if steps < 0 or llm_calls < 0 or tokens < 0 or cost_cny < 0:
            raise ValueError("charge 不接受负增量：预算只能被消耗")
        self.used["steps"] += steps
        self.used["llm_calls"] += llm_calls
        self.used["tokens"] += tokens
        self.used["cost_cny"] += cost_cny

    def elapsed_seconds(self) -> float:
        """本次运行已消耗的墙钟秒数（含从检查点恢复之前的那一段）。"""

        return float(self.clock()) - self.started_at

    def used_of(self, dimension: str) -> float:
        """某个维度的已用量。`wall_seconds` 是实时读数，其余是累加值。"""

        if dimension not in DIMENSION_FIELDS:
            raise KeyError(f"未知预算维度：{dimension}；合法维度：{', '.join(EXHAUSTION_ORDER)}")
        if dimension == "wall_seconds":
            return self.elapsed_seconds()
        return float(self.used[dimension])

    def limit_of(self, dimension: str) -> float:
        """某个维度的上限。"""

        if dimension not in DIMENSION_FIELDS:
            raise KeyError(f"未知预算维度：{dimension}；合法维度：{', '.join(EXHAUSTION_ORDER)}")
        return float(getattr(self.budget, DIMENSION_FIELDS[dimension]))

    def exhausted(self) -> str | None:
        """越限则返回维度名，否则返回 `None`。

        判定用严格大于：`max_steps = 24` 表示允许走满 24 步，第 25 步才熔断。

        返回状态单调：`used` 只增不减（`charge` 拒绝负值、墙钟用单调时钟），
        上限是 frozen 常量，因此一旦返回非 `None` 就不可能再回到 `None`。
        编排层可以放心地在循环开头查一次。
        """

        for dimension in EXHAUSTION_ORDER:
            if self.used_of(dimension) > self.limit_of(dimension):
                return dimension
        return None

    def to_dict(self) -> dict[str, Any]:
        """检查点序列化。存相对量 `elapsed_seconds` 而不是绝对 `started_at`。"""

        return {
            "used": {dimension: float(value) for dimension, value in self.used.items()},
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "limits": asdict(self.budget),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> BudgetState:
        """从检查点恢复。把已用墙钟折进新的 `started_at`，而不是从 0 重新计时。

        直接重置为 0 等于白送一次完整墙钟预算，恢复后的运行会比正常运行跑得久，
        从而掩盖真正的超时问题。
        """

        state = cls(budget=Budget(**data["limits"]), clock=clock)
        for dimension, value in data["used"].items():
            if dimension in DIMENSION_FIELDS:
                state.used[dimension] = float(value)
        # 构造时的 `clock()` 读数就是「恢复时刻」：以它为基准回推历史墙钟，
        # 不再多读一次时钟（多读一次会把「恢复时刻」推到将来，历史时间被吃掉）。
        state.started_at -= float(data["elapsed_seconds"])
        return state
