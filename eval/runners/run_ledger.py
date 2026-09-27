"""按「运行次数」记的硬闸门：整场实验总共能跑多少次，写在盘上（2026-09-25 用户决策）。

为什么不能继续用 `BudgetGuard` 控闸：那个管的是**一次 CLI 调用内部**花了多少钱，进程一退
就归零。两次调用各跑 40 次，总次数就变成 80 次，闸门形同虚设。用户选的控闸口径是**实验
次数**（定为 60 次），所以次数必须落盘，而且必须**在发请求之前**判定——超支的那一次调用
一旦发出去就收不回来了。

三条口径写在最前面，因为报告里的"花了多少钱、跑了几次"都要按它读：

1. 一次「运行」= 一个系统在一道题上跑一次（= 一条 `RunRecord`）。judge 的调用**不算**独立
   运行——它跟着运行走，钱由 `run_experiment.py` 里的 `judge_guard` 那本账管。
2. **先记账、后调用**：`claim()` 返回时盘上的 `used` 已经 +1。中途 Ctrl-C 或崩溃只会
   多留一笔已用次数（少跑一次），不会少记（超跑一次）。宁可少跑，不可超跑。
3. 账本读不动就**报错退出**，绝不静默从 0 重新计——静默重置等于把闸门拆掉。

两个 runner（`run_experiment.py` / `fault_injection.py`）默认共用 `data/run-ledger.json`，
于是 60 次是**整场实验的总数**，而不是每个脚本各 60 次。要开一场新实验，就显式改
`--ledger` 指向另一个文件（旧账留着，不删也能对账）；要抬上限，必须显式写 `--max-runs N`。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LEDGER_VERSION = 1
DEFAULT_MAX_RUNS = 60                       # 用户 2026-09-25 决定：整场实验 60 次运行
DEFAULT_LEDGER_PATH = Path("data/run-ledger.json")


class RunLedgerError(RuntimeError):
    """账本不可用（文件读不动、字段非法、上限/计数非法）。"""


class RunCountExceeded(RunLedgerError):
    """运行次数触顶：调用方应当中断整场实验，而不是"跑完再算总账"。"""


@dataclass(frozen=True)
class LedgerState:
    """账本的一份只读快照。"""

    path: Path
    max_runs: int
    used: int
    history: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def remaining(self) -> int:
        return max(self.max_runs - self.used, 0)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ledger": str(self.path),
            "max_runs": self.max_runs,
            "used": self.used,
            "remaining": self.remaining,
        }


class RunLedger:
    """落盘的次数闸门。

    `max_runs=None` 表示"沿用账本里记的上限"（账本不存在时用 `DEFAULT_MAX_RUNS`）；显式传
    一个数表示"这一场就用这个上限"，它会覆盖账本里的值，并通过 `cap_changed` 报出来——
    抬上限必须是一个被看得见的动作，不能悄悄发生。

    属性与 `claim()` 每次都**重新读盘**：同一场实验里两个 runner 共用一本账时，手里的旧
    对象不会拿着过期数字放行（否则两个进程各读到 `used=59`，就能跑出 61 次）。
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_LEDGER_PATH,
        *,
        max_runs: int | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_runs is not None and max_runs < 1:
            raise RunLedgerError(f"运行次数上限必须 >= 1：{max_runs}")
        self._path = Path(path)
        self._clock = clock
        self._explicit_cap = max_runs
        recorded = self._read()
        if recorded is None:
            self._max_runs = DEFAULT_MAX_RUNS if max_runs is None else int(max_runs)
            self._used = 0
            self._history: list[dict[str, Any]] = []
        else:
            self._max_runs, self._used, self._history = recorded
            if max_runs is not None:
                self._max_runs = int(max_runs)

    # ---- 只读视图 ----------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def max_runs(self) -> int:
        self._refresh()
        return self._max_runs

    @property
    def used(self) -> int:
        self._refresh()
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.max_runs - self.used)

    @property
    def cap_changed(self) -> bool:
        """显式传的上限与账本里记的**不一致**时才算改过；账本不存在时不算。"""
        recorded = self._read()
        if recorded is None or self._explicit_cap is None:
            return False
        return int(self._explicit_cap) != recorded[0]

    @property
    def state(self) -> LedgerState:
        self._refresh()
        return LedgerState(
            path=self._path,
            max_runs=self._max_runs,
            used=self._used,
            history=tuple(self._history),
        )

    # ---- 记账 --------------------------------------------------------------

    def claim(
        self,
        *,
        phase: str,
        system: str,
        task_id: str,
        count: int = 1,
    ) -> int:
        """占一次运行额度，成功即落盘并返回累计次数；触顶抛 `RunCountExceeded`。

        触顶时**不改盘上任何数字**：已用次数必须始终是"真实跑过的次数"，
        否则中断后的对账会显示一个比实际更大的数字。
        """
        if count < 1:
            raise RunLedgerError(f"count 必须 >= 1：{count}")
        self._refresh()
        if self._used + count > self._max_runs:
            raise RunCountExceeded(
                f"运行次数触顶：已用 {self._used}/{self._max_runs} 次，"
                f"本次还想跑 {count} 次（账本：{self._path}；"
                f"要抬上限必须显式传 --max-runs，要开新账用 --ledger 指别的文件）"
            )
        self._used += count
        self._history.append(
            {
                "at": self._stamp(),
                "phase": phase,
                "system": system,
                "task_id": task_id,
                "used": self._used,
            }
        )
        self._write()
        return self._used

    # ---- 落盘 --------------------------------------------------------------

    def _stamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self._clock()))

    def _refresh(self) -> None:
        """以盘上的账为准；文件被删掉时才退回内存状态。"""
        recorded = self._read()
        if recorded is None:
            return
        recorded_cap, recorded_used, recorded_history = recorded
        self._used = recorded_used
        self._history = list(recorded_history)
        if self._explicit_cap is None:
            self._max_runs = recorded_cap

    def _read(self) -> tuple[int, int, tuple[dict[str, Any], ...]] | None:
        """读账本；文件不存在返回 None，存在但读不动/字段非法则抛错（不静默重置）。"""
        if not self._path.exists():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunLedgerError(
                f"运行次数账本无法解析：{self._path}（{type(exc).__name__}: {exc}）。"
                f"不会从 0 重新计数——请手工核对或改 --ledger 指向新账本"
            ) from exc
        if not isinstance(payload, dict):
            raise RunLedgerError(f"运行次数账本必须是一个 JSON 对象：{self._path}")
        max_runs = payload.get("max_runs")
        used = payload.get("used")
        if not isinstance(max_runs, int) or isinstance(max_runs, bool) or max_runs < 1:
            raise RunLedgerError(f"账本 max_runs 必须是不小于 1 的整数：{max_runs!r}（{self._path}）")
        if not isinstance(used, int) or isinstance(used, bool) or used < 0:
            raise RunLedgerError(f"账本 used 必须是不小于 0 的整数：{used!r}（{self._path}）")
        history = payload.get("history", [])
        if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
            raise RunLedgerError(f"账本 history 必须是对象数组（{self._path}）")
        return max_runs, used, tuple(history)

    def _write(self) -> None:
        """先写临时文件再 `os.replace`：账本不会因为写到一半崩掉而变成半截 JSON。"""
        payload = {
            "version": LEDGER_VERSION,
            "max_runs": self._max_runs,
            "used": self._used,
            "history": self._history,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, self._path)
