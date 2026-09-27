from dataclasses import dataclass
from typing import Any

# deepseek-flash 官方价目（美元 / 1M tokens，2026-09-23 文档）
PRICES_USD_PER_MTOK: dict[bool, dict[str, float]] = {
    True: {"hit": 0.006, "miss": 0.30, "out": 1.20},   # 高峰
    False: {"hit": 0.003, "miss": 0.15, "out": 0.60},  # 低峰（半价）
}
USD_TO_CNY = 7.1


@dataclass(frozen=True)
class Usage:
    hit_tokens: int = 0
    miss_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.hit_tokens + self.miss_tokens + self.output_tokens

    @classmethod
    def from_api(cls, usage_obj: Any) -> "Usage":
        return cls(
            hit_tokens=int(getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0),
            miss_tokens=int(getattr(usage_obj, "prompt_cache_miss_tokens", 0) or 0),
            output_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
        )

    def cost_cny(self, *, peak: bool) -> float:
        price = PRICES_USD_PER_MTOK[peak]
        usd = (
            self.hit_tokens * price["hit"]
            + self.miss_tokens * price["miss"]
            + self.output_tokens * price["out"]
        ) / 1_000_000
        return usd * USD_TO_CNY

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.hit_tokens + other.hit_tokens,
            self.miss_tokens + other.miss_tokens,
            self.output_tokens + other.output_tokens,
        )
