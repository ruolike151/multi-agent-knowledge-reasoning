import json
import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from akr.model.client import LLMClient, Mode

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


class StructuredOutputError(RuntimeError):
    """模型输出无法解析为合 schema 的对象，且自修复次数已耗尽。

    严禁在解析失败时返回空对象——参考项目的静默降级正是缺陷 5 的一部分。
    """


def _strip_fences(raw: str) -> str:
    return _FENCE.sub("", raw).strip()


def parse_with_repair(
    raw: str,
    schema: type[BaseModel],
    *,
    repair: Callable[[str], str] | None = None,
    max_repair: int = 2,
) -> BaseModel:
    text = raw
    last_error: Exception | None = None
    for _ in range(max_repair + 1):
        try:
            return schema.model_validate(json.loads(_strip_fences(text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if repair is None:
                break
            text = repair(f"{text}\n\n<校验错误>{exc}</校验错误>")
    raise StructuredOutputError(f"结构化输出解析失败：{last_error}") from last_error


def make_repair_fn(
    client: LLMClient, *, system: str, mode: Mode = "non-thinking"
) -> Callable[[str], str]:
    """把「原始输出 + 校验错误」回灌模型要求修复（spec §7.4）。"""

    def repair(prompt: str) -> str:
        return client.chat(
            system=system,
            messages=[{"role": "user", "content": prompt}],
            mode=mode,
            json_object=True,
        ).text

    return repair
