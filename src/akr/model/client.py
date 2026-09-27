from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from openai import OpenAI

from akr.config import Settings
from akr.model.cost import Usage

Mode = Literal["thinking", "non-thinking"]


class Transport(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


@dataclass
class LLMResult:
    text: str
    reasoning: str
    usage: Usage
    mode: Mode
    tool_calls: list[dict] = field(default_factory=list)


class LLMClient:
    """与 DeepSeek 通信的唯一出口。

    所有 prompt 都必须经此构造，以保证「稳定前缀 + 动态后缀」的
    前缀缓存友好结构（spec §7.5）。
    """

    def __init__(self, settings: Settings, transport: Transport | None = None) -> None:
        self._settings = settings
        if transport is None:
            transport = OpenAI(
                api_key=settings.api_key, base_url=settings.base_url
            ).chat.completions
        self._transport = transport

    def chat(
        self,
        *,
        system: str,
        messages: list[dict],
        mode: Mode,
        tools: list[dict] | None = None,
        json_object: bool = False,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if mode == "thinking":
            # thinking 模式下 temperature 等参数会被静默忽略，故一概不传
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = "high"
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            kwargs["temperature"] = 0
        if tools:
            kwargs["tools"] = tools
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._transport.create(**kwargs)
        message = response.choices[0].message
        # 保留 OpenAI 的**原样形状**（`id` / `type` / `function.name` / `function.arguments`），
        # 不压扁成 `{id, name, arguments}`：
        # 1. 这些字典会被调用方原样塞回下一轮 `messages`，少一个 `type` 服务端直接 422
        #    （2026-09-25 实测：`messages[2]: missing field 'type'`）；
        # 2. 工具分发侧按 `tool_call["function"]["name"]` 取名字，压扁后取到空名字，
        #    每次工具调用都会退化成"未知工具 ''"。
        tool_calls = [
            {
                "id": call.id,
                "type": getattr(call, "type", "function") or "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        return LLMResult(
            text=message.content or "",
            reasoning=getattr(message, "reasoning_content", "") or "",
            usage=Usage.from_api(response.usage),
            mode=mode,
            tool_calls=tool_calls,
        )
