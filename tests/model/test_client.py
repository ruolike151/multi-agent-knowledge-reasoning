from types import SimpleNamespace

from akr.config import Settings
from akr.model.client import LLMClient


class FakeTransport:
    """记录最后一次请求参数，返回固定响应；不发任何网络请求。"""

    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok", reasoning_content="思考", tool_calls=None
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_cache_hit_tokens=100,
                prompt_cache_miss_tokens=50,
                completion_tokens=20,
            ),
        )


class ToolCallTransport:
    """返回一条带工具调用的回复；形状照抄 openai SDK：`function` 是嵌套对象。"""

    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        reasoning_content="先检索",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="search_documents", arguments='{"query": "甲"}'
                                ),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_cache_hit_tokens=0,
                prompt_cache_miss_tokens=10,
                completion_tokens=5,
            ),
        )


def _client():
    transport = FakeTransport()
    settings = Settings(api_key="sk-test")
    return LLMClient(settings, transport=transport), transport


def test_tool_calls_keep_the_openai_shape_so_they_can_be_echoed_back():
    """回传服务端的 `tool_calls` 必须带 `type` 与嵌套 `function`。

    压扁成 `{id, name, arguments}` 会同时砸掉两处（2026-09-25 真跑实测，两种系统 4/4 全挂）：
    1. 原样塞回下一轮 `messages` 时被服务端拒收：`messages[2]: missing field 'type'`（422）；
    2. 工具分发侧读 `tool_call["function"]["name"]` 取到空名字，每次调用都退化成"未知工具 ''"。
    """

    transport = ToolCallTransport()
    client = LLMClient(Settings(api_key="sk-test"), transport=transport)
    result = client.chat(
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        mode="thinking",
        tools=[{"type": "function", "function": {"name": "search_documents"}}],
    )

    assert result.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_documents", "arguments": '{"query": "甲"}'},
        }
    ]
    # 把这条回复原样塞回下一轮 messages：这就是合法请求体，字段少一个服务端就 422
    echoed = {"role": "assistant", "content": "", "tool_calls": result.tool_calls}
    assert all(call["type"] == "function" for call in echoed["tool_calls"])
    assert echoed["tool_calls"][0]["function"]["name"] == "search_documents"


def test_thinking_mode_enables_thinking_and_uses_high_effort():
    client, transport = _client()
    result = client.chat(system="sys", messages=[{"role": "user", "content": "q"}], mode="thinking")
    assert transport.last_kwargs["model"] == "deepseek-flash"
    assert transport.last_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert transport.last_kwargs["reasoning_effort"] == "high"
    assert "temperature" not in transport.last_kwargs
    assert result.reasoning == "思考"
    assert result.usage.hit_tokens == 100


def test_non_thinking_mode_disables_thinking_and_pins_temperature():
    client, transport = _client()
    client.chat(system="sys", messages=[{"role": "user", "content": "q"}], mode="non-thinking")
    assert transport.last_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert transport.last_kwargs["temperature"] == 0
    assert "reasoning_effort" not in transport.last_kwargs


def test_system_prompt_is_a_stable_prefix():
    client, transport = _client()
    client.chat(system="固定前缀", messages=[{"role": "user", "content": "动态后缀"}], mode="non-thinking")
    sent = transport.last_kwargs["messages"]
    assert sent[0] == {"role": "system", "content": "固定前缀"}
    assert sent[1] == {"role": "user", "content": "动态后缀"}
