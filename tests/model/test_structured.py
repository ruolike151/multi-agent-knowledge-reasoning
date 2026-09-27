import pytest
from pydantic import BaseModel

from akr.model.structured import (
    StructuredOutputError,
    make_repair_fn,
    parse_with_repair,
)


class Answer(BaseModel):
    a: int


def test_strips_json_code_fences():
    result = parse_with_repair('```json\n{"a": 1}\n```', Answer)
    assert result.a == 1


def test_repair_is_called_with_validation_error_and_can_succeed():
    prompts: list[str] = []

    def repair(prompt: str) -> str:
        prompts.append(prompt)
        return '{"a": 2}'

    result = parse_with_repair('{"a": "不是整数"}', Answer, repair=repair)
    assert result.a == 2
    assert "<校验错误>" in prompts[0]
    assert "不是整数" in prompts[0]


def test_raises_structured_error_after_max_repair():
    with pytest.raises(StructuredOutputError):
        parse_with_repair("完全不是 JSON", Answer, repair=lambda _: "还是不是", max_repair=2)


def test_raises_immediately_when_no_repair_fn():
    with pytest.raises(StructuredOutputError):
        parse_with_repair("完全不是 JSON", Answer)


def test_make_repair_fn_requests_json_object_mode():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return type("R", (), {"text": '{"a": 3}'})()

    client = FakeClient()
    repair = make_repair_fn(client, system="修复助手")
    assert repair("坏输出") == '{"a": 3}'
    assert client.calls[0]["json_object"] is True
    assert client.calls[0]["mode"] == "non-thinking"
