from types import SimpleNamespace

import pytest

from akr.config import Settings
from akr.model.client import LLMClient, LLMResult
from akr.model.cost import Usage
from eval.baselines.react_agent import (
    SYSTEM,
    TOOL_NAMES,
    TOOL_SPECS,
    BaselineResult,
    ReActAgent,
    collect_payload_citations,
)

QUESTION = "甲公司 2024 年的营业收入是多少？"


def llm(text="", *, reasoning="", tool_calls=(), tokens=100):
    """构造一条模型回复；usage 固定成 miss=100 / out=20，方便断言总 token。"""
    return LLMResult(
        text=text,
        reasoning=reasoning,
        usage=Usage(hit_tokens=0, miss_tokens=tokens, output_tokens=20),
        mode="thinking",
        tool_calls=list(tool_calls),
    )


def call(name, arguments, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeClient:
    """按脚本顺序返回 LLMResult；脚本用完还收到调用就是测试脚本写错了。"""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append(
            {"system": system, "messages": messages, "mode": mode, "tools": tools}
        )
        if not self._script:
            raise AssertionError(f"脚本已用完，收到第 {len(self.calls)} 次模型调用")
        return self._script.pop(0)


class FakeGateway:
    """离线工具网关；默认返回两条同 chunk 的命中（向量 + BM25 各一条），用来验去重。"""

    def __init__(self, *, hits=2):
        self._hits = hits
        self.search_calls = []

    def search_documents(self, query, top_k=8, doc_filters=None):
        self.search_calls.append({"query": query, "top_k": top_k, "doc_filters": doc_filters})
        hit = {
            "chunk_id": "doc1#chunk-03",
            "doc_id": "doc1",
            "source": "年报2024.pdf",
            "text": "2024 年公司实现营业收入 12.3 亿元",
            "score": 0.9,
        }
        return {"hits": [dict(hit, retriever="vector"), dict(hit, retriever="bm25")][: self._hits]}

    def fetch_chunk(self, chunk_id):
        return {
            "chunk_id": chunk_id,
            "doc_id": "doc1",
            "source": "年报2024.pdf",
            "text": "2024 年公司实现营业收入 12.3 亿元，同比增长 8%",
            "page": 1,
        }

    def walk_graph(self, entity, hops=2, relation_types=None):
        return {"paths": []}

    def resolve_entity(self, mention, context=""):
        return {"candidates": []}

    def list_documents(self):
        return {"documents": [{"doc_id": "doc1", "source": "年报2024.pdf", "chunks": 1}]}

    def close(self):
        return None


def test_react_agent_answers_after_one_tool_call():
    client = FakeClient(
        [
            llm(tool_calls=[call("search_documents", '{"query": "甲公司 营业收入"}')]),
            llm("甲公司 2024 年营业收入为 12.3 亿元。"),
        ]
    )
    result = ReActAgent(client, FakeGateway()).run(QUESTION)

    assert isinstance(result, BaselineResult)
    assert result.system == SYSTEM == "react"
    assert result.answer == "甲公司 2024 年营业收入为 12.3 亿元。"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert result.tokens == 240  # 两次调用，每次 miss 100 + out 20
    assert result.cost_cny > 0
    assert result.degraded is False
    # 两条命中指向同一个 chunk：引用必须去重，且带上来源
    assert [c.chunk_id for c in result.citations] == ["doc1#chunk-03"]
    assert result.citations[0].source == "年报2024.pdf"
    assert result.latency_ms >= 0


def test_tool_messages_replay_reasoning_content_and_tool_call_id():
    """DeepSeek 带 tools 的请求：assistant.reasoning_content 与 tool.tool_call_id 都不能丢。"""
    client = FakeClient(
        [
            llm(reasoning="先检索年报", tool_calls=[call("search_documents", '{"query": "x"}')]),
            llm("答案"),
        ]
    )

    ReActAgent(client, FakeGateway()).run(QUESTION)

    messages = client.calls[1]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["reasoning_content"] == "先检索年报"
    assert messages[1]["tool_calls"][0]["id"] == "c1"
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "c1"
    assert "12.3" in messages[2]["content"]


def test_max_steps_exhaustion_is_degraded_with_partial_evidence():
    client = FakeClient(
        [llm(tool_calls=[call("search_documents", '{"query": "x"}')])] * 5
    )
    agent = ReActAgent(client, FakeGateway(), max_steps=3)

    result = agent.run(QUESTION)

    assert len(client.calls) == 3
    assert result.steps == 3
    assert result.degraded is True
    assert "未能" in result.answer
    assert [c.chunk_id for c in result.citations] == ["doc1#chunk-03"]  # 已拿到的证据不能丢


def test_wall_clock_stops_the_loop_and_blames_time_not_steps():
    """两组必须共用同一条 120 秒墙钟；到点收尾时降级文案要写对维度。

    `ours` 的降级文案就踩过「把校验没过也写成预算已耗尽」这个坑（偏差日志第二十八节
    缺陷 C），基线这条同理：超时写成「步数用尽」就是在答案里喂错归因。
    """
    client = FakeClient([llm(tool_calls=[call("search_documents", '{"query": "x"}')])] * 5)
    ticks = iter([0.0, 0.0, 200.0, 200.0])

    agent = ReActAgent(
        client,
        FakeGateway(),
        clock=lambda: next(ticks),
        max_wall_seconds=120.0,
    )
    result = agent.run(QUESTION)

    assert len(client.calls) == 1  # 到点之后不再发起新的模型调用
    assert result.degraded is True
    assert "时间" in result.answer
    assert "步数" not in result.answer
    assert [c.chunk_id for c in result.citations] == ["doc1#chunk-03"]


def test_unknown_tool_becomes_tool_message_not_crash():
    """模型点错工具名：回一条错误 tool 消息让它换路，绝不 KeyError 炸掉整次评测。"""
    client = FakeClient(
        [llm(tool_calls=[call("drop_database", "{}")]), llm("换个方式：无法确定")]
    )

    result = ReActAgent(client, FakeGateway()).run(QUESTION)

    assert client.calls[1]["messages"][2]["role"] == "tool"
    assert client.calls[1]["messages"][2]["content"].startswith("错误：")
    assert result.degraded is False
    assert len(result.answer) > 0


def test_bad_json_arguments_become_tool_message_not_crash():
    client = FakeClient(
        [llm(tool_calls=[call("search_documents", "{不是 json}")]), llm("参数写错了，放弃")]
    )

    result = ReActAgent(client, FakeGateway()).run(QUESTION)

    assert client.calls[1]["messages"][2]["content"].startswith("错误：")
    assert result.tool_calls == 1


@pytest.mark.parametrize("arguments", ['["甲公司 营业收入"]', '"甲公司 营业收入"', "123", "null"])
def test_non_object_arguments_become_tool_message_not_crash(arguments):
    """参数形状写错（JSON 数组 / 字符串 / 数字 / null）不能炸掉整次运行。

    2026-09-25 主实验实测：模型偶发把 `arguments` 写成 **JSON 数组**，`args.get(...)` 抛
    `AttributeError: 'list' object has no attribute 'get'`；它当时不在 `_execute_tool` 的
    捕获名单里，于是 30 次 Baseline A 里 3 次整条运行崩掉（`state=error`、成本记 0、回答空）。
    这条守住那三类形状都只变成一条"错误："工具消息。
    """

    client = FakeClient(
        [llm(tool_calls=[call("search_documents", arguments)]), llm("换个方式再查")]
    )

    result = ReActAgent(client, FakeGateway()).run(QUESTION)

    tool_message = client.calls[1]["messages"][2]
    assert tool_message["role"] == "tool"
    assert tool_message["content"].startswith("错误：")
    assert "JSON 对象" in tool_message["content"]
    assert result.answer == "换个方式再查"
    assert result.steps == 2
    assert result.tool_calls == 1


def test_tool_specs_cover_gateway_protocol():
    """工具定义只依赖 ToolGateway 协议的 5 个方法，不 import 工具层内部符号。"""
    names = [spec["function"]["name"] for spec in TOOL_SPECS]
    assert tuple(names) == TOOL_NAMES
    assert all(spec["type"] == "function" for spec in TOOL_SPECS)
    assert all(spec["function"]["parameters"]["type"] == "object" for spec in TOOL_SPECS)


def test_collect_payload_citations_walks_nested_payload():
    payload = {
        "paths": [
            {"entity": "甲", "evidence": {"chunk_id": "doc2#chunk-01", "doc_id": "doc2",
                                          "source": "年报2023.pdf"}}
        ],
        "note": "no chunk here",
    }

    citations = collect_payload_citations(payload)

    assert [c.chunk_id for c in citations] == ["doc2#chunk-01"]
    assert citations[0].doc_id == "doc2"


class ScriptedTransport:
    """按脚本返回 **SDK 形状** 的回复，让真 `LLMClient` 参与解析与回传。"""

    def __init__(self, script):
        self._script = list(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self._script.pop(0)
        calls = [
            SimpleNamespace(
                id=item["id"],
                type="function",
                function=SimpleNamespace(name=item["name"], arguments=item["arguments"]),
            )
            for item in step.get("tool_calls", [])
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=step.get("content", ""),
                        reasoning_content=step.get("reasoning", ""),
                        tool_calls=calls or None,
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_cache_hit_tokens=0,
                prompt_cache_miss_tokens=100,
                completion_tokens=20,
            ),
        )


def test_react_agent_works_with_the_real_llm_client():
    """真 `LLMClient` 喂真 `ReActAgent`：这条是"两处 tool_calls 形状必须一致"的机器化版本。

    前面所有用例都自己伪造 `LLMResult`，于是 `LLMClient` 把 tool_calls 压扁成
    `{id, name, arguments}`（丢了 `type`、把 `function.name` 提到顶层）这件事谁都拦不住：
    2026-09-25 真跑时 Baseline A 每轮都在服务端 422，且工具分发退化成"未知工具 ''"。
    """

    transport = ScriptedTransport(
        [
            {
                "reasoning": "先检索年报",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_documents",
                        "arguments": '{"query": "甲公司 营业收入"}',
                    }
                ],
            },
            {"content": "甲公司 2024 年营业收入为 12.3 亿元。"},
        ]
    )
    gateway = FakeGateway()
    agent = ReActAgent(LLMClient(Settings(api_key="sk-test"), transport=transport), gateway)

    result = agent.run(QUESTION)

    # 工具真的被分发了（而不是退化成"未知工具 ''"），证据真的被收下了
    assert gateway.search_calls == [
        {"query": "甲公司 营业收入", "top_k": 8, "doc_filters": None}
    ]
    assert [citation.chunk_id for citation in result.citations] == ["doc1#chunk-03"]
    assert result.answer == "甲公司 2024 年营业收入为 12.3 亿元。"
    assert result.degraded is False

    # 第 2 轮请求：[system, user, assistant, tool]——回传的 assistant 消息必须是合法 OpenAI 形状
    second = transport.requests[1]["messages"]
    assert second[2]["tool_calls"][0]["type"] == "function"
    assert second[2]["tool_calls"][0]["function"]["name"] == "search_documents"
    # 回传的 tool 消息里是真实检索结果，不是错误文本
    assert "doc1#chunk-03" in second[3]["content"]
    assert "未知工具" not in second[3]["content"]
