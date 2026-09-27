"""嵌入层单测：默认（硅基流动 API）与兜底（本地模型）两条路径。

两条路径都**不联网**：API 路径注入假的 `EmbeddingTransport`，本地路径注入假 `Encoder`。
真实请求只出现在集成测试与 `scripts/m1_smoke.py` 里。
"""

from __future__ import annotations

import builtins
import json
import math
from types import SimpleNamespace

import pytest

from akr.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingSettings,
)
from akr.storage.embeddings import (
    MAX_INPUT_CHARS,
    QUERY_PREFIX,
    ApiEncoder,
    EmbeddingBackendMissing,
    EmbeddingResponseError,
    UsageMeter,
    build_embedder,
    build_local_embedder,
    build_local_query_embedder,
    build_query_embedder,
    dump_usage,
    load_encoder,
    prepare_texts,
)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts, *, normalize_embeddings, batch_size):
        self.calls.append(
            {
                "texts": list(texts),
                "normalize": normalize_embeddings,
                "batch_size": batch_size,
            }
        )
        return [[float(len(text)), 1.0] for text in texts]


class FakeTransport:
    """OpenAI SDK `client.embeddings` 的假实现：记录入参，返回可预测的定长向量。"""

    def __init__(self, *, dimensions: int = 3) -> None:
        self.calls: list[dict] = []
        self._dimensions = dimensions

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = list(kwargs["input"])
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=index,
                    embedding=[float(index + 1)] + [1.0] * (self._dimensions - 1),
                )
                for index in range(len(inputs))
            ]
        )


def fake_settings() -> EmbeddingSettings:
    return EmbeddingSettings(api_key="sk-fake")


# --- 共用工具 -------------------------------------------------------------


def test_prepare_texts_strips_and_drops_blank():
    assert prepare_texts(["  营收 100 亿 ", "", "   "]) == ["营收 100 亿"]


def test_prepare_texts_applies_prefix_only_when_given():
    assert prepare_texts(["营收"], prefix=QUERY_PREFIX) == [QUERY_PREFIX + "营收"]
    assert prepare_texts(["营收"]) == ["营收"]


# --- 本地兜底路径 ---------------------------------------------------------


def test_local_embedder_normalizes_and_forwards_batch_size():
    encoder = FakeEncoder()
    embed = build_local_embedder(batch_size=16, loader=lambda name: encoder)
    # 假编码器返回 float(len(text))："营收 100" 是 6 个字符（2 汉字 + 空格 + 3 数字），
    # "员工 30" 是 5 个字符。计划文档此处写 7.0，属算术笔误。
    assert embed(["营收 100", "员工 30"]) == [[6.0, 1.0], [5.0, 1.0]]
    assert encoder.calls[0]["normalize"] is True
    assert encoder.calls[0]["batch_size"] == 16


def test_local_query_embedder_adds_retrieval_prefix():
    encoder = FakeEncoder()
    build_local_query_embedder(loader=lambda name: encoder)(["子公司有哪些"])
    assert encoder.calls[0]["texts"] == [QUERY_PREFIX + "子公司有哪些"]


def test_local_embedder_returns_empty_for_empty_input():
    encoder = FakeEncoder()
    assert build_local_embedder(loader=lambda name: encoder)([]) == []
    assert encoder.calls == []  # 空输入不打后端


def test_missing_backend_raises_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(EmbeddingBackendMissing) as excinfo:
        load_encoder()
    assert 'pip install -e ".[embed]"' in str(excinfo.value)


# --- 默认路径：硅基流动 API ----------------------------------------------


def test_documented_defaults_point_at_siliconflow_bge_large():
    assert DEFAULT_EMBEDDING_BASE_URL == "https://api.siliconflow.cn/v1"
    assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-large-zh-v1.5"


def test_api_embedder_sends_model_and_input_and_returns_unit_vectors():
    transport = FakeTransport()
    embed = build_embedder(fake_settings(), transport=transport)
    vectors = embed(["营收 100", "员工 30"])

    assert transport.calls == [
        {"model": "BAAI/bge-large-zh-v1.5", "input": ["营收 100", "员工 30"]}
    ]
    assert len(vectors) == 2
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)


def test_api_query_embedder_adds_retrieval_prefix():
    transport = FakeTransport()
    build_query_embedder(fake_settings(), transport=transport)(["子公司有哪些"])
    assert transport.calls[0]["input"] == [QUERY_PREFIX + "子公司有哪些"]


def test_api_embedder_batches_requests_but_keeps_row_order():
    transport = FakeTransport()
    embed = build_embedder(fake_settings(), transport=transport, batch_size=2)
    vectors = embed(["甲", "乙", "丙", "丁", "戊"])

    assert [len(call["input"]) for call in transport.calls] == [2, 2, 1]
    assert len(vectors) == 5


def test_api_embedder_sorts_response_by_index():
    """服务端并发返回时顺序不保证；必须按 `index` 重排，否则向量会错位到别的 chunk。"""
    transport = FakeTransport()
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0, 0.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )

    transport.create = create  # type: ignore[method-assign]
    assert build_embedder(fake_settings(), transport=transport)(["第一段", "第二段"]) == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]


def test_api_embedder_raises_when_response_row_count_mismatches():
    transport = FakeTransport()
    transport.create = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
    )
    embed = build_embedder(fake_settings(), transport=transport)
    with pytest.raises(EmbeddingResponseError, match="期望 2 条"):
        embed(["第一段", "第二段"])


def test_api_embedder_rejects_zero_vector():
    transport = FakeTransport()
    transport.create = lambda **kwargs: SimpleNamespace(  # type: ignore[method-assign]
        data=[SimpleNamespace(index=0, embedding=[0.0, 0.0])]
    )
    embed = build_embedder(fake_settings(), transport=transport)
    with pytest.raises(EmbeddingResponseError, match="零向量"):
        embed(["营收"])


def test_api_embedder_clips_overlong_input_to_token_safe_budget():
    transport = FakeTransport()
    embed = build_embedder(fake_settings(), transport=transport)
    embed(["营" * (MAX_INPUT_CHARS + 200)])
    assert len(transport.calls[0]["input"][0]) == MAX_INPUT_CHARS


def test_api_embedder_returns_empty_for_empty_input():
    transport = FakeTransport()
    assert build_embedder(fake_settings(), transport=transport)([]) == []
    assert transport.calls == []


# --- 用量记账（2026-09-25 加：重建向量索引要花钱，钱只能从服务端 usage 推）-----


def _transport_with_usage(texts_per_call: int, tokens_per_call: int) -> FakeTransport:
    """假后端 + 假 usage：条数由输入长度算，token 数固定，便于断言累加逻辑。"""

    transport = FakeTransport()
    base_create = transport.create

    def create(**kwargs):
        response = base_create(**kwargs)
        response.usage = SimpleNamespace(
            total_tokens=tokens_per_call,
            prompt_tokens=tokens_per_call,
        )
        return response

    transport.create = create  # type: ignore[method-assign]
    return transport


def test_usage_meter_accumulates_reported_tokens():
    transport = _transport_with_usage(texts_per_call=2, tokens_per_call=120)
    meter = UsageMeter()
    build_embedder(fake_settings(), transport=transport, batch_size=2, usage=meter)(["甲", "乙", "丙"])

    assert meter.as_dict() == {"requests": 2, "inputs": 3, "tokens": 240, "unreported": 0}


def test_usage_meter_counts_responses_without_usage_instead_of_assuming_zero():
    """没有 usage 的响应必须单记一栏：否则「没记账」会被读成「没花钱」。"""

    meter = UsageMeter()
    embed = build_embedder(fake_settings(), transport=FakeTransport(), usage=meter)
    embed(["甲", "乙"])
    assert meter.as_dict() == {"requests": 1, "inputs": 2, "tokens": 0, "unreported": 1}


def test_usage_path_is_flushed_after_every_request(tmp_path):
    """每批覆盖写一次：进程被杀也留得住已经花掉的钱，不必等跑完。"""

    target = tmp_path / "usage.json"
    transport = _transport_with_usage(texts_per_call=1, tokens_per_call=7)
    meter = UsageMeter()
    build_embedder(
        fake_settings(), transport=transport, batch_size=1, usage=meter, usage_path=target
    )(["甲", "乙"])

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "model": "BAAI/bge-large-zh-v1.5",
        "requests": 2,
        "inputs": 2,
        "tokens": 14,
        "unreported": 0,
    }


def test_dump_usage_creates_parent_directory(tmp_path):
    meter = UsageMeter()
    meter.add(inputs=3, tokens=30)
    target = tmp_path / "深" / "usage.json"
    dump_usage(meter, target, model="m")
    assert json.loads(target.read_text(encoding="utf-8"))["tokens"] == 30


def test_dump_usage_without_a_model_keeps_the_recorded_one(tmp_path):
    """收尾那次汇总写不带模型名时，不许把「钱花在哪个模型上」擦掉。"""

    target = tmp_path / "usage.json"
    meter = UsageMeter()
    meter.add(inputs=16, tokens=160)
    dump_usage(meter, target, model="BAAI/bge-large-zh-v1.5")

    later = UsageMeter()
    later.add(inputs=1, tokens=10)
    dump_usage(later, target)  # 不带 model：常见于脚本收尾

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["model"] == "BAAI/bge-large-zh-v1.5"
    assert (payload["requests"], payload["tokens"]) == (1, 10)


def test_dump_usage_survives_a_corrupted_previous_ledger(tmp_path):
    """上一份账本坏了不该让本次落盘炸掉，只是拿不回模型名而已。"""

    target = tmp_path / "usage.json"
    target.write_text("{ 不是 JSON", encoding="utf-8")
    meter = UsageMeter()
    meter.add(inputs=2, tokens=20)
    dump_usage(meter, target)
    assert json.loads(target.read_text(encoding="utf-8"))["model"] == ""


def test_document_side_prefix_is_opt_in_and_truncation_budget_is_shared():
    """文档侧前缀是显式开关：开了才拼，且拼完仍走同一个 500 字符截断。"""

    transport = FakeTransport()
    embed = build_embedder(fake_settings(), transport=transport, prefix="甲公司（002129） ")
    embed(["营收 100", "营" * (MAX_INPUT_CHARS + 50)])

    assert transport.calls[0]["input"][0] == "甲公司（002129） 营收 100"
    # 前缀挤占同一份预算：原来 500 个「营」，现在只剩 500 - 前缀长度 个
    assert len(transport.calls[0]["input"][1]) == MAX_INPUT_CHARS
    assert transport.calls[0]["input"][1].count("营") == MAX_INPUT_CHARS - len("甲公司（002129） ")


def test_default_embedder_fails_fast_without_key(monkeypatch, tmp_path):
    """缺 Key 时在构造期抛错：不要等第一次入库才在服务端拿到 401。"""
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # 避免读到仓库根的 .env
    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        build_embedder()


def test_api_encoder_exposes_configured_model():
    assert ApiEncoder(fake_settings()).model == DEFAULT_EMBEDDING_MODEL
