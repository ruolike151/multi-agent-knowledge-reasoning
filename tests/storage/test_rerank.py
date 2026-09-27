"""重排层单测：硅基流动 `/v1/rerank`（cross-encoder）这一路。

全程**不联网**：注入假的 `RerankTransport` 替换真实 httpx 客户端，与
`tests/storage/test_embeddings.py` 同一套路。真实请求只在 `tmp/probe_rerank.py`
与需要花钱的对账脚本里出现。

这一层的失败模式值得单测，因为它是**唯一一条每次检索都多一次网络往返**的路径：
候选与分值错配（顺序还原错了）比不重排更糟，静默退化又会让「重排有没有用」永远量不出来。
"""

from __future__ import annotations

import json

import pytest

from akr.config import RerankSettings
from akr.storage.rerank import (
    MAX_DOC_CHARS,
    ApiReranker,
    HttpRerankTransport,
    RerankBackendError,
    RerankResponseError,
    build_reranker,
    reported_rerank_tokens,
)
from akr.storage.embeddings import UsageMeter


def settings(**overrides) -> RerankSettings:
    return RerankSettings(api_key="sk-fake", **overrides)


class FakeTransport:
    """记录入参、按脚本返回响应体（不是 httpx 客户端）。"""

    def __init__(self, payload, *, raises: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._payload = payload
        self._raises = raises

    def create(self, *, model, query, documents):
        self.calls.append({"model": model, "query": query, "documents": list(documents)})
        if self._raises is not None:
            raise self._raises
        return self._payload


def payload(scores: dict[int, float], *, input_tokens: int | None = 42) -> dict:
    """服务端形状：`results` 里是 `index` + `relevance_score`，**不保证按输入顺序**。"""

    body: dict = {
        "id": "req-1",
        "results": [
            {"index": index, "relevance_score": score} for index, score in scores.items()
        ],
    }
    if input_tokens is not None:
        body["meta"] = {"tokens": {"input_tokens": input_tokens}}
    return body


# --- 顺序还原：这是「比不重排更糟」的那一类错 ---------------------------------


def test_scores_are_returned_in_input_order_even_when_the_server_reverses_them():
    """服务端可以不按输入顺序回；调用方拿到的 `scores[i]` 必须真的是第 i 条的分。

    这条是本模块最要紧的断言：错配会让「重排后的名次」与「候选原文」对不上，
    而且它**看起来仍然是一列有理有据的分数**。
    """

    transport = FakeTransport(payload({2: 0.9, 0: 0.7, 1: 0.1}))
    rerank = ApiReranker(settings(), transport=transport).scores

    assert rerank("问题", ["甲", "乙", "丙"]) == [0.7, 0.1, 0.9]
    assert transport.calls[0]["documents"] == ["甲", "乙", "丙"]


def test_the_model_name_and_query_are_forwarded_verbatim():
    transport = FakeTransport(payload({0: 0.5}))
    rerank = ApiReranker(settings(model="BAAI/bge-reranker-v2-m3"), transport=transport).scores

    rerank("净利润是多少", ["文本"])

    assert transport.calls[0]["model"] == "BAAI/bge-reranker-v2-m3"
    assert transport.calls[0]["query"] == "净利润是多少"


def test_each_candidate_is_truncated_to_the_shared_budget():
    """单条候选截到 `MAX_DOC_CHARS`：与嵌入侧 `MAX_INPUT_CHARS` 同一个数，两处口径一致。"""

    transport = FakeTransport(payload({0: 0.5, 1: 0.4}))
    rerank = ApiReranker(settings(), transport=transport).scores

    rerank("问题", ["字" * (MAX_DOC_CHARS + 50), "短"])

    assert transport.calls[0]["documents"] == ["字" * MAX_DOC_CHARS, "短"]


def test_empty_candidates_never_touch_the_backend():
    """空候选直接返回 `[]`，一次网络都不打——池子为空不该产生一次付费请求。"""

    transport = FakeTransport(payload({}))
    rerank = ApiReranker(settings(), transport=transport).scores

    assert rerank("问题", []) == []
    assert transport.calls == []


# --- 响应校验：宁可报错，也不静默退化成「不重排」 -----------------------------


def test_a_short_result_list_is_an_error_not_a_silent_partial_rerank():
    """服务端按 `top_n` 截断过时必须报错：半个池子的重排会让尾部候选的名次凭空变好。"""

    transport = FakeTransport(payload({0: 0.9}))
    rerank = ApiReranker(settings(), transport=transport).scores

    with pytest.raises(RerankResponseError, match="1 条重排分，期望 3 条"):
        rerank("问题", ["甲", "乙", "丙"])


@pytest.mark.parametrize(
    "body, match",
    [
        (payload({0: 0.1, 3: 0.2}), "越界或缺失的 index"),
        (payload({0: 0.1, 0.5: 0.2}), "越界或缺失的 index"),
        ({"results": [{"index": 0, "relevance_score": 0.1}, {"index": 0, "relevance_score": 0.2}]}, "出现了两次"),
        ({"results": [{"index": 0}, {"index": 1}]}, "缺 relevance_score"),
        ({"results": {}}, "非列表"),
        ({"results": [{"index": 0, "relevance_score": 1.0}]}, "1 条重排分，期望 2 条"),
        ({"results": [{"index": 0, "relevance_score": 1.0}, 7]}, "非对象元素"),
    ],
)
def test_malformed_results_raise_instead_of_guessing(body, match):
    """索引越界/重复/缺分/不是列表——每一种都会让「哪条对应哪个分」变成猜测。"""

    rerank = ApiReranker(settings(), transport=FakeTransport(body)).scores

    with pytest.raises(RerankResponseError, match=match):
        rerank("问题", ["甲", "乙"])


def test_a_transport_failure_surfaces_as_a_backend_error():
    """后端抛错时原样包成 `RerankBackendError`，绝不返回一个假的「原始顺序」。"""

    transport = FakeTransport(None, raises=RuntimeError("连接被重置"))
    rerank = ApiReranker(settings(), transport=transport).scores

    with pytest.raises(RerankBackendError, match="连接被重置"):
        rerank("问题", ["甲"])


# --- 记账：只记服务端报的 token，取不到就记 unreported ------------------------


def test_reported_rerank_tokens_reads_the_rerank_endpoint_shape():
    """rerank 端点的用量在 `meta.tokens.input_tokens`，**不是** embeddings 那个 `usage`。"""

    assert reported_rerank_tokens(payload({0: 1.0}, input_tokens=59)) == 59
    # 形状缺失一律 None（= unreported），不按 0 记：0 会让账本看起来「这次不要钱」。
    assert reported_rerank_tokens({"results": []}) is None
    assert reported_rerank_tokens({"meta": {"tokens": {}}}) is None
    assert reported_rerank_tokens({"meta": {"tokens": {"input_tokens": "x"}}}) is None


def test_usage_meter_counts_inputs_and_tokens():
    meter = UsageMeter()
    rerank = ApiReranker(
        settings(),
        transport=FakeTransport(payload({0: 1.0, 1: 0.8}, input_tokens=59)),
        usage=meter,
    ).scores

    rerank("问题", ["甲", "乙"])

    assert meter.as_dict() == {"requests": 1, "inputs": 2, "tokens": 59, "unreported": 0}


def test_missing_usage_is_recorded_as_unreported_not_as_zero():
    meter = UsageMeter()
    rerank = ApiReranker(settings(), transport=FakeTransport(payload({0: 1.0}, input_tokens=None)), usage=meter).scores

    rerank("问题", ["甲"])

    assert meter.tokens == 0 and meter.unreported == 1


def test_usage_is_not_tracked_when_no_meter_was_asked_for():
    """默认不开账本：不传 meter / 不传路径时，重排一次也不写任何状态。"""

    reranker = ApiReranker(settings(), transport=FakeTransport(payload({0: 1.0})))
    assert reranker.usage is None


def test_build_reranker_writes_the_usage_ledger_when_given_a_path(tmp_path):
    """给了账本路径才落盘，且带上模型名——「这笔钱花在哪个模型上」不能丢。"""

    target = tmp_path / "rerank-usage.json"
    rerank = build_reranker(
        settings(model="BAAI/bge-reranker-v2-m3"),
        transport=FakeTransport(payload({0: 1.0}, input_tokens=59)),
        usage_path=target,
    )

    assert rerank("问题", ["甲"]) == [1.0]
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "model": "BAAI/bge-reranker-v2-m3",
        "requests": 1,
        "inputs": 1,
        "tokens": 59,
        "unreported": 0,
    }


# --- httpx 传输层：重试策略（4xx 不重试，5xx/429 重试到上限） -----------------


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body, ensure_ascii=False)

    def json(self):
        return self._body


class FakeClient:
    """按顺序吐出预设响应；记录每次调用（含 headers 与 payload）。"""

    def __init__(self, responses) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)

    def post(self, url, *, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._responses.pop(0)


def http_transport(responses, *, max_retries=2):
    client = FakeClient(responses)
    return (
        HttpRerankTransport(
            base_url="https://api.siliconflow.cn/v1",
            api_key="sk-fake",
            max_retries=max_retries,
            client=client,
        ),
        client,
    )


def test_http_transport_posts_to_the_rerank_path_with_bearer_auth(monkeypatch):
    monkeypatch.setattr("akr.storage.rerank.time.sleep", lambda _: None)
    transport, client = http_transport([FakeResponse(200, payload({0: 0.9}))])

    body = transport.create(model="m", query="q", documents=["甲"])

    assert body["results"][0]["relevance_score"] == 0.9
    call = client.calls[0]
    assert call["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert call["headers"]["Authorization"] == "Bearer sk-fake"
    # `return_documents=False`：`index` 已经足够把分值对上，回显候选只是多花流量。
    assert call["json"]["documents"] == ["甲"]
    assert call["json"]["return_documents"] is False
    # 不传 `top_n`：传了服务端就会截断，重排必须覆盖整个候选池（见 `_unpack` 的报错）。
    assert "top_n" not in call["json"]


def test_http_transport_does_not_retry_a_400(monkeypatch):
    """模型名填错（实测 400 `Model does not exist`）当下就该有人看：重试只是白等。"""

    sleeps: list[float] = []
    monkeypatch.setattr("akr.storage.rerank.time.sleep", sleeps.append)
    transport, client = http_transport(
        [FakeResponse(400, {"code": 20012, "message": "Model does not exist."})]
    )

    with pytest.raises(RerankBackendError, match="HTTP 400"):
        transport.create(model="BAAI/bge-large-zh-v1.5", query="q", documents=["甲"])

    assert len(client.calls) == 1
    assert sleeps == []


def test_http_transport_retries_5xx_up_to_the_limit_and_reports_the_last_error(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("akr.storage.rerank.time.sleep", sleeps.append)
    transport, client = http_transport(
        [FakeResponse(503), FakeResponse(503), FakeResponse(503)], max_retries=2
    )

    with pytest.raises(RerankBackendError, match="HTTP 503"):
        transport.create(model="m", query="q", documents=["甲"])

    assert len(client.calls) == 3  # 首次 + 2 次重试
    assert sleeps == [pytest.approx(0.5), pytest.approx(1.0)]  # 指数退避


def test_http_transport_retries_a_429_and_can_succeed(monkeypatch):
    """限流是**会好**的错，与 400 不同：退避后重试，成功就正常返回。"""

    sleeps: list[float] = []
    monkeypatch.setattr("akr.storage.rerank.time.sleep", sleeps.append)
    transport, client = http_transport(
        [FakeResponse(429), FakeResponse(200, payload({0: 0.8}))]
    )

    body = transport.create(model="m", query="q", documents=["甲"])

    assert body["results"][0]["relevance_score"] == 0.8
    assert len(client.calls) == 2
    assert sleeps == [pytest.approx(0.5)]


def test_http_transport_wraps_a_network_exception(monkeypatch):
    """网络层异常（超时、DNS）一律转成本模块的 `RerankBackendError`，不泄漏 httpx 类型。"""

    monkeypatch.setattr("akr.storage.rerank.time.sleep", lambda _: None)

    class Exploding:
        def post(self, *args, **kwargs):
            raise TimeoutError("read timeout")

    transport = HttpRerankTransport(
        base_url="https://api.siliconflow.cn/v1", api_key="sk-fake", client=Exploding()
    )

    with pytest.raises(RerankBackendError, match="TimeoutError"):
        transport.create(model="m", query="q", documents=["甲"])


def test_build_reranker_without_settings_reads_env(monkeypatch):
    """`settings=None` 才去读环境（缺 Key 当场报错）——显式给了 settings 就不碰 .env。"""

    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr("akr.config.load_dotenv", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        build_reranker()


def test_rerank_settings_defaults_to_the_free_siliconflow_model(monkeypatch):
    """默认模型名就是本轮在硅基流动实测可用的那个免费 cross-encoder。"""

    from akr.config import DEFAULT_RERANK_MODEL, RERANK_MODEL_ENV

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-fake")
    monkeypatch.delenv(RERANK_MODEL_ENV, raising=False)
    monkeypatch.setattr("akr.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("SILICONFLOW_BASE_URL", raising=False)

    resolved = RerankSettings.from_env()

    assert resolved.model == DEFAULT_RERANK_MODEL == "BAAI/bge-reranker-v2-m3"
    assert resolved.base_url == "https://api.siliconflow.cn/v1"


def test_settings_can_point_at_another_model(monkeypatch):
    """模型名是独立的一组配置：嵌入模型名填进 rerank 会 400，两者不可互换。"""

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-fake")
    monkeypatch.setenv("SILICONFLOW_RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    monkeypatch.setattr("akr.config.load_dotenv", lambda *args, **kwargs: None)

    assert RerankSettings.from_env().model == "Qwen/Qwen3-Reranker-0.6B"


def test_http_transport_rejects_a_non_object_json_body(monkeypatch):
    """服务端回 `[...]`（数组）时当场报 `RerankResponseError`，不是 `AttributeError`。"""

    monkeypatch.setattr("akr.storage.rerank.time.sleep", lambda _: None)
    transport, client = http_transport([FakeResponse(200, [1, 2, 3])])

    with pytest.raises(RerankResponseError, match="不是 JSON 对象"):
        transport.create(model="m", query="q", documents=["甲"])

    assert len(client.calls) == 1
