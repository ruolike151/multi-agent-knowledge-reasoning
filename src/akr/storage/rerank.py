"""重排层：走硅基流动（SiliconFlow）的 `/v1/rerank`，默认模型 `BAAI/bge-reranker-v2-m3`。

**为什么是 cross-encoder，而不是「拿嵌入模型再算一次余弦」。** 两者不是一类东西：

* bi-encoder（本项目 `akr.storage.embeddings` 用的 `BAAI/bge-large-zh-v1.5`）= 查询与文档
  **各自**过一遍编码器，文档向量可以在建库时算好，检索退化成一次向量近邻；代价是「查询
  与文档从没在同一个上下文里被看过」，细粒度的「这句问的到底是不是这一行」判不出来。
* cross-encoder（本模块）= 把 `[查询][文档]` **拼成一条序列**送进同一个编码器，直接输出一个
  相关性标量。它不能预计算、每个候选都要一次前向，所以只能挂在「粗排之后、截断之前」。
* LLM listwise 重排 = 一次把 N 条候选塞进 prompt 让模型排序，最贵、最慢，本仓没走这条。

因此：**拿嵌入模型做重排 = 余弦相似度**，那正是 `akr.storage.fusion.cosine_rerank`
（工具层 `rerank="cosine"`）已经实现、并且**实测全负**的那条路（第三十八节）。真要做重排，
就只能是 cross-encoder 这一路。

**为什么走 API 而不是本地 `FlagEmbedding`。** 与嵌入同一个网关、同一把 Key，本地不用装
2~3 GB 的 `torch`（本仓刻意避开了这个依赖，见 `embeddings.py` 的模块 docstring）。代价是
重排**不能离线预计算**，每次检索都要多一次网络往返——所以它的位置在工具层的**可选旋钮**上，
不进默认路径（默认 `rerank="rrf"` 一行都不走这里）。

**模型名不可互换（实测）**：`BAAI/bge-large-zh-v1.5` 填进 `/v1/rerank` 返回
`400 {"code":20012,"message":"Model does not exist."}`——它是嵌入模型，只有 `/v1/embeddings`
认它。所以 `SILICONFLOW_RERANK_MODEL` 是**独立**的一组配置（`akr.config.RerankSettings`）。

三条口径与嵌入侧对齐：

* 单条候选截断到 `MAX_DOC_CHARS`（与 `embeddings.MAX_INPUT_CHARS` 同一个数）：工具层回给
  模型的片段本来就只有几百字符，不截断只会把 token 花在候选链路的尾部；
* 空候选不打后端、直接返回 `[]`；
* 后端异常（鉴权失败、条数不符、索引缺号）一律**显式抛出**，绝不静默退化成「不重排」——
  静默退化会让「重排有没有用」这个问题永远量不出来。

用量记账沿用嵌入那一份 `UsageMeter`（同一份账本形状），取的是响应里
`meta.tokens.input_tokens`——不是 `usage`，这是硅基流动 rerank 端点的字段形状。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from akr.config import RerankSettings
from akr.storage.embeddings import UsageMeter, dump_usage

# 硅基流动 rerank 端点的路径（base_url 已含 `/v1`）。
RERANK_PATH = "/rerank"

API_TIMEOUT_SECONDS = 30.0
API_MAX_RETRIES = 2
# 退避基数：重试之间的等待秒数按 `BACKOFF_BASE * 2**attempt` 增长（最多重试 2 次）。
BACKOFF_BASE_SECONDS = 0.5

# 单条候选字符上限。与 `embeddings.MAX_INPUT_CHARS` 同值：两个端点共用同一个 512 token 的
# 编码器家族预算，一边截 500 另一边不截，等于让「截断」这件事有两套口径。
MAX_DOC_CHARS = 500


class RerankBackendError(RuntimeError):
    """重排服务不可用（网络 / 鉴权 / 5xx）。显式报错，绝不静默退化成不重排。"""


class RerankResponseError(RuntimeError):
    """重排服务返回的数据与请求不匹配（条数 / 索引 / 分值字段）。"""


# 工具层 `ToolContext.reranker` 的类型：`(query, documents) -> 与 documents 等长的分值列表`。
RerankFn = Callable[[str, Sequence[str]], list[float]]


class RerankTransport(Protocol):
    """`/v1/rerank` 的最小接口，单测用假实现替换它（与 `EmbeddingTransport` 同一套路）。"""

    def create(
        self, *, model: str, query: str, documents: Sequence[str]
    ) -> Mapping[str, Any]: ...


class HttpRerankTransport:
    """`httpx` 版实现：硅基流动的 rerank 不是 OpenAI 标准端点，SDK 里没有对应资源。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = API_TIMEOUT_SECONDS,
        max_retries: int = API_MAX_RETRIES,
        client: Any | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + RERANK_PATH
        self._api_key = api_key
        self._max_retries = max_retries
        if client is None:
            import httpx

            client = httpx.Client(timeout=timeout)
        self._client = client

    def create(
        self, *, model: str, query: str, documents: Sequence[str]
    ) -> Mapping[str, Any]:
        payload = {
            "model": model,
            "query": query,
            "documents": list(documents),
            # 不回显候选原文：`index` 已经足够把分值与输入对上，回显只是多花流量。
            "return_documents": False,
        }
        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    self._url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
            except Exception as exc:  # noqa: BLE001 - 网络层的异常一律转成本模块的错
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    body = response.json()
                    if not isinstance(body, Mapping):
                        raise RerankResponseError(
                            f"{model} 返回的不是 JSON 对象：{type(body).__name__}"
                        )
                    return body
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                # 4xx（除 429）重试没有意义：模型名写错、余额不足都是当场要人看的错。
                if response.status_code < 500 and response.status_code != 429:
                    break
            if attempt < self._max_retries:
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
        raise RerankBackendError(f"调用 {self._url} 失败（model={model}）：{last_error}")


def reported_rerank_tokens(payload: Mapping[str, Any]) -> int | None:
    """从响应体里取本次用量（`meta.tokens.input_tokens`）；取不到返回 None，绝不按 0 记账。

    字段缺失、形状不对、或者**解析不出整数**（`"x"`）：一律返回 None，账本里记成一条
    `unreported`。记账字段解析失败不该把一次检索打挂，但它也不能被吞掉——`unreported`
    计数就是那条可见的痕迹（`UsageMeter.as_dict`）。
    """

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        return None
    tokens = meta.get("tokens")
    if not isinstance(tokens, Mapping):
        return None
    value = tokens.get("input_tokens")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ApiReranker:
    """把硅基流动的 `/v1/rerank` 包成 `RerankFn`（输入查询 + 候选原文，输出对齐的分值）。"""

    def __init__(
        self,
        settings: RerankSettings,
        *,
        transport: RerankTransport | None = None,
        usage: UsageMeter | None = None,
        usage_path: str | Path | None = None,
        max_doc_chars: int = MAX_DOC_CHARS,
    ) -> None:
        self._settings = settings
        self._max_doc_chars = max_doc_chars
        # 记账是**可选**的：传了 meter（或落盘路径）才累加，默认不开销任何内存/IO。
        self.usage = usage if usage is not None else (UsageMeter() if usage_path else None)
        self._usage_path = Path(usage_path) if usage_path else None
        if transport is None:
            transport = HttpRerankTransport(
                base_url=settings.base_url, api_key=settings.api_key
            )
        self._transport = transport

    @property
    def model(self) -> str:
        return self._settings.model

    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        """逐条候选的重排分，**顺序与入参一一对应**（不是按分值降序返回）。

        服务端返回的是 `index` + `relevance_score` 的列表（不保证按输入顺序），这里按 `index`
        还原成输入顺序：调用方拿到的 `scores[i]` 必须真的是 `documents[i]` 的分，否则重排会
        把候选和分值错配——那比不重排更糟。
        """

        docs = [text[: self._max_doc_chars] for text in documents]
        if not docs:
            return []
        try:
            payload = self._transport.create(
                model=self._settings.model, query=query, documents=docs
            )
        except RerankBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - 传输层的任何异常都是「后端不可用」
            # 本模块对外的失败只有两种形状：后端不可用、响应对不上。传输层自己抛什么类型，
            # 调用方（工具层）不该去猜——否则「重排失败」在日志里会长成两种样子。
            raise RerankBackendError(f"调用 {self._settings.model} 重排失败：{exc}") from exc
        self._record(len(docs), payload)
        return _unpack(payload, expected=len(docs), model=self._settings.model)

    def _record(self, count: int, payload: Mapping[str, Any]) -> None:
        if self.usage is None:
            return
        self.usage.add(inputs=count, tokens=reported_rerank_tokens(payload))
        if self._usage_path is not None:
            dump_usage(self.usage, self._usage_path, model=self._settings.model)


def _unpack(payload: Mapping[str, Any], *, expected: int, model: str) -> list[float]:
    """`{"results": [{"index": i, "relevance_score": s}, ...]}` → 按输入顺序的分值列表。"""

    results = payload.get("results")
    if not isinstance(results, list) or len(results) != expected:
        got = len(results) if isinstance(results, list) else "非列表"
        raise RerankResponseError(
            f"{model} 返回 {got} 条重排分，期望 {expected} 条"
            "（服务端按 top_n 截断过？重排必须覆盖整个候选池）"
        )
    slots: list[float | None] = [None] * expected
    for item in results:
        if not isinstance(item, Mapping):
            raise RerankResponseError(f"{model} 的 results 里有非对象元素：{item!r}")
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < expected:
            raise RerankResponseError(f"{model} 返回了越界或缺失的 index：{index!r}")
        if score is None:
            raise RerankResponseError(f"{model} 的第 {index} 条缺 relevance_score")
        if slots[index] is not None:
            raise RerankResponseError(f"{model} 的 index={index} 出现了两次")
        slots[index] = float(score)
    missing = [position for position, value in enumerate(slots) if value is None]
    if missing:
        raise RerankResponseError(f"{model} 没给这些候选打分：{missing}")
    return [value for value in slots if value is not None]


def build_reranker(
    settings: RerankSettings | None = None,
    *,
    transport: RerankTransport | None = None,
    usage: UsageMeter | None = None,
    usage_path: str | Path | None = None,
) -> RerankFn:
    """`settings=None` 时读 `.env`（缺 Key 立刻抛 RuntimeError，不发起任何请求）。"""

    if settings is None:
        settings = RerankSettings.from_env()
    return ApiReranker(
        settings, transport=transport, usage=usage, usage_path=usage_path
    ).scores
