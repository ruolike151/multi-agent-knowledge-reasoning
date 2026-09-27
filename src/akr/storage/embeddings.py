"""嵌入层：默认走硅基流动（SiliconFlow）的 `BAAI/bge-large-zh-v1.5`。

为什么是 API 而不是本地模型：DeepSeek 官方 API 没有 `/embeddings` 端点（spec §8.2），
所以向量化必须另找出口。硅基流动提供 OpenAI 兼容的 `/v1/embeddings`，且
`BAAI/bge-large-zh-v1.5` 是中文检索的常用型号，同时免去本地 500 MB 的 `torch` 依赖
（`.[embed]` 保留为可选兜底路径，见 `build_local_embedder`）。

两条路径共用同一套约定：

- 文档侧不加前缀，查询侧加 bge 系列要求的检索指令前缀（`QUERY_PREFIX`）；
- 输出交给 Chroma 的 cosine 空间，因此统一做 L2 归一化；
- 空输入不打后端，直接返回 `[]`；
- 任何后端异常（缺依赖、鉴权失败、返回条数不符）都显式抛出，**不静默降级**成随机
  向量或零向量——否则检索会「看起来能用但永远查不到」。

「文档侧不加前缀」指的是**本模块不替调用方拼前缀**：文档身份前缀（公司 / 代码 / 报告期）
是**逐 chunk** 派生的，只有拿到 `Chunk` 的调用方才知道该拼什么（`akr.ingest.doc_meta`），
所以它由 `VectorStore(index_text_fn=...)` 在**嵌入前**施加，`build_embedder(prefix=...)`
只负责把传进来的文本原样送给后端。两边都改会变成「前缀拼两次」。

用量记账（2026-09-25 新增）：`ApiEncoder` 会在**每次请求后**读服务端返回的 `usage`，
累加进 `UsageMeter`（请求数 / 输入条数 / token 数 / 缺 usage 的响应数）。存在的理由：
重建向量索引要花钱，而钱只能从服务端的 `total_tokens` 推，不能从字符数反推
（`MAX_INPUT_CHARS=500` 的截断、中英混排、前缀挤占预算都会让「字符→token」的换算失真）。
没带 `usage` 的假后端（单测）记 `unreported`，绝不按 0 记账。
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from akr.config import EmbeddingSettings
from akr.storage.vector_store import EmbedFn

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
MAX_SEQ_LENGTH = 512
INSTALL_HINT = 'pip install -e ".[embed]"'

# 硅基流动 embedding：一次请求的条数（保守值，可用 batch_size 调整）与网络参数。
API_BATCH_SIZE = 16
API_TIMEOUT_SECONDS = 30.0
API_MAX_RETRIES = 2

# 单条输入字符上限。bge 系列最大输入 512 token，中文近似「一字一 token」，故留出余量；
# 超长文本按字符截断，与本地路径 `max_seq_length=512` 的截断行为等价，
# 否则分块上限（800 字符 + 100 重叠）会直接被服务端拒绝。
MAX_INPUT_CHARS = 500


class EmbeddingBackendMissing(RuntimeError):
    """本地嵌入依赖缺失。显式报错，绝不静默降级成随机向量。"""


class EmbeddingResponseError(RuntimeError):
    """嵌入服务返回的数据与请求不匹配（条数 / 形状 / 零向量）。"""


@dataclass
class UsageMeter:
    """嵌入用量账本：只累加服务端返回的 `usage`，不按字符数猜 token。"""

    requests: int = 0
    inputs: int = 0
    tokens: int = 0
    unreported: int = 0

    def add(self, *, inputs: int, tokens: int | None) -> None:
        self.requests += 1
        self.inputs += inputs
        if tokens is None:
            self.unreported += 1
        else:
            self.tokens += int(tokens)

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "inputs": self.inputs,
            "tokens": self.tokens,
            "unreported": self.unreported,
        }


def reported_tokens(response: Any) -> int | None:
    """从响应体里取本次用量（`total_tokens` 优先，退到 `prompt_tokens`）；取不到返回 None。"""

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    for field in ("total_tokens", "prompt_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            return int(value)
    return None


def dump_usage(meter: UsageMeter, path: str | Path, *, model: str = "") -> None:
    """把账本落盘（每批覆盖写一次，进程被杀也留得住已经花掉的钱）。

    空 `model` **不会**擦掉已经记下来的模型名：`ApiEncoder` 每批都带着模型名落盘，收尾的
    汇总写如果不带，就会把「这笔钱花在哪个模型上」抹掉——账本不该越写信息越少。
    """

    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    if not model and target.exists():
        model = _recorded_model(target)
    payload = {"model": model, **meter.as_dict()}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _recorded_model(path: Path) -> str:
    """读回上一次落盘的模型名；文件缺失/被手改坏/字段不是字符串都当成「没有」。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("model")
    return value if isinstance(value, str) else ""


class Encoder(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool,
        batch_size: int,
    ) -> Sequence[Sequence[float]]: ...


class EmbeddingTransport(Protocol):
    """OpenAI SDK `client.embeddings` 的最小接口，单测用假实现替换它。"""

    def create(self, **kwargs: Any) -> Any: ...


def prepare_texts(texts: Sequence[str], *, prefix: str = "") -> list[str]:
    prepared = [text.strip() for text in texts]
    return [prefix + text for text in prepared if text]


def _unit(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise EmbeddingResponseError("嵌入服务返回了零向量：无法用于 cosine 检索")
    return [value / norm for value in values]


# ---------------------------------------------------------------------------
# 默认路径：硅基流动（SiliconFlow）OpenAI 兼容端点
# ---------------------------------------------------------------------------


class ApiEncoder:
    """把 OpenAI 兼容的 `/embeddings` 端点包成 `Encoder`（与本地编码器同签名）。"""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        transport: EmbeddingTransport | None = None,
        batch_size: int = API_BATCH_SIZE,
        usage: UsageMeter | None = None,
        usage_path: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._batch_size = batch_size
        # 记账是**可选**的：传了 meter（或落盘路径）才累加，默认不开销任何内存/IO。
        self.usage = usage if usage is not None else (UsageMeter() if usage_path else None)
        self._usage_path = Path(usage_path) if usage_path else None
        if transport is None:
            from openai import OpenAI

            transport = OpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=API_TIMEOUT_SECONDS,
                max_retries=API_MAX_RETRIES,
            ).embeddings
        self._transport = transport

    @property
    def model(self) -> str:
        return self._settings.model

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool,
        batch_size: int,
    ) -> list[list[float]]:
        limit = batch_size or self._batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), limit):
            window = [text[:MAX_INPUT_CHARS] for text in texts[start : start + limit]]
            response = self._transport.create(model=self._settings.model, input=window)
            self._record(window, response)
            vectors.extend(self._unpack(response, expected=len(window)))
        return [_unit(vector) for vector in vectors] if normalize_embeddings else vectors

    def _record(self, window: Sequence[str], response: Any) -> None:
        if self.usage is None:
            return
        self.usage.add(inputs=len(window), tokens=reported_tokens(response))
        if self._usage_path is not None:
            dump_usage(self.usage, self._usage_path, model=self._settings.model)

    def _unpack(self, response: Any, *, expected: int) -> list[list[float]]:
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != expected:
            raise EmbeddingResponseError(
                f"{self._settings.model} 返回 {len(data)} 条向量，期望 {expected} 条"
            )
        return [[float(value) for value in item.embedding] for item in data]


def build_api_embedder(
    settings: EmbeddingSettings | None = None,
    *,
    transport: EmbeddingTransport | None = None,
    batch_size: int = API_BATCH_SIZE,
    prefix: str = "",
    usage: UsageMeter | None = None,
    usage_path: str | Path | None = None,
) -> EmbedFn:
    """`settings=None` 时读 `.env`（缺 Key 立刻抛 RuntimeError，不发起任何请求）。"""
    if settings is None:
        settings = EmbeddingSettings.from_env()
    encoder = ApiEncoder(
        settings,
        transport=transport,
        batch_size=batch_size,
        usage=usage,
        usage_path=usage_path,
    )
    return _wrap(encoder, prefix=prefix, batch_size=batch_size)


def build_embedder(
    settings: EmbeddingSettings | None = None,
    *,
    transport: EmbeddingTransport | None = None,
    batch_size: int = API_BATCH_SIZE,
    prefix: str = "",
    usage: UsageMeter | None = None,
    usage_path: str | Path | None = None,
) -> EmbedFn:
    """默认嵌入器（文档侧）：硅基流动 `BAAI/bge-large-zh-v1.5`，不加检索前缀。

    `prefix` 只用于**测试或单文本场景**：生产路径的文档身份前缀依赖 `Chunk` 元数据，
    由 `VectorStore(index_text_fn=...)` 拼（见模块 docstring）。
    """

    return build_api_embedder(
        settings,
        transport=transport,
        batch_size=batch_size,
        prefix=prefix,
        usage=usage,
        usage_path=usage_path,
    )


def build_query_embedder(
    settings: EmbeddingSettings | None = None,
    *,
    transport: EmbeddingTransport | None = None,
    batch_size: int = API_BATCH_SIZE,
) -> EmbedFn:
    """查询侧嵌入：bge 系列要求查询加检索指令前缀，文档侧不加。"""
    return build_api_embedder(
        settings, transport=transport, batch_size=batch_size, prefix=QUERY_PREFIX
    )


# ---------------------------------------------------------------------------
# 可选兜底路径：本地 sentence-transformers（`pip install -e ".[embed]"`）
# ---------------------------------------------------------------------------


def load_encoder(model_name: str = DEFAULT_LOCAL_MODEL) -> Encoder:
    """延迟导入 sentence-transformers：只在真正需要嵌入时付出加载代价。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingBackendMissing(
            f"未安装本地嵌入依赖，无法加载 {model_name}。请执行：{INSTALL_HINT}"
        ) from exc
    model = SentenceTransformer(model_name)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def build_local_embedder(
    model_name: str = DEFAULT_LOCAL_MODEL,
    *,
    batch_size: int = 32,
    loader: Callable[[str], Encoder] = load_encoder,
) -> EmbedFn:
    return _wrap(loader(model_name), prefix="", batch_size=batch_size)


def build_local_query_embedder(
    model_name: str = DEFAULT_LOCAL_MODEL,
    *,
    batch_size: int = 8,
    loader: Callable[[str], Encoder] = load_encoder,
) -> EmbedFn:
    return _wrap(loader(model_name), prefix=QUERY_PREFIX, batch_size=batch_size)


def _wrap(encoder: Encoder, *, prefix: str, batch_size: int) -> EmbedFn:
    def embed(texts: list[str]) -> list[list[float]]:
        prepared = prepare_texts(texts, prefix=prefix)
        if not prepared:
            return []
        vectors = encoder.encode(
            prepared,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        return [[float(value) for value in vector] for vector in vectors]

    return embed
