import os
from dataclasses import dataclass

from dotenv import load_dotenv

API_KEY_ENV = "DEEPSEEK_API_KEY"

# 嵌入服务（硅基流动 / SiliconFlow）——与 DeepSeek 的 chat 端点是两个独立的服务，
# 因此用一组独立的变量名，避免和 M0 基线冒烟用的 OPENAI_* 混在一起。
EMBEDDING_API_KEY_ENV = "SILICONFLOW_API_KEY"
EMBEDDING_BASE_URL_ENV = "SILICONFLOW_BASE_URL"
EMBEDDING_MODEL_ENV = "SILICONFLOW_EMBEDDING_MODEL"

DEFAULT_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# 重排服务：**同一家**（硅基流动）、**同一把 Key**、同一个 base_url，只有模型名不同。
# 为什么单独一组变量名而不是复用 `SILICONFLOW_EMBEDDING_MODEL`：嵌入是 bi-encoder、
# 重排是 cross-encoder，两者的模型**不可互换**（把嵌入模型名填进 `/v1/rerank` 会 400
# 「Model does not exist」，实测见内部偏差日志第四十节）。合成一个变量名，
# 迟早会有人在同一个进程里想用两种模型。
RERANK_MODEL_ENV = "SILICONFLOW_RERANK_MODEL"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    budget_cny: float = 5.0
    peak_pricing: bool = True

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file)
        key = os.getenv(API_KEY_ENV, "").strip()
        if not key:
            raise RuntimeError(
                f"{API_KEY_ENV} 未设置：请复制 .env.example 为 .env 并填入 API Key"
            )
        return cls(api_key=key)


@dataclass(frozen=True)
class EmbeddingSettings:
    """嵌入服务配置（默认：硅基流动的 `BAAI/bge-large-zh-v1.5`）。"""

    api_key: str
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    model: str = DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "EmbeddingSettings":
        if env_file:
            load_dotenv(env_file)
        key = os.getenv(EMBEDDING_API_KEY_ENV, "").strip()
        if not key:
            raise RuntimeError(
                f"{EMBEDDING_API_KEY_ENV} 未设置：请在 .env 里填入硅基流动（SiliconFlow）"
                "的 API Key（见 .env.example 的「嵌入服务」一节）"
            )
        return cls(
            api_key=key,
            base_url=os.getenv(EMBEDDING_BASE_URL_ENV, "").strip()
            or DEFAULT_EMBEDDING_BASE_URL,
            model=os.getenv(EMBEDDING_MODEL_ENV, "").strip() or DEFAULT_EMBEDDING_MODEL,
        )


@dataclass(frozen=True)
class RerankSettings:
    """重排服务配置（默认：硅基流动的 `BAAI/bge-reranker-v2-m3`）。

    Key 与 base_url 与嵌入**共用**（同一个网关、同一把 Key），缺 Key 的报错说明也照抄嵌入那一份：
    重排是可选的一路，但**一旦要用就必须能连上**，不允许静默退化成不重排。
    """

    api_key: str
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    model: str = DEFAULT_RERANK_MODEL

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "RerankSettings":
        if env_file:
            load_dotenv(env_file)
        key = os.getenv(EMBEDDING_API_KEY_ENV, "").strip()
        if not key:
            raise RuntimeError(
                f"{EMBEDDING_API_KEY_ENV} 未设置：重排与嵌入共用同一把硅基流动 Key"
                "（见 .env.example 的「嵌入服务」一节）"
            )
        return cls(
            api_key=key,
            base_url=os.getenv(EMBEDDING_BASE_URL_ENV, "").strip()
            or DEFAULT_EMBEDDING_BASE_URL,
            model=os.getenv(RERANK_MODEL_ENV, "").strip() or DEFAULT_RERANK_MODEL,
        )
