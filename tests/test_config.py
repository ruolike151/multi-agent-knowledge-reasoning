import pytest

from akr.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingSettings,
    Settings,
)


def test_missing_api_key_fails_fast(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        Settings.from_env(env_file=None)


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    settings = Settings.from_env(env_file=None)
    assert settings.api_key == "sk-test"
    assert settings.model == "deepseek-flash"
    assert settings.base_url == "https://api.deepseek.com"


def test_embedding_settings_fail_fast_without_key(monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        EmbeddingSettings.from_env(env_file=None)


def test_embedding_settings_default_to_siliconflow_bge_large(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-sf-test")
    monkeypatch.delenv("SILICONFLOW_BASE_URL", raising=False)
    monkeypatch.delenv("SILICONFLOW_EMBEDDING_MODEL", raising=False)
    settings = EmbeddingSettings.from_env(env_file=None)
    assert settings.api_key == "sk-sf-test"
    assert settings.base_url == DEFAULT_EMBEDDING_BASE_URL == "https://api.siliconflow.cn/v1"
    assert settings.model == DEFAULT_EMBEDDING_MODEL == "BAAI/bge-large-zh-v1.5"


def test_embedding_settings_honor_env_overrides(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-sf-test")
    monkeypatch.setenv("SILICONFLOW_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")
    settings = EmbeddingSettings.from_env(env_file=None)
    assert settings.base_url == "https://example.test/v1"
    assert settings.model == "BAAI/bge-m3"
