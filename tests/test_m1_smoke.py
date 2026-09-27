"""`scripts/m1_smoke.py` 的参数与快速失败路径单测。

与 `tests/test_fetch_corpus.py` 保持一致：脚本不在 src/ 下，用 importlib 按路径加载。
这里刻意不触发真实链路（不加载嵌入模型、不连 Neo4j），真实链路由脚本本身跑。
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m1_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("m1_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


m1_smoke = _load()


def test_missing_corpus_fails_fast_without_loading_model(monkeypatch, tmp_path, capsys):
    """语料缺失时必须立刻返回 1：不要先去加载嵌入模型再报错。"""
    monkeypatch.setattr(m1_smoke, "ANNUAL", tmp_path / "not-downloaded.pdf")
    assert m1_smoke.main([]) == 1
    assert "语料不存在" in capsys.readouterr().err


def test_smoke_targets_the_annual_report_of_the_named_doc():
    assert Path(m1_smoke.ANNUAL).name == "600438_通威股份_2025年报.pdf"
    assert m1_smoke.DOC_ID == "annual-600438-2025"
    assert 0 < m1_smoke.MAX_REEMBED_RATIO < 1


def test_smoke_defaults_to_the_siliconflow_api_embedder():
    """默认走硅基流动（`.env` 的 SILICONFLOW_API_KEY）；`--embedder local` 才用本地模型。"""
    default = inspect.signature(m1_smoke.build_context).parameters["embedder"].default
    assert default == "api"


def test_smoke_rejects_unknown_embedder():
    import pytest

    with pytest.raises(SystemExit):
        m1_smoke.main(["--embedder", "magic"])
