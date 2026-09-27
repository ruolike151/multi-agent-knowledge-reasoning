"""`scripts/build_unit_chunks.py` 的单测。

脚本不在 `src/` 下，按仓库既有做法用 importlib 按路径加载（与 `tests/test_retrieval_ab.py`
一致）。本文件**不解析 PDF、不触网、不读生产 `data/`**：只测落盘/读回的口径与闸门——
真实的 unit 切块由 `akr.ingest.unit_chunks` 自己测（`tests/ingest/test_unit_chunks.py`）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from akr.ingest.chunker import Chunk
from akr.ingest.unit_chunks import UnitChunk

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_unit_chunks.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_unit_chunks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load()


def make_unit(unit_id: str, parent_id: str, text: str, index_text: str) -> UnitChunk:
    return UnitChunk(
        chunk=Chunk(
            chunk_id=unit_id,
            doc_id="doc1",
            source="doc1.pdf",
            ordinal=0,
            text=text,
            content_hash="hash",
            section_path="第一节",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(text),
        ),
        parent_id=parent_id,
        index_text=index_text,
    )


def test_guard_target_refuses_the_two_libraries_that_are_already_baselines() -> None:
    """v1 是所有历史数字的基准、v2 是上一轮的产物：写进去会让既往结论无法复算。"""

    with pytest.raises(ValueError, match="不许覆盖"):
        build.guard_target("data/chroma")
    with pytest.raises(ValueError, match="不许覆盖"):
        build.guard_target("data/chroma-v2")
    # 新目录放行（闸门只拦既有基准，不拦正常建库）。
    build.guard_target("data/chroma-v3")


def test_dump_and_load_units_round_trip_every_field(tmp_path: Path) -> None:
    """落盘的是**唯一**事实来源：读回来必须逐字一样，否则建库口径与 dump 分叉。"""

    units = [
        make_unit("doc1#u0001", "doc1#c0001", "营收 100", "TCL中环（002129） 2024年\n营收 100"),
        make_unit("doc1#u0002", "doc1#c0001", "员工 5 人", "TCL中环（002129） 2024年\n员工 5 人"),
    ]
    target = tmp_path / "nested" / "units.json"

    build.dump_units(units, target)
    loaded = build.load_units(target)

    assert loaded == units


def test_load_units_rejects_an_empty_payload(tmp_path: Path) -> None:
    """空 dump 是「切了 0 条」的信号，不能被当成「这份语料没有 unit」静默放过。"""

    target = tmp_path / "units.json"
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="没有 unit 数组"):
        build.load_units(target)


def test_load_unit_map_returns_a_plain_id_mapping(tmp_path: Path) -> None:
    target = tmp_path / "unit-map.json"
    target.write_text(json.dumps({"doc1#u0001": "doc1#c0001"}), encoding="utf-8")
    assert build.load_unit_map(target) == {"doc1#u0001": "doc1#c0001"}


def test_load_unit_map_refuses_to_degrade_into_no_folding(tmp_path: Path) -> None:
    """缺映射时静默返回空表，会让 unit 级命中全落在 gold 空间之外（召回率假性归零）。"""

    target = tmp_path / "unit-map.json"
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="unit → 父块"):
        build.load_unit_map(target)
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unit → 父块"):
        build.load_unit_map(target)


def test_plan_reports_the_four_numbers_the_budget_check_needs(tmp_path: Path) -> None:
    """预演必须报「几条超窗口」——真跑同口径，不然预算只能靠真花钱才知道。"""

    units = [
        make_unit("doc1#u0001", "doc1#c0001", "短", "前缀\n短"),
        make_unit("doc1#u0002", "doc1#c0001", "长" * 600, "前缀\n" + "长" * 600),
    ]
    stats = build.plan(units, chroma_path=tmp_path / "chroma-v3")
    assert stats == {
        "units": 2,
        "parents": 1,
        "index_chars": len("前缀\n短") + len("前缀\n" + "长" * 600),
        "raw_chars": 1 + 600,
        "clipped": 1,
    }
