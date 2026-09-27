"""`scripts/retrieval_tool_smoke.py` 的单测。

与 `tests/test_retrieval_ab.py` 同一套做法：importlib 按路径加载脚本，**不触网**。
真实对账（真实 Chroma + 真实嵌入）由脚本自己跑，单测里只验纯函数与默认口径。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrieval_tool_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("retrieval_tool_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load()


def test_mismatches_reports_only_disagreeing_rows() -> None:
    expected = {"a": 1, "b": None, "c": 7}
    actual = {"a": 1, "b": None, "c": 8}
    assert smoke.mismatches(expected, actual) == [("c", 7, 8)]


def test_mismatches_treats_a_task_present_on_only_one_side_as_a_difference() -> None:
    """少算一道题比名次差一名更可疑：那说明两边看的不是同一批题。"""

    assert smoke.mismatches({"a": 1}, {"a": 1, "b": 3}) == [("b", None, 3)]
    assert smoke.mismatches({"a": 1, "b": 3}, {"a": 1}) == [("b", 3, None)]


def test_matches_distinguishes_none_from_a_number() -> None:
    """MISS（None）与第 9 名不是一回事，不能被当成相等。"""

    assert smoke.mismatches({"a": None}, {"a": 9}) == [("a", None, 9)]


def test_hit_at_counts_only_ranks_within_depth() -> None:
    ranks = {"a": 1, "b": 8, "c": 9, "d": None}
    assert smoke.hit_at(ranks, 8) == 2
    assert smoke.hit_at(ranks, 50) == 3
    assert smoke.hit_at({}, 8) == 0


def test_default_inputs_are_the_same_files_the_offline_ab_scripts_use() -> None:
    """对账脚本的默认输入必须与离线表、线上装配指向同一份数据，否则对的是两批事实。"""

    from akr.mcp_server import server

    assert smoke.DEFAULT_BM25 == server.DEFAULT_BM25_PATH
    assert smoke.DEFAULT_MANIFEST == server.DEFAULT_MANIFEST_PATH
    assert smoke.DEFAULT_CHROMA == server.DEFAULT_CHROMA_PATH


def test_default_config_is_the_paired_metadata_configuration() -> None:
    """默认对照的是「前缀 + 收窄」成对那一档——只上前缀是退步的那一档。"""

    assert smoke.DEFAULT_CONFIG == "stage2@8/M3+meta+narrow"
    assert (smoke.DEFAULT_TOP_K, smoke.DEFAULT_TWO_STAGE_DOCS) == (8, 3)


def test_evidence_records_ranks_not_just_the_verdict() -> None:
    """证据里要有**名次表**：结论能被复算，才不是一句没法核对的话。"""

    payload = smoke.evidence(
        config="stage2@8/M3+meta+narrow",
        top_k=8,
        two_stage_docs=3,
        expected={"a": 1, "b": None},
        actual={"a": 1, "b": 2},
    )
    assert payload["offline_hits"] == 1 and payload["online_hits"] == 2
    assert payload["total"] == 2
    assert payload["diffs"] == [{"task_id": "b", "offline": None, "online": 2}]
    assert payload["ranks"]["online"] == {"a": 1, "b": 2}


def test_default_db_is_the_same_sqlite_file_the_online_server_opens() -> None:
    """对账用的 SQLite 必须与线上装配指向同一个文件，否则邻块取的是另一套 ordinal。"""

    from akr.mcp_server import server

    assert smoke.DEFAULT_DB == server.DEFAULT_DB_PATH


def test_evidence_records_the_new_retrieval_knobs() -> None:
    """证据里要留下这一轮用的池深/重排/扩窗三个值：否则名次表复算不出来。

    默认那组必须跟着**线上口径**走（池深 50、`rrf_k` 20、`rerank=cross` + 锁 4、折叠 `rrf_sum`）
    ——它记的是「线上真正返回的顺序」。要复算第三十四~三十八节的历史表，就显式传
    `pool_depth=0` / `rerank="rrf"` / `fold="best"`。
    """

    plain = smoke.evidence(
        config="stage2@8/M3+meta+narrow",
        top_k=8,
        two_stage_docs=3,
        expected={"a": 1},
        actual={"a": 1},
    )
    assert (plain["pool_depth"], plain["rerank"], plain["expand_neighbors"]) == (50, "cross", 0)
    assert (plain["rrf_k"], plain["rerank_lock"]) == (20, 4)
    assert plain["fold"] == smoke.DEFAULT_FOLD == "rrf_sum"

    knobbed = smoke.evidence(
        config="expand@8/prod±2/3anchors",
        top_k=8,
        two_stage_docs=3,
        expected={"a": 1},
        actual={"a": 1},
        pool_depth=24,
        rerank="cosine",
        rrf_k=40,
        rerank_lock=4,
        expand_neighbors=2,
        fold="best",
    )
    assert (knobbed["pool_depth"], knobbed["rerank"], knobbed["expand_neighbors"]) == (24, "cosine", 2)
    assert (knobbed["rrf_k"], knobbed["rerank_lock"]) == (40, 4)
    assert knobbed["fold"] == "best"


def test_the_new_cli_choices_come_from_the_tool_layer_not_a_local_copy() -> None:
    """`--rerank` 的白名单与扩窗默认值必须与工具层同源：另抄一份就是等它漂移。"""

    from akr.mcp_server import tools

    assert smoke.RERANK_MODES == tools.RERANK_MODES
    assert smoke.RERANK_DEFAULT == tools.RERANK_DEFAULT
    assert smoke.POOL_DEPTH_DEFAULT == tools.POOL_DEPTH_DEFAULT
    assert smoke.RRF_K_DEFAULT == tools.RRF_K_DEFAULT
    assert smoke.RERANK_LOCK_DEFAULT == tools.RERANK_LOCK_DEFAULT
    assert smoke.EXPAND_NEIGHBORS == tools.EXPAND_NEIGHBORS
