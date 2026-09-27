"""`scripts/fetch_corpus.py` 纯函数单测（不联网、不下载）。

脚本放在 scripts/ 下而非 src/ 下（它是一次性的语料准备工具，不是运行时模块），
因此用 importlib 按路径加载。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_corpus.py"


@pytest.fixture(scope="module")
def fc():
    spec = importlib.util.spec_from_file_location("fetch_corpus", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass 在解析 `from __future__ import annotations` 的注解时要按
    # cls.__module__ 回查 sys.modules，因此必须先注册再执行。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_company_chain_covers_five_photovoltaic_stages(fc):
    assert [c.code for c in fc.COMPANIES] == [
        "600438",
        "002129",
        "601012",
        "002459",
        "300274",
    ]
    # 6 开头走沪市 column，0/3 开头走深市
    assert {c.code: c.column for c in fc.COMPANIES}["600438"] == "sse"
    assert {c.code: c.column for c in fc.COMPANIES}["300274"] == "szse"


def test_pick_annual_reports_excludes_summary_and_english(fc):
    items = [
        {"announcementTitle": "XX公司2025年年度报告", "announcementTime": 3},
        {"announcementTitle": "XX公司2025年年度报告摘要", "announcementTime": 2},
        {"announcementTitle": "XX公司2025年年度报告（英文版）", "announcementTime": 1},
        {"announcementTitle": "XX公司2024年年度报告", "announcementTime": 4},
        {"announcementTitle": "XX公司2025年半年度报告", "announcementTime": 5},
    ]
    picked = fc.pick_annual_reports(items, fc.COMPANIES[0])
    assert sorted(picked) == [2024, 2025]
    assert picked[2025]["announcementTime"] == 3


def test_candidate_announcements_buckets_and_excludes_noise(fc):
    items = [
        {"announcementTitle": "关于签订重大合同的公告", "announcementTime": 10},
        {"announcementTitle": "关于签订重大合同终止的公告", "announcementTime": 20},
        {"announcementTitle": "对外投资管理制度（2025年11月）", "announcementTime": 30},
        {"announcementTitle": "关于2026年度日常关联交易预计的公告", "announcementTime": 40},
        {"announcementTitle": "2025年度业绩预告", "announcementTime": 50},
        {"announcementTitle": "2024年半年度业绩预告", "announcementTime": 60},
        {"announcementTitle": "关于参加投资者网上集体接待日活动的公告", "announcementTime": 70},
    ]
    buckets = fc.candidate_announcements(items)
    assert [it["clean_title"] for it in buckets["重大合同"]] == ["关于签订重大合同的公告"]
    assert buckets["对外投资"] == []
    assert len(buckets["关联交易"]) == 1
    # 优先关键词（年份 + 年度业绩预告）命中者排在时间更晚的半年度预告之前
    assert [it["clean_title"] for it in buckets["业绩预告"]] == [
        "2025年度业绩预告",
        "2024年半年度业绩预告",
    ]
    assert len(buckets["投资者关系活动记录"]) == 1


def test_score_announcement_ranks_amount_and_peers(fc):
    peers = ["隆基绿能", "晶澳科技"]
    score, info = fc.score_announcement(
        "重大合同", "公司签订销售协议，合同金额 45.71GW，客户含隆基绿能。", peers
    )
    assert (score, info["has_amount"], info["mentions_peers"]) == (3, True, ["隆基绿能"])

    score, info = fc.score_announcement(
        "重大合同", "公司向通威太阳能有限公司增资 108,000 万元。", peers
    )
    assert (score, info["has_amount"], info["mentions_peers"]) == (2, True, [])

    score, _ = fc.score_announcement("重大合同", "本公告无任何数字与实体。", peers)
    assert score == 0


def test_score_announcement_relaxes_investor_relations_only(fc):
    body = "关于参加投资者网上集体接待日活动的公告，联系人：中环新能源科技股份有限公司。"
    score, info = fc.score_announcement("投资者关系活动记录", body, [])
    assert (score, info["filter_relaxed"]) == (1, True)
    # 同一条正文放到其他四类不做放宽
    assert fc.score_announcement("重大合同", body, [])[0] == 0


def test_period_label_uses_title_period(fc):
    cases = {
        "2025年度业绩预告": "2025年度",
        "2024年半年度业绩预告": "2024年半年度",
        "关于2026年度日常关联交易预计的公告": "2026年度",
        "关于参加2026年投资者网上集体接待日暨半年报业绩说明会活动的公告": "2026年半年度",
    }
    for title, expected in cases.items():
        item = {"clean_title": title, "announcementTime": 1767225600000}  # 2026-01-01
        assert fc.period_label(item) == expected
    # 标题里没有年份时退回披露年份
    assert fc.period_label({"clean_title": "关于中标公告", "announcementTime": 1745107200000}) == "2025年度"
