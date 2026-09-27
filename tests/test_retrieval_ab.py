"""`scripts/retrieval_ab.py` 的单测。

脚本不在 `src/` 下，按仓库既有做法用 importlib 按路径加载（与 `tests/test_rebuild_vector_index.py`
一致）。本文件**不触网、不读生产 `data/`**：只测纯函数——真实 A/B 的向量与 BM25 由脚本自己跑，
单测里绝不允许出现真实 API 调用。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrieval_ab.py"


def _load():
    spec = importlib.util.spec_from_file_location("retrieval_ab", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ab = _load()


def hit(chunk_id: str, doc_id: str, rank_score: float = 1.0) -> SimpleNamespace:
    """假命中：`rank_of` / `doc_rank_of` 只要 `chunk_id` 与 `doc_id`，`rrf_fuse` 还要 `retriever`。"""

    return SimpleNamespace(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source="data/corpus/x.pdf",
        score=rank_score,
        retriever="vector",
        text="",
    )


def test_normalize_cjk_joins_chinese_split_by_newline() -> None:
    """PDF 行尾切词（「硅\\n片」）必须被接回去——这是修法①针对的那类断裂。"""

    assert ab.normalize_cjk("硅\n片整体市占率") == "硅片整体市占率"
    assert ab.normalize_cjk("硅 \u3000 片") == "硅片"


def test_normalize_cjk_keeps_spaces_between_latin_and_chinese() -> None:
    """中英之间的空格是有意义的，不能被压掉，否则「002129 TCL」会粘成一个 token。"""

    assert ab.normalize_cjk("代码 002129 TCL\n中环") == "代码 002129 TCL中环"


def test_bigram_tokens_only_cjk() -> None:
    """bigram 只覆盖中文串：英文与数字的 2-gram 全是噪声。"""

    assert ab.bigram_tokens("硅片出货量") == ["硅片", "片出", "出货", "货量"]
    assert ab.bigram_tokens("TCL 2024") == []


def test_rank_of_returns_best_rank_and_none_on_miss() -> None:
    hits = [hit("a#1", "a"), hit("b#2", "b"), hit("a#3", "a")]
    assert ab.rank_of(["a#3", "b#2"], hits) == 2
    assert ab.rank_of(["a#3"], hits) == 3
    assert ab.rank_of(["z#9"], hits) is None


def test_doc_rank_of_uses_document_not_chunk() -> None:
    """文档级口径：第一个「属于 gold 文档」的命中就算命中，哪怕 chunk_id 不同。"""

    hits = [hit("x#1", "x"), hit("g#7", "gold-doc")]
    assert ab.doc_rank_of({"gold-doc"}, hits) == 2
    assert ab.doc_rank_of({"absent"}, hits) is None


def test_window_hit_requires_anchor_in_the_same_gold_document() -> None:
    """锚点必须落在 gold 文档内；别文档的 ordinal 恰好接近不算覆盖。"""

    ordinals = {"other#5": 5, "gold#21": 21, "gold#23": 23}
    doc_of = {"other#5": "other", "gold#21": "gold", "gold#23": "gold"}
    hits = [hit("other#5", "other"), hit("gold#21", "gold")]
    covered, rank = ab.window_hit(["gold#23"], doc_of, hits, ordinals, width=2)
    assert covered is True and rank == 2

    # 窗口收窄到 1 就够不着了：other#5 的 ordinal 再近也不算，因为文档不同。
    covered, rank = ab.window_hit(["gold#23"], doc_of, hits, ordinals, width=1)
    assert covered is False and rank is None


def test_merge_ranked_orders_by_best_rank_across_queries() -> None:
    """多个 retrieve 子任务的池子是并集，排序按「在任意一次查询里的最好名次」。"""

    first = [hit("a#1", "a"), hit("b#2", "b")]
    second = [hit("c#3", "c"), hit("a#1", "a")]
    merged = ab.merge_ranked([first, second])
    assert [item.chunk_id for item in merged] == ["a#1", "c#3", "b#2"]


def test_fused_respects_depth() -> None:
    """融合口径与工具层一致：两路各取 depth、RRF 再截 depth。"""

    vector = [hit(f"v#{index}", "d") for index in range(1, 6)]
    bm25 = [hit(f"b#{index}", "d") for index in range(1, 6)]
    assert len(ab.fused(vector, bm25, 3)) == 3
    assert len(ab.fused(vector, bm25, 50)) == 10


def test_load_chunks_fills_fields_missing_from_export(tmp_path: Path) -> None:
    """`data/chunks.jsonl` 的导出器不带溯源字段，加载时要补占位值而不是抛 TypeError。"""

    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "d#c1",
                "doc_id": "d",
                "ordinal": 0,
                "page_start": 1,
                "page_end": 1,
                "section_path": "",
                "source": "data/corpus/x.pdf",
                "text": "原文",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    chunks = ab.load_chunks(path)
    assert len(chunks) == 1
    assert chunks[0].text == "原文"
    assert chunks[0].content_hash == ""


def test_load_ours_goals_raises_when_join_missing(tmp_path: Path) -> None:
    """配对文件缺失要显式报错（不能退回「猜」run_id：评测与 trace 的主键本来就不同名）。"""

    try:
        ab.load_ours_goals(tmp_path / "akr.sqlite", tmp_path / "missing.json")
    except ab.AbError as error:
        assert "join_run_trace" in str(error)
    else:  # pragma: no cover - 没抛出就是坏了
        raise AssertionError("缺配对文件时必须抛 AbError")


def _row(ranks: dict[str, int | None], recalls: dict[str, float], doc_rank: int | None,
         covered: bool) -> dict:
    """一行逐题明细的最小骨架（只需 `summarize` 真正读的那几个键）。"""

    return {
        "ranks": ranks,
        "recalls": recalls,
        "doc_ranks": {"question@8": doc_rank},
        "window": {"window±2": {"covered": covered, "anchor_rank": None}},
    }


def test_mrr_at_takes_reciprocal_best_rank_and_zero_on_miss() -> None:
    """MRR 是 `ranks` 的纯函数：命中记 1/名次，MISS 记 0，再对题数求均值。"""

    rows = [
        _row({"c": 2}, {"c": 0.5}, 1, True),
        _row({"c": None}, {"c": 0.0}, 2, False),
    ]
    assert ab.mrr_at(rows, "c", 8) == 0.25


def test_mrr_at_ignores_ranks_beyond_k() -> None:
    """截断深度要真的生效：名次 30 在第 8 位口径下算 MISS，在第 50 位口径下算 1/30。"""

    rows = [_row({"c": 30}, {"c": 0.0}, None, False)]
    assert ab.mrr_at(rows, "c", 8) == 0.0
    assert ab.mrr_at(rows, "c", 50) == 1 / 30


def test_summarize_folds_rows_into_three_tables() -> None:
    """三张汇总表都只从 `rows` + `depth_of` 派生——这是复算通路成立的前提。"""

    rows = [
        _row({"a@8": 2, "b@50": 30}, {"a@8": 0.5, "b@50": 0.5}, 1, True),
        _row({"a@8": None, "b@50": 3}, {"a@8": 0.0, "b@50": 1.0}, 2, False),
    ]
    summary, doc_summary, window_summary = ab.summarize(rows, {"a@8": 8, "b@50": 50})

    assert summary["a@8"] == {
        "hit@8": 1,
        "hit@50": 1,
        "recall_k": 8,
        "recall": 0.25,
        "mrr@8": 0.25,
        "mrr@50": 0.25,
        "total": 2,
    }
    # b@50 的第 2 题名次 3 落在 top-8 里，所以 hit@8 也是 1；第 1 题名次 30 只在 top-50 见效。
    assert summary["b@50"]["hit@8"] == 1
    assert summary["b@50"]["hit@50"] == 2
    assert summary["b@50"]["mrr@8"] == round(1 / 6, 4)
    assert summary["b@50"]["mrr@50"] == round((1 / 30 + 1 / 3) / 2, 4)
    assert doc_summary == {"question@8": 2}
    assert window_summary == {"window±2": 1}


def test_resummarize_refreshes_summary_from_stored_ranks(tmp_path: Path) -> None:
    """加一列汇总不该逼人重跑检索：`--from-json` 只重算汇总，逐题名次原样保留。"""

    rows = [
        _row({"a@8": 2, "b@50": 30}, {"a@8": 0.5, "b@50": 0.5}, 1, True),
        _row({"a@8": None, "b@50": 3}, {"a@8": 0.0, "b@50": 1.0}, 2, False),
    ]
    source = tmp_path / "ab.json"
    source.write_text(
        json.dumps(
            {
                "depth": 50,
                "answerable": 2,
                # 旧版产物里没有 MRR 列——recall_k 是复算时唯一还需要的深度来源。
                "summary": {"a@8": {"recall_k": 8}, "b@50": {"recall_k": 50}},
                "doc_summary": {"question@8": 0},
                "window_summary": {"window±2": 0},
                "rows": rows,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert ab.resummarize(source, None) == 0
    payload = json.loads(source.read_text(encoding="utf-8"))
    assert payload["config_depth"] == {"a@8": 8, "b@50": 50}
    assert payload["summary"]["b@50"]["mrr@50"] == round((1 / 30 + 1 / 3) / 2, 4)
    assert payload["doc_summary"] == {"question@8": 2}
    assert payload["window_summary"] == {"window±2": 1}
    assert payload["rows"] == rows  # 名次一个字节都没动：复算不产生新事实


def chunk(chunk_id: str, ordinal: int, *, doc_id: str | None = None):
    """假块：`expand_window` 只读 `chunk_id` / `doc_id` / `ordinal` 三个字段。"""

    return ab.Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id or chunk_id.split("#")[0],
        source="data/corpus/x.pdf",
        ordinal=ordinal,
        text=f"text of {chunk_id}",
        content_hash="",
        section_path="第一章",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=0,
    )


def doc_chunks(doc_id: str, count: int) -> dict:
    """一份「文档内 ordinal 从 0 连续排到 count-1」的语料快照，键是 chunk_id。"""

    return {
        f"{doc_id}#c{index}": chunk(f"{doc_id}#c{index}", index, doc_id=doc_id)
        for index in range(count)
    }


def test_chunk_id_accepts_hits_and_plain_ids() -> None:
    """扩窗列返回的是纯 id 列表、别的列是命中对象——同一把尺子必须认这两种形状。"""

    assert ab._chunk_id("d#c1") == "d#c1"
    assert ab._chunk_id(hit("d#c1", "d")) == "d#c1"


def test_rank_of_accepts_plain_chunk_id_lists() -> None:
    """`rank_of` 直接吃 id 列表：否则扩窗列得先包一层假命中对象，白多一份形状转换。"""

    hits = ["a#1", "b#2", "a#3"]
    assert ab.rank_of(["a#3"], hits) == 3
    assert ab.rank_of(["b#2", "a#1"], hits) == 1
    assert ab.rank_of(["z#9"], hits) is None


def test_recall_at_accepts_plain_chunk_id_lists() -> None:
    """标准 recall 要能对 id 列表算：命中数 / gold 总数（与「命中率」不是一回事）。"""

    hits = ["a#1", "b#2", "a#3"]
    assert ab.recall_at(["a#1", "a#3"], hits, 8) == 1.0
    assert ab.recall_at(["a#3", "z#9"], hits, 1) == 0.0


def test_window_offsets_start_at_the_anchor_and_go_outwards() -> None:
    """锚点（0）先占位，再按距离由近到远；同距离先取下一条（`-1` 在 `+1` 之前）。"""

    assert ab._window_offsets(0) == (0,)
    assert ab._window_offsets(2) == (0, -1, 1, -2, 2)


def test_expand_window_emits_the_anchor_then_its_neighbours() -> None:
    ids = ab.expand_window(
        [hit("d#c2", "d")], anchors=1, width=2, top_k=8, chunk_at=doc_chunks("d", 5)
    )
    assert ids == ["d#c2", "d#c1", "d#c3", "d#c0", "d#c4"]


def test_expand_window_truncates_to_top_k() -> None:
    """扩出来的尾巴必须被截掉：返回给模型的条数永远等于 top_k。"""

    ids = ab.expand_window(
        [hit("d#c2", "d")], anchors=1, width=2, top_k=3, chunk_at=doc_chunks("d", 5)
    )
    assert ids == ["d#c2", "d#c1", "d#c3"]


def test_expand_window_never_crosses_the_document_boundary() -> None:
    """ordinal 只在文档内有意义：别的文档里 ordinal 恰好接近的块不算邻块。"""

    corpus = {**doc_chunks("d", 3), **doc_chunks("other", 3)}
    ids = ab.expand_window([hit("d#c2", "d")], anchors=1, width=5, top_k=8, chunk_at=corpus)
    assert ids == ["d#c2", "d#c1", "d#c0"]


def test_expand_window_dedupes_overlapping_anchors() -> None:
    """两个锚点挨着时邻块会重叠：同一条块只能出现一次，否则白占一个名额。"""

    ids = ab.expand_window(
        [hit("d#c0", "d"), hit("d#c1", "d")],
        anchors=2,
        width=1,
        top_k=8,
        chunk_at=doc_chunks("d", 3),
    )
    assert ids == ["d#c0", "d#c1", "d#c2"]


def test_expand_window_uses_only_the_first_anchors_hits() -> None:
    """`anchors` 是上限不是建议：第 2 条以后的命中不许参与扩窗。"""

    ids = ab.expand_window(
        [hit("d#c0", "d"), hit("d#c2", "d")],
        anchors=1,
        width=0,
        top_k=8,
        chunk_at=doc_chunks("d", 3),
    )
    assert ids == ["d#c0"]


def test_expand_window_skips_an_anchor_missing_from_the_corpus() -> None:
    """锚点不在语料快照里就跳过它：跨库（v1/v2）混用的锚点缺一条不该让整次测量崩掉。"""

    ids = ab.expand_window(
        [hit("d#cz", "d"), hit("d#c1", "d")],
        anchors=2,
        width=1,
        top_k=8,
        chunk_at=doc_chunks("d", 3),
    )
    assert ids == ["d#c1", "d#c0", "d#c2"]


def test_expand_window_returns_nothing_when_disabled() -> None:
    """`anchors=0`（默认关）与 `top_k=0` 都必须产出空列表，而不是「尽量扩一点」。"""

    corpus = doc_chunks("d", 3)
    assert (
        ab.expand_window([hit("d#c1", "d")], anchors=0, width=2, top_k=8, chunk_at=corpus) == []
    )
    assert (
        ab.expand_window([hit("d#c1", "d")], anchors=2, width=2, top_k=0, chunk_at=corpus) == []
    )


def test_expand_window_output_feeds_both_tables() -> None:
    """8 个新列同时进两张表：名次/recall 走 `by_config`，覆盖数走 `window`——同一列 id。"""

    ids = ab.expand_window(
        [hit("d#c0", "d")], anchors=1, width=2, top_k=8, chunk_at=doc_chunks("d", 3)
    )
    rows = [
        {
            "ranks": {"expand@8/x": ab.rank_of(["d#c2"], ids)},
            "recalls": {"expand@8/x": ab.recall_at(["d#c2"], ids, 8)},
            "doc_ranks": {},
            "window": {
                "expand@8/x": {
                    "covered": any(gold in set(ids) for gold in ["d#c2"]),
                    "anchor_rank": None,
                }
            },
        }
    ]
    summary, doc_summary, window_summary = ab.summarize(rows, {"expand@8/x": 8})
    assert summary["expand@8/x"]["hit@8"] == 1
    assert summary["expand@8/x"]["recall"] == 1.0
    assert doc_summary == {}
    assert window_summary == {"expand@8/x": 1}
