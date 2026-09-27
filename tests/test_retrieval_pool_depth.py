"""`scripts/retrieval_pool_depth.py` 的单测。

与 `tests/test_retrieval_ab.py` 同一套做法：importlib 按路径加载脚本，**不触网**。
真实曲线（真实 Chroma + 真实嵌入）由脚本自己跑，单测里只验纯函数与默认口径。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from akr.mcp_server.tools import ToolContext
from akr.storage.hits import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrieval_pool_depth.py"


def _load():
    spec = importlib.util.spec_from_file_location("retrieval_pool_depth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


depth = _load()


def _hit(chunk_id: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id)


def _rows() -> list[dict]:
    return [
        {"task_id": "t1", "gold_chunk_ids": ["a"]},
        {"task_id": "t2", "gold_chunk_ids": ["b", "c"]},
    ]


def test_metrics_reports_the_three_denominators_separately() -> None:
    """命中率的分母是题数、recall 的分母是 gold 条数——两个口径不能互相顶替。"""

    hits = {
        # t1：gold 落在第 3 名
        "t1": [_hit("x"), _hit("y"), _hit("a"), _hit("z")],
        # t2：两条 gold 只命中一条（第 1 名），标准 recall 记 1/2
        "t2": [_hit("b"), _hit("y"), _hit("z"), _hit("w")],
    }
    summary = depth.metrics(_rows(), hits, top_k=8)
    assert summary["total"] == 2
    assert summary["hit"] == 2 and summary["hit_rate"] == 1.0
    assert summary["recall"] == 0.75  # (1/1 + 1/2) / 2
    assert summary["mrr"] == pytest.approx((1 / 3 + 1 / 1) / 2, abs=1e-4)


def test_metrics_truncates_at_top_k_and_scores_a_miss_as_zero() -> None:
    """池子再深，进了返回条数之外的 gold 也不算命中，MRR 记 0。"""

    hits = {"t1": [_hit("x")] * 8 + [_hit("a")], "t2": [_hit("b")]}
    summary = depth.metrics(_rows(), hits, top_k=8)
    assert summary["hit"] == 1  # t1 的 gold 在第 9 名，出局
    assert summary["ranks"] == {"t1": None, "t2": 1}
    assert summary["mrr"] == pytest.approx(0.5, abs=1e-4)
    assert summary["recall"] == pytest.approx(0.25, abs=1e-4)  # (0 + 1/2) / 2


def test_mrr_at_empty_ranks_is_zero_not_a_crash() -> None:
    assert depth.mrr_at({}) == 0.0
    assert depth.mrr_at({"a": None, "b": 4}) == pytest.approx(1 / 8, abs=1e-4)


def test_parse_depths_sorts_and_dedupes() -> None:
    assert depth.parse_depths("24, 8, 24,16") == (8, 16, 24)


def test_parse_depths_rejects_a_non_positive_pool() -> None:
    """池深 0 等于不检索——它不是「便宜一档」，是静默返回空。"""

    with pytest.raises(depth.AbError):
        depth.parse_depths("0")
    with pytest.raises(depth.AbError):
        depth.parse_depths(" ,, ")
    # 尾随逗号是笔误不是配置，容忍掉；真正空的一串才报错（上面那条）。
    assert depth.parse_depths("8,, ") == (8,)


def test_default_inputs_are_the_same_files_the_other_retrieval_scripts_use() -> None:
    """基准列钉在 v1、线上列跟着线上装配走——这两件事都对，因为它们不是同一件事。

    线上 2026-09-26 已切到 unit 级的 `data/chroma-v3`；`--chroma`（这一列是第三十四~三十七节
    所有横向对照的**基准**）刻意钉字面量 `data/chroma`，跟着线上漂就没法逐格复算历史表。
    「线上那一份」由 `--chroma-v3` 的默认值承担（见下一条用例）。
    """

    from akr.mcp_server import server

    assert depth.DEFAULT_BM25 == server.DEFAULT_BM25_PATH
    assert depth.DEFAULT_MANIFEST == server.DEFAULT_MANIFEST_PATH
    assert depth.DEFAULT_CHROMA == Path("data/chroma")
    assert depth.DEFAULT_CHROMA != server.DEFAULT_CHROMA_PATH  # 线上已不在 v1 上
    assert depth.DEFAULT_AB.name == "retrieval-ab.json"


def test_the_online_column_and_the_fold_come_from_the_server_wiring() -> None:
    """`--chroma-v3` / `--unit-map` / `--fold` 的默认值必须与线上装配同源。

    这一列量的是**线上真正返回的顺序**：库换了（v3）、读法换了（同父块累加），脚本少跟一步，
    量出来的就是另一台机器上的另一件事。
    """

    from akr.mcp_server import server

    assert depth.DEFAULT_UNIT_MAP == server.DEFAULT_UNIT_MAP_PATH
    assert depth.DEFAULT_FOLD == server.DEFAULT_VECTOR_FOLD
    assert depth.DEFAULT_FOLD in depth.FOLD_MODES

    args = depth.build_parser().parse_args([])
    assert args.chroma_v3 == Path(server.DEFAULT_CHROMA_PATH)
    assert args.unit_map == Path(server.DEFAULT_UNIT_MAP_PATH)
    assert args.fold == server.DEFAULT_VECTOR_FOLD


def test_build_context_forwards_the_unit_map_and_the_fold(tmp_path: Path, monkeypatch) -> None:
    """装配出来的 `VectorStore` 必须真的拿到 `id_map` 与 `fold`——差一个就静默量错。"""

    seen: dict = {}
    bm25_seen: dict = {}

    class _FakeVectors:
        def __init__(self, path, embed_fn, *, id_map=None, fold="best"):
            seen.update(path=path, id_map=id_map, fold=fold)

    class _FakeBm25:
        @staticmethod
        def load(path, doc_metas=None, index_text_fn=None):
            bm25_seen.update(path=path, index_text_fn=index_text_fn)
            return object()

    monkeypatch.setattr(depth, "VectorStore", _FakeVectors)
    monkeypatch.setattr(depth, "build_embedder", lambda: object())
    monkeypatch.setattr(depth, "load_doc_meta", lambda path: {})
    monkeypatch.setattr(depth, "Bm25Index", _FakeBm25)

    depth.build_context(
        chroma_path=Path("data/chroma-v3"),
        bm25_path=Path("data/bm25.json"),
        manifest_path=Path("data/manifest.json"),
        id_map={"d#u1": "d#c1"},
        fold="rrf_sum",
    )

    assert seen == {
        "path": Path("data/chroma-v3"),
        "id_map": {"d#u1": "d#c1"},
        "fold": "rrf_sum",
    }
    # BM25 侧也带上导读行：它是装配层旋钮，少了这一步脚本与线上打的就是两份分。
    assert bm25_seen["path"] == Path("data/bm25.json")
    assert bm25_seen["index_text_fn"] is not None


def test_default_top_k_matches_the_live_worker_return_count() -> None:
    """池深是变量，返回条数不是：线上 worker 看的就是 8 条，别在脚本里悄悄放大。"""

    assert depth.DEFAULT_TOP_K == 8
    assert 8 in depth.DEFAULT_DEPTHS


def _retrieval_hit(chunk_id: str) -> RetrievalHit:
    """真 `RetrievalHit`：工具层会把命中转成 pydantic `Hit`，`SimpleNamespace` 过不了。"""
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        source="s.pdf",
        text=f"text of {chunk_id}",
        score=1.0,
        retriever="vector",
    )


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []


class _FakeVectors:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def query(self, text, *, top_k=8, doc_ids=None):
        self._recorder.calls.append(("vector", top_k))
        return [_retrieval_hit(f"d1#c{i:03d}") for i in range(top_k)]


class _FakeBm25:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def search(self, query, *, top_k=8, doc_ids=None):
        self._recorder.calls.append(("bm25", top_k))
        return []


class _RankedVectors:
    """gold 固定落在第 `rank` 名：返回条数不够长时它是 MISS，池子够深时它才现身。"""

    def __init__(self, recorder: _Recorder, gold: str, rank: int) -> None:
        self._recorder = recorder
        self._gold = gold
        self._rank = rank

    def query(self, text, *, top_k=8, doc_ids=None):
        self._recorder.calls.append(("vector", top_k))
        ids = [f"d9#c{i:03d}" for i in range(top_k)]
        if top_k >= self._rank:
            ids[self._rank - 1] = self._gold
        return [_retrieval_hit(chunk_id) for chunk_id in ids]


def test_measure_passes_the_pool_depth_as_a_tool_argument() -> None:
    """池深必须是工具层的入参，不是脚本自己截池——脚本与线上要跑同一段代码。

    显式传 `rerank="rrf"`：这个裸 `ToolContext` 没装配重排器，而工具层默认档已是
    `cross`（2026-09-26 开线上默认），不写就会在「便宜的检查」里当场报错。这条验的是
    池深透传，不该顺带验重排。
    """
    recorder = _Recorder()
    ctx = ToolContext(
        conn=None,
        vectors=_FakeVectors(recorder),
        bm25=_FakeBm25(recorder),
        graph=None,
        artifacts=None,
        doc_meta=None,
    )
    rows = [{"task_id": "t1", "gold_chunk_ids": ["d1#c000"], "question": "问题"}]

    measured = depth.measure(ctx, rows, (24,), rerank="rrf")

    assert recorder.calls == [("vector", 24), ("bm25", 24)]
    assert measured["24"]["top_k"] == 8
    assert measured["24"]["hit"] == 1
    assert measured["24"]["ranks"] == {"t1": 1}


def test_measure_vector_only_scores_the_depth_as_the_return_count() -> None:
    """纯向量那一列：`top_k` 就是返回条数，所以「第 20 名」只在深池里算命中。

    同一份命中，`top_k=8` 记 MISS、`top_k=50` 记名次 20——这正是「深召回好」与
    「进得了 top-8」两件事不能互相顶替的原因，也是这列要单独量的原因。
    """

    recorder = _Recorder()
    ctx = ToolContext(
        conn=None,
        vectors=_RankedVectors(recorder, gold="d1#c000", rank=20),
        bm25=_FakeBm25(recorder),
        graph=None,
        artifacts=None,
        doc_meta=None,
    )
    rows = [{"task_id": "t1", "gold_chunk_ids": ["d1#c000"], "question": "问题"}]

    measured = depth.measure_vector_only(ctx, rows, (8, 50))

    assert measured["8"]["top_k"] == 8
    assert measured["8"]["hit"] == 0 and measured["8"]["ranks"] == {"t1": None}
    assert measured["50"]["top_k"] == 50
    assert measured["50"]["hit"] == 1 and measured["50"]["ranks"] == {"t1": 20}
    # 「纯向量」这一列必须一次 BM25 都不碰，否则它量的是融合而不是嵌入。
    assert [call for call in recorder.calls if call[0] != "vector"] == []


def test_describe_vector_labels_the_return_count_as_the_depth() -> None:
    """标题写 `@depth`：这列返回条数就是池深，写成 `top-8` 会把两个口径混成一句话。"""

    line = depth.describe_vector(
        "v3", 50, {"top_k": 50, "total": 25, "hit": 21, "hit_rate": 0.84, "recall": 0.74, "mrr": 0.2}
    )
    assert "纯向量@50" in line and "21/25" in line and "top-8" not in line


def test_resolve_depths_prefers_the_single_pool_depth_over_the_curve() -> None:
    assert depth.resolve_depths(24, "8,16,50") == (24,)
    assert depth.resolve_depths(None, "50, 8") == (8, 50)


def test_resolve_depths_rejects_a_non_positive_single_pool_depth() -> None:
    """`--pool-depth 0` 等于「不放大池子」，与「只想量这一档」是两件事，必须报错。"""
    with pytest.raises(depth.AbError):
        depth.resolve_depths(0, "8")


def _scored_hit(chunk_id: str, score: float) -> RetrievalHit:
    """带真实余弦分的命中：Task 2 的 `_retrieval_hit` 分数全 1.0，验不出重排。"""
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        source="s.pdf",
        text=f"text of {chunk_id}",
        score=score,
        retriever="vector",
    )


class _ScoredVectors:
    """前两条的余弦分与名次反着来：只有真的跑了重排，它们才会换位。"""

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def query(self, text, *, top_k=8, doc_ids=None):
        self._recorder.calls.append(("vector", top_k))
        return [_scored_hit("d1#c000", 0.10), _scored_hit("d1#c001", 0.90)]


def _scored_ctx() -> tuple[ToolContext, _Recorder]:
    recorder = _Recorder()
    return (
        ToolContext(
            conn=None,
            vectors=_ScoredVectors(recorder),
            bm25=_FakeBm25(recorder),
            graph=None,
            artifacts=None,
            doc_meta=None,
        ),
        recorder,
    )


def test_measure_forwards_rerank_and_alpha_to_the_tool() -> None:
    """同一份候选集：`rrf` 记名次 2，`cosine`（alpha=1）把余弦高的那条翻到名次 1。

    基线那格显式写 `rerank="rrf"`：脚本默认档跟着线上走（现在是 `cross`），而这个
    上下文没装配重排器；「重排前长什么样」必须是显式说出来的口径。
    """
    ctx, _ = _scored_ctx()
    rows = [{"task_id": "t1", "gold_chunk_ids": ["d1#c001"], "question": "问题"}]

    plain = depth.measure(ctx, rows, (8,), rerank="rrf")
    reranked = depth.measure(ctx, rows, (8,), rerank="cosine", alpha=1.0)

    assert plain["8"]["ranks"] == {"t1": 2}
    assert reranked["8"]["ranks"] == {"t1": 1}


def test_measure_surfaces_an_unknown_rerank_mode_without_embedding() -> None:
    """模式名打错必须当场报错，且一次嵌入都不许花——这条锁住「先校验后检索」。"""
    ctx, recorder = _scored_ctx()
    rows = [{"task_id": "t1", "gold_chunk_ids": ["d1#c000"], "question": "问题"}]

    with pytest.raises(ValueError, match="rerank"):
        depth.measure(ctx, rows, (8,), rerank="magic")

    assert recorder.calls == []


def test_measure_cross_without_a_reranker_fails_before_spending_an_embedding() -> None:
    """`--rerank cross` 而装配层没建重排器时：当场报错，一次嵌入都不花。

    脚本自己会在 `main` 里按需建（`build_reranker`），但这条锁住「没建就报错」——
    静默退回 rrf 会印出一列看起来正常的数字，那比崩溃难查得多。
    """

    ctx, recorder = _scored_ctx()
    rows = [{"task_id": "t1", "gold_chunk_ids": ["d1#c000"], "question": "问题"}]

    with pytest.raises(ValueError, match="reranker"):
        depth.measure(ctx, rows, (8,), rerank="cross")

    assert recorder.calls == []


def test_build_context_hands_the_table_lead_rule_to_bm25(monkeypatch) -> None:
    """`Bm25Index.load` 拿到的必须是**导读行那条规则**，而且它不是恒等变换。"""

    from dataclasses import replace as dataclass_replace

    from akr.ingest.chunker import Chunk
    from akr.ingest.doc_meta import index_text

    captured: dict = {}

    class _FakeVectors:
        def __init__(self, path, embed_fn, *, id_map=None, fold="best"):
            pass

    class _FakeBm25:
        @staticmethod
        def load(path, doc_metas=None, index_text_fn=None):
            captured["index_text_fn"] = index_text_fn
            return object()

    monkeypatch.setattr(depth, "VectorStore", _FakeVectors)
    monkeypatch.setattr(depth, "build_embedder", lambda: object())
    monkeypatch.setattr(depth, "load_doc_meta", lambda path: {})
    monkeypatch.setattr(depth, "Bm25Index", _FakeBm25)

    depth.build_context(
        chroma_path=Path("data/chroma"),
        bm25_path=Path("data/bm25.json"),
        manifest_path=Path("data/manifest.json"),
    )

    render = captured["index_text_fn"]
    table = Chunk(
        chunk_id="d1#c0000",
        doc_id="d1",
        source="s.pdf",
        ordinal=0,
        text="2024年 2023年 本年比上年增减\n营业收入（元） 100 90 11%\n净利润（元） 20 10 100%",
        content_hash="h",
        section_path="",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
    )
    assert render(table).startswith("2024年 2023年 本年比上年增减")
    assert render(table) != index_text(table, None)
    prose = dataclass_replace(table, text="营业收入为 100 亿元")
    assert render(prose) == index_text(prose, None)


def test_the_cross_rerank_ledger_path_is_an_optional_cli_knob() -> None:
    """`--rerank` 的默认值跟着线上走（`cross`，2026-09-26 开线上默认）；账本仍是可选档。

    代价要看清：默认档已经会花重排调用，而 `--rerank-usage` 默认 None = 这一跑**不落账本**。
    所以「这一跑花了多少 token」这件事在产物里是显式的 `rerank_usage: null`；要记账必须
    显式传 `--rerank-usage`。
    """

    args = depth.build_parser().parse_args([])
    assert args.rerank == depth.RERANK_DEFAULT == "cross"
    assert args.rerank_usage is None
    assert "cross" in depth.RERANK_MODES

    chosen = depth.build_parser().parse_args(
        ["--rerank", "cross", "--rerank-usage", "tmp/rerank-usage.json"]
    )
    assert chosen.rerank == "cross"
    assert chosen.rerank_usage == Path("tmp/rerank-usage.json")


def test_resolve_stores_only_adds_the_comparison_libraries_that_were_given() -> None:
    """v1 永远量；v2/v3 只有显式给了路径才加，免得缺目录直接崩。"""
    assert depth.resolve_stores(Path("a"), None, None) == [("v1", Path("a"))]
    assert depth.resolve_stores(Path("a"), Path("b"), Path("c")) == [
        ("v1", Path("a")),
        ("v2", Path("b")),
        ("v3", Path("c")),
    ]


def test_resolve_id_maps_only_folds_the_unit_level_library(tmp_path: Path) -> None:
    """v2 是块级索引（id 就是父块 id，不需要折叠）；v3 是 unit 级，必须折叠。"""

    unit_map = tmp_path / "unit-map.json"
    unit_map.write_text('{"d#u1": "d#c1"}', encoding="utf-8")

    # 没量 v3 时不读映射（映射文件不存在也不该报错）。
    assert depth.resolve_id_maps([("v1", Path("a")), ("v2", Path("b"))], tmp_path / "缺.json") == {}
    assert depth.resolve_id_maps([("v1", Path("a")), ("v3", Path("c"))], unit_map) == {
        "v3": {"d#u1": "d#c1"}
    }


def test_resolve_id_maps_refuses_to_measure_v3_without_the_mapping(tmp_path: Path) -> None:
    """缺映射还照跑，v3 那一列会以「召回率归零」的形状骗过所有检查——必须当场报错。"""

    with pytest.raises(depth.AbError, match="unit → 父块"):
        depth.resolve_id_maps([("v1", Path("a")), ("v3", Path("c"))], tmp_path / "缺.json")
