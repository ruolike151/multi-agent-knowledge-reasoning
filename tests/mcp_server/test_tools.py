import inspect
from dataclasses import dataclass

import pytest

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import DocMeta
from akr.mcp_server import tools as tools_module
from akr.mcp_server.schemas import TOOL_SCHEMAS
from akr.mcp_server.tools import (
    TOOL_FUNCTIONS,
    ToolContext,
    dispatch,
    dispatch_safe,
    fetch_chunk,
    list_documents,
    resolve_entity,
    search_documents,
    walk_graph,
)
from akr.storage import sqlite_store
from akr.storage.artifact_store import ArtifactNotFound
from akr.storage.graph_store import EntityCandidate, EntityRef, GraphPath, RelationRef
from akr.storage.hits import RetrievalHit


def hit(chunk_id, text, score, retriever="vector", doc_id="d1", source="s.pdf"):
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=source,
        text=text,
        score=score,
        retriever=retriever,
    )


class FakeVectors:
    def __init__(self, hits, recorder):
        self._hits = hits
        self._recorder = recorder

    def query(self, text, *, top_k=8, doc_ids=None):
        self._recorder.append(("vector", text, top_k, tuple(doc_ids or ())))
        return list(self._hits)


class FakeBm25:
    def __init__(self, hits, recorder):
        self._hits = hits
        self._recorder = recorder

    def search(self, query, *, top_k=8, doc_ids=None):
        self._recorder.append(("bm25", query, top_k, tuple(doc_ids or ())))
        return list(self._hits)


class ScriptedVectors:
    """按调用序号给不同的命中批次：第 1 次是第一段、第 2 次是第二段（文档内重搜）。

    真假向量库对同一个问句加不同过滤条件会返回不同排序，这里用脚本把它显式化，
    才能验「第二段确实改变了名次」，而不是只验参数被传下去了。
    """

    def __init__(self, first, second, recorder):
        self._batches = [list(first), list(second)]
        self._recorder = recorder

    def query(self, text, *, top_k=8, doc_ids=None):
        self._recorder.append(("vector", text, top_k, tuple(doc_ids or ())))
        return self._batches.pop(0) if self._batches else []


class FakeGraph:
    def __init__(self, candidates=None, paths=None):
        self._candidates = candidates or []
        self._paths = paths or []
        self.walk_calls = []

    def resolve(self, mention, *, context="", limit=5):
        self.resolve_call = (mention, context, limit)
        return list(self._candidates)[:limit]

    def walk(self, entity_id, *, hops=2, relation_types=None, limit=20):
        self.walk_calls.append((entity_id, hops, relation_types))
        return list(self._paths)


class FakeArtifacts:
    def __init__(self, artifact=None):
        self._artifact = artifact
        self.seen = []

    def get(self, doc_id, chunk_id):
        self.seen.append((doc_id, chunk_id))
        if self._artifact is None:
            raise ArtifactNotFound(chunk_id)
        return self._artifact


@dataclass(frozen=True)
class FakeArtifactRow:
    chunk_id: str
    doc_id: str
    text: str
    char_range: tuple
    page: int


def flat_reranker(query, documents):
    """测试默认重排器：所有候选同分。

    为什么 `make_ctx` 默认装配它：工具层的**线上默认已经是 `rerank="cross"` + 锁 4**
    （2026-09-26 拍板），而 `cross` 缺 `ctx.reranker` 时会当场报错。大多数用例关心的是融合 /
    池深 / 扩窗 / 两段式的语义，不该被「没装配重排器」挡住；装配一个**不改顺序**的实现，
    既让它们真的把默认路径跑了一遍，又让既有断言（名次、命中集合）原样成立——
    `score_rerank` 的排序键是 `(-score, index)`，同分时稳定排序保留候选顺序。

    要「装配层没给重排器」那条口径的用例显式传 `reranker=None`。
    """
    return [0.0] * len(documents)


def make_ctx(
    *,
    vectors=None,
    bm25=None,
    graph=None,
    artifacts=None,
    conn=None,
    doc_meta=None,
    reranker=flat_reranker,
):
    recorder = []
    return ToolContext(
        conn=conn,
        vectors=vectors or FakeVectors([], recorder),
        bm25=bm25 or FakeBm25([], recorder),
        graph=graph or FakeGraph(),
        artifacts=artifacts or FakeArtifacts(),
        doc_meta=doc_meta,
        reranker=reranker,
    ), recorder


def chunk(chunk_id, *, doc_id="d1", source="s.pdf", ordinal=0):
    """最小 Chunk：只用于让 SQLite 的 chunk_count 有真实来源。"""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=source,
        ordinal=ordinal,
        text=f"第 {ordinal} 段正文",
        content_hash=f"hash{ordinal}",
        section_path="",
        page_start=1,
        page_end=1,
        char_start=ordinal * 10,
        char_end=ordinal * 10 + 10,
    )


def test_search_documents_fuses_vector_and_bm25_with_rrf():
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors([hit("d1#c1", "向量命中", 0.9)], recorder),
        bm25=FakeBm25([hit("d1#c2", "关键词命中", 12.0, retriever="bm25")], recorder),
    )
    result = search_documents(ctx, "甲公司 供应商", top_k=3)
    assert result.fusion == "rrf"
    assert {h.chunk_id for h in result.hits} == {"d1#c1", "d1#c2"}
    assert result.retriever_stats["vector"] == 1
    assert result.retriever_stats["bm25"] == 1


def test_search_documents_marks_hits_found_by_both_retrievers():
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors([hit("d1#c1", "同一段", 0.9)], recorder),
        bm25=FakeBm25([hit("d1#c1", "同一段", 9.0, retriever="bm25")], recorder),
    )
    result = search_documents(ctx, "同一段")
    assert [h.retriever for h in result.hits] == ["both"]


def test_search_documents_passes_doc_filters_as_doc_ids():
    """`doc_filters` 转成 `doc_ids` 传给两路。池深与这条无关，显式钉 0（= 池深等于 top_k）：
    断言的 8 是「历史上两路各查 8 条」，不该跟着线上默认池深（现在 50）漂。"""
    ctx, recorder = make_ctx()
    search_documents(ctx, "问题", doc_filters={"doc_ids": ["d2"]}, pool_depth=0)
    assert recorder == [("vector", "问题", 8, ("d2",)), ("bm25", "问题", 8, ("d2",))]


def test_search_documents_rejects_unknown_filter_keys():
    ctx, _ = make_ctx()
    with pytest.raises(ValueError):
        search_documents(ctx, "问题", doc_filters={"year": 2024})


def test_two_stage_is_off_by_default_and_adds_no_calls():
    """不开两段式就一次多余调用都不许加：不多一次嵌入、不多一次 BM25。

    `pool_depth=0`（历史口径）是显式钉的——「默认池深是几」由
    `test_pool_depth_default_deepens_both_retrievers_and_keeps_top_k` 单独量，
    免得这条用例的 8/8 跟着池深默认值漂。
    """
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors([hit("d1#c1", "向量命中", 0.9)], recorder),
        bm25=FakeBm25([hit("d2#c9", "关键词命中", 5.0, retriever="bm25")], recorder),
    )

    result = search_documents(ctx, "问题", pool_depth=0)

    assert recorder == [("vector", "问题", 8, ()), ("bm25", "问题", 8, ())]
    assert "stage2_docs" not in result.retriever_stats


def test_two_stage_researches_top_documents_and_reranks_them_first():
    """第二段：用第一段的前 N 个**文档**当过滤条件重搜，结果排最前、其余按序补上。"""
    recorder = []
    first = [
        hit("d2#x", "第二名", 0.5, doc_id="d2"),
        hit("d3#y", "第三名", 0.4, doc_id="d3"),
        hit("d1#z", "第四名", 0.3, doc_id="d1"),
        hit("d9#w", "第五名", 0.2, doc_id="d9"),
    ]
    second = [
        hit("d3#y", "第三名", 0.9, doc_id="d3"),
        hit("d2#x", "第二名", 0.8, doc_id="d2"),
    ]
    ctx, _ = make_ctx(
        vectors=ScriptedVectors(first, second, recorder),
        bm25=FakeBm25([], recorder),
    )

    result = search_documents(ctx, "问题", top_k=4, two_stage_docs=2)

    assert recorder[2:] == [
        ("vector", "问题", 4, ("d2", "d3")),
        ("bm25", "问题", 4, ("d2", "d3")),
    ]
    assert [h.chunk_id for h in result.hits] == ["d3#y", "d2#x", "d1#z", "d9#w"]
    assert result.retriever_stats["stage2_docs"] == 2
    assert result.retriever_stats["stage2"] == 2


def test_two_stage_picks_documents_from_the_deep_pool_not_the_returned_top_k():
    """定文档的池子必须是深池，不是返回给模型的 top_k。

    返回给模型的 top-8 里常常只有 2~3 个不同文档（同一份文档占了好几个名次），
    拿它定文档会把真正的目标文档漏掉——只读 A/B 实测：浅池定文档 6/25、深池 10/25。

    `pool_depth=0` 是**显式**钉的，不是笔误：只有 0 才会走到 `TWO_STAGE_PICK_DEPTH`（50）。
    线上默认池深现在是 50，它同时把这一段也定在 50——见下面那条
    `test_two_stage_first_stage_depth_follows_the_pool_depth_knob`。
    """
    recorder = []
    deep = [
        hit("d1#a", "一", 0.9, doc_id="d1"),
        hit("d1#b", "二", 0.8, doc_id="d1"),
        hit("d2#a", "三", 0.7, doc_id="d2"),
        hit("d7#a", "四", 0.6, doc_id="d7"),
    ]
    second = [hit("d7#a", "四", 0.9, doc_id="d7")]
    ctx, _ = make_ctx(
        vectors=ScriptedVectors(deep, second, recorder),
        bm25=FakeBm25([], recorder),
    )

    result = search_documents(ctx, "问题", top_k=2, two_stage_docs=3, pool_depth=0)

    assert recorder[0] == ("vector", "问题", 50, ())
    assert recorder[2:] == [
        ("vector", "问题", 2, ("d1", "d2", "d7")),
        ("bm25", "问题", 2, ("d1", "d2", "d7")),
    ]
    assert result.hits[0].chunk_id == "d7#a"
    assert result.retriever_stats["stage2_docs"] == 3


def test_two_stage_first_stage_depth_follows_the_pool_depth_knob():
    """池深是**唯一**的深度旋钮：它一改，两段式第一段「两路各查多深」跟着改。

    这是 Task 2 定下的口径（池深与「取候选的深度」是同一个数），不是本轮接线引入的；
    接线把默认值从 0 改成 50 之后，默认池深恰好**等于** `TWO_STAGE_PICK_DEPTH`——定文档的
    深池（实测口径浅池 6/25 vs 深池 10/25）不会因为走默认路径而缩水。两段式本来就从未在
    线上触发过（问句抽码收窄恒命中，`doc_ids` 恒非空，见 `search_documents` 顶部注释），
    所以这条只锁行为形状：**池深一旦被调用方调浅，定文档的池子就必须跟着变浅**，不能一边
    说「池深 12」一边偷偷用 50 定文档。真要上线两段式，把池深留在默认 50 或显式传
    `pool_depth=0` 都行，两个值现在给出同一个深度。
    """
    recorder = []
    deep = [hit(f"d{i}#a", "段", 1.0 - i / 10, doc_id=f"d{i}") for i in range(1, 13)]
    ctx, _ = make_ctx(
        vectors=ScriptedVectors(deep, [], recorder),
        bm25=FakeBm25([], recorder),
    )

    search_documents(ctx, "问题", top_k=2, two_stage_docs=3)

    assert recorder[0] == ("vector", "问题", 50, ())  # 默认池深 50，与 TWO_STAGE_PICK_DEPTH 同值


def test_two_stage_rejects_negative_span():
    ctx, _ = make_ctx()
    with pytest.raises(ValueError, match="two_stage_docs"):
        search_documents(ctx, "问题", two_stage_docs=-1)


def test_two_stage_is_skipped_when_the_caller_already_filtered_documents():
    """调用方自己给了 `doc_filters` 时不再叠加第二段：过滤意图是它的，不该被覆盖。

    池深钉 0 的理由同 `test_two_stage_is_off_by_default_and_adds_no_calls`。
    """
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors([hit("d1#c1", "向量命中", 0.9)], recorder),
        bm25=FakeBm25([], recorder),
    )

    result = search_documents(
        ctx, "问题", doc_filters={"doc_ids": ["d1"]}, two_stage_docs=3, pool_depth=0
    )

    assert recorder == [("vector", "问题", 8, ("d1",)), ("bm25", "问题", 8, ("d1",))]
    assert "stage2_docs" not in result.retriever_stats


def fake_metas():
    """两家同业公司的身份表：代码唯一、公司名互不为子串，才能验「按代码收窄」。"""

    return {
        "d1": DocMeta(doc_id="d1", company="甲公司", code="002129", period="2024年年度"),
        "d2": DocMeta(doc_id="d2", company="乙公司", code="300274", period="2024年年度"),
        "d3": DocMeta(doc_id="d3", company="甲公司", code="002129", period="2025年年度"),
    }


def test_search_documents_narrows_candidates_when_the_question_names_a_code():
    """问句里有证券代码时，两路检索都只在该公司的文档里跑。

    这条量的是**收窄**（`doc_ids`），不是池深：池深钉 0 让两个断言的数各归各的意思。
    """
    ctx, recorder = make_ctx(doc_meta=fake_metas())

    search_documents(ctx, "甲公司（002129）2024 年的营业收入是多少？", pool_depth=0)

    assert [call[0] for call in recorder] == ["vector", "bm25"]
    assert [call[2] for call in recorder] == [8, 8]
    assert [call[3] for call in recorder] == [("d1",), ("d1",)]


def test_search_documents_keeps_all_years_of_the_named_company_when_year_is_absent():
    """问句只给公司名时收窄到该公司全部分文档，而不是只留一份。"""
    ctx, recorder = make_ctx(doc_meta=fake_metas())

    search_documents(ctx, "甲公司的研发投入")

    assert recorder[0][3] == ("d1", "d3")


def test_search_documents_does_not_narrow_without_a_document_identity_table():
    """没有身份表就退化成全库检索——历史行为不能被这条规则悄悄改掉。"""
    ctx, recorder = make_ctx()

    search_documents(ctx, "甲公司（002129）2024 年的营业收入是多少？")

    assert [call[3] for call in recorder] == [(), ()]


def test_search_documents_does_not_narrow_when_the_question_has_no_code_or_company():
    """抽不到线索就不收窄：宁可全库检索，也不能猜错文档把 gold 过滤掉。"""
    ctx, recorder = make_ctx(doc_meta=fake_metas())

    search_documents(ctx, "2024 年硅片出货量是多少？")

    assert [call[3] for call in recorder] == [(), ()]


def test_search_documents_respects_caller_filters_instead_of_narrowing():
    """调用方给的 `doc_filters` 优先：它的过滤意图不能被问句抽出的线索覆盖。"""
    ctx, recorder = make_ctx(doc_meta=fake_metas())

    search_documents(
        ctx, "甲公司（002129）2024 年的营业收入是多少？", doc_filters={"doc_ids": ["d2"]}
    )

    assert [call[3] for call in recorder] == [("d2",), ("d2",)]


def test_narrowing_supersedes_the_two_stage_deep_pool():
    """收窄后候选只剩 1~3 份文档，没有「再定文档」的余地：不该再多买一次深池查询。"""
    ctx, recorder = make_ctx(doc_meta=fake_metas())

    result = search_documents(
        ctx, "甲公司（002129）2024 年的营业收入是多少？", two_stage_docs=3, pool_depth=0
    )

    assert [call[2] for call in recorder] == [8, 8]  # 不是深池的 50
    assert "stage2_docs" not in result.retriever_stats


def test_walk_graph_resolves_mention_then_walks():
    paths = [
        GraphPath(
            nodes=(
                EntityRef("company:甲公司", "甲公司", "company"),
                EntityRef("company:乙公司", "乙公司", "company"),
            ),
            relations=(RelationRef("SUPPLIES", "company:甲公司", "company:乙公司"),),
        )
    ]
    graph = FakeGraph([EntityCandidate("company:甲公司", "甲公司", "company", (), 1.0)], paths)
    ctx, _ = make_ctx(graph=graph)
    result = walk_graph(ctx, "甲公司", hops=2)
    assert graph.walk_calls == [("company:甲公司", 2, None)]
    assert result.paths[0].text == "甲公司 -[SUPPLIES]-> 乙公司"
    assert result.paths[0].relations[0].target_id == "company:乙公司"


def test_walk_graph_returns_empty_when_entity_unresolved():
    ctx, _ = make_ctx(graph=FakeGraph([]))
    assert walk_graph(ctx, "不存在的公司").paths == []


def test_resolve_entity_forwards_mention_and_context():
    graph = FakeGraph([EntityCandidate("company:甲公司", "甲公司", "company", ("甲",), 0.8)])
    ctx, _ = make_ctx(graph=graph)
    result = resolve_entity(ctx, "甲", context="供应商章节")
    assert graph.resolve_call == ("甲", "供应商章节", 5)
    assert result.candidates[0].entity_id == "company:甲公司"
    assert result.candidates[0].aliases == ["甲"]


def test_fetch_chunk_derives_doc_id_and_returns_provenance():
    artifact = FakeArtifactRow(
        chunk_id="d1#cabc", doc_id="d1", text="原文片段", char_range=(10, 20), page=3
    )
    artifacts = FakeArtifacts(artifact)
    conn = sqlite_store.connect(":memory:")
    sqlite_store.init_schema(conn)
    sqlite_store.upsert_document(
        conn,
        doc_id="d1",
        source="s.pdf",
        doc_type="annual_report",
        page_count=1,
        updated_at="2026-09-23T00:00:00+00:00",
    )
    ctx, _ = make_ctx(artifacts=artifacts, conn=conn)
    result = fetch_chunk(ctx, "d1#cabc")
    assert artifacts.seen == [("d1", "d1#cabc")]
    assert (result.doc_id, result.source, result.char_range) == ("d1", "s.pdf", [10, 20])


def test_list_documents_maps_sqlite_rows():
    conn = sqlite_store.connect(":memory:")
    sqlite_store.init_schema(conn)
    sqlite_store.upsert_document(
        conn,
        doc_id="d1",
        source="甲公司2024年年度报告.pdf",
        doc_type="annual_report",
        page_count=120,
        updated_at="2026-09-23T00:00:00+00:00",
    )
    sqlite_store.replace_chunks(
        conn,
        "d1",
        [
            chunk("d1#c1", doc_id="d1", source="甲公司2024年年度报告.pdf", ordinal=0),
            chunk("d1#c2", doc_id="d1", source="甲公司2024年年度报告.pdf", ordinal=1),
        ],
    )
    ctx, _ = make_ctx(conn=conn)
    result = list_documents(ctx)
    assert [doc.doc_id for doc in result.documents] == ["d1"]
    assert result.documents[0].chunk_count == 2
    assert result.documents[0].source == "甲公司2024年年度报告.pdf"


def test_dispatch_returns_json_ready_dict():
    conn = sqlite_store.connect(":memory:")
    sqlite_store.init_schema(conn)
    ctx, _ = make_ctx(conn=conn)
    payload = dispatch(ctx, "list_documents", {})
    assert payload == {"documents": []}
    assert isinstance(payload, dict)


def test_dispatch_rejects_unknown_tool_name():
    ctx, _ = make_ctx()
    with pytest.raises(ValueError):
        dispatch(ctx, "drop_all_tables", {})


def test_dispatch_safe_turns_errors_into_payload():
    ctx, _ = make_ctx()
    payload = dispatch_safe(ctx, "fetch_chunk", {"chunk_id": "d9#cmissing"})
    assert "error" in payload and "d9#cmissing" in payload["error"]


def test_tool_schemas_cover_every_registered_function():
    assert set(TOOL_FUNCTIONS) == {
        "search_documents",
        "walk_graph",
        "resolve_entity",
        "fetch_chunk",
        "list_documents",
    }
    from akr.mcp_server.schemas import TOOL_SCHEMAS

    assert {schema["name"] for schema in TOOL_SCHEMAS} == set(TOOL_FUNCTIONS)
    for schema in TOOL_SCHEMAS:
        assert schema["inputSchema"]["type"] == "object"


def test_tools_module_does_not_import_model_client():
    source = inspect.getsource(tools_module)
    assert "akr.model" not in source  # 硬约束：工具层不得触碰模型层


def pool_hits(prefix, count):
    """造 `count` 条同文档、分数递减的命中：只有够深，池深才看得出来。"""
    return [hit(f"d1#{prefix}{i:02d}", f"第 {i} 段", 1.0 - i / 100) for i in range(count)]


def test_pool_depth_default_constant_is_the_wired_online_value():
    """2026-09-26 起默认 50（线上口径）；0 仍在，但它是「不放大池子」的哨兵而不是默认值。

    这个常量同时是工具层形参默认值与 MCP schema 的默认值（见下一条 schema 用例）。
    改它之前先看 `POOL_DEPTH_DEFAULT` 上面那段实测注释：深池不是越深越好——50 这一档是
    **配着 `RRF_K_DEFAULT=20`** 才成立的（只动池深、k 留在 60 会把 MRR 从 0.6490 拉到 0.6247）。
    """
    assert tools_module.POOL_DEPTH_DEFAULT == 50


def test_pool_depth_default_deepens_both_retrievers_and_keeps_top_k():
    """线上默认路径：两路各查 50 条去融合，返回给模型的仍然是 `top_k` 条。"""
    ctx, recorder = make_ctx()

    result = search_documents(ctx, "问题")

    assert recorder == [("vector", "问题", 50, ()), ("bm25", "问题", 50, ())]
    assert result.retriever_stats == {"vector": 0, "bm25": 0, "fused": 0}


def test_pool_depth_zero_is_still_the_historical_path():
    """显式传 0 才等于改动前的行为（池深 = top_k）。

    第三十四~三十八节那几张历史表都是这个口径量的，它们靠这条路径复算，而不是靠默认值。
    """
    ctx, recorder = make_ctx()

    search_documents(ctx, "问题", pool_depth=0)

    assert recorder == [("vector", "问题", 8, ()), ("bm25", "问题", 8, ())]


def test_pool_depth_deepens_both_retrievers_and_still_returns_top_k():
    """池深 = 两条检索路各取多少候选去融合；返回给模型的条数不变。

    后半句是硬约束：`top_k` 就是进 prompt 的片段数，池深再深也不许把它撑大。
    """
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors(pool_hits("v", 24), recorder),
        bm25=FakeBm25(pool_hits("b", 24), recorder),
    )

    result = search_documents(ctx, "问题", pool_depth=24)

    assert [(call[0], call[2]) for call in recorder] == [("vector", 24), ("bm25", 24)]
    assert result.retriever_stats["vector"] == 24
    assert result.retriever_stats["bm25"] == 24
    assert len(result.hits) == 8


def test_pool_depth_sets_the_two_stage_first_stage_depth_too():
    """同时开两段式与池深时，第一段的两路查询用池深（不再固定 50）。

    第一段的命中既拿去定文档、又是补在第二段后面的「第一段额外命中」，
    所以它的池子必须跟着池深走，否则两个旋钮互相打架。
    """
    recorder = []
    ctx, _ = make_ctx(
        vectors=ScriptedVectors(pool_hits("v", 24), [], recorder),
        bm25=FakeBm25([], recorder),
    )

    search_documents(ctx, "问题", two_stage_docs=2, pool_depth=24)

    assert recorder[0] == ("vector", "问题", 24, ())
    assert recorder[1] == ("bm25", "问题", 24, ())


def test_pool_depth_and_narrowing_apply_together():
    """收窄只改 `doc_ids`、池深只改条数：两个旋钮互不覆盖。"""
    ctx, recorder = make_ctx(doc_meta=fake_metas())

    search_documents(ctx, "甲公司（002129）2024 年的营业收入是多少？", pool_depth=16)

    assert [call[2] for call in recorder] == [16, 16]
    assert [call[3] for call in recorder] == [("d1",), ("d1",)]


def test_pool_depth_rejects_a_negative_pool():
    """负数不是「按默认走」：它是调用方的笔误，宁可报错也不要静默改口径。"""
    ctx, _ = make_ctx()
    with pytest.raises(ValueError, match="pool_depth"):
        search_documents(ctx, "问题", pool_depth=-1)


def test_search_documents_schema_exposes_the_pool_depth_knob():
    """schema 是给 MCP 客户端（模型）看的：旋钮不进 properties，调用方就发现不了它。

    `schemas.py` 不能 import `tools.py`（`tools.py` 已经 import 了 schemas，会成环），
    所以默认值在两边各写一份字面量，由这条断言把它们钉在一起。
    """
    entry = next(item for item in TOOL_SCHEMAS if item["name"] == "search_documents")
    prop = entry["inputSchema"]["properties"]["pool_depth"]
    assert prop["type"] == "integer"
    assert prop["default"] == tools_module.POOL_DEPTH_DEFAULT


def test_search_documents_schema_exposes_the_rrf_k_and_rerank_lock_knobs():
    """同一份等式约束 `rrf_k` / `rerank_lock`：两边各写一份字面量，靠断言钉死。"""
    entry = next(item for item in TOOL_SCHEMAS if item["name"] == "search_documents")
    props = entry["inputSchema"]["properties"]

    assert props["rrf_k"]["type"] == "integer"
    assert props["rrf_k"]["default"] == tools_module.RRF_K_DEFAULT
    assert props["rrf_k"]["minimum"] == 1

    assert props["rerank_lock"]["type"] == "integer"
    assert props["rerank_lock"]["default"] == tools_module.RERANK_LOCK_DEFAULT
    assert props["rerank_lock"]["minimum"] == 0


def test_rrf_k_default_is_decoupled_from_the_vector_fold_constant():
    """工具层的 `rrf_k` 刻意**不**等于 `akr.storage.fusion.DEFAULT_K`（=60）。

    两处公式同形（`1/(k + 名次)`），优化目标却相反：融合调 k 是为了「靠后的同名次共识别把
    前面的强命中顶掉」（深池必须配小 k），而 `fusion.DEFAULT_K` 同时被向量侧的 `rrf_sum`
    折叠（`VectorStore._aggregate`，线上口径 `server.DEFAULT_VECTOR_FOLD`）当作同一父块内
    多条 unit 的累加权重用。共用一个常数 = 「调融合」顺带改掉「检索候选集合」，量到的数字
    立刻与线上脱钩。这条断言就是那次拆分的守卫：谁把两者重新合并，这里会先红。
    """
    from akr.storage import fusion

    assert tools_module.RRF_K_DEFAULT == 20
    assert fusion.DEFAULT_K == 60
    assert tools_module.RRF_K_DEFAULT != fusion.DEFAULT_K


def test_rrf_k_changes_the_fused_order_for_the_same_candidates():
    """同一批候选，`rrf_k` 从 60 收到 20 会**改顺序**——所以它必须落盘才复算得出来。

    构造刻意选在两种 k 的交叉点上：`d1#x` 只在向量路出现且排第 1（得 `1/(k+1)`），
    `d1#s00..s29` 在两路都出现、名次一路比一路深（得 `1/(k+r-1) + 1/(k+r)`）。
    两条路的项数差一倍，k 越大「双路出现」越占便宜：

    * k=60 → 30 条共享块全部压过 `d1#x`，它是第 31 名（垫底）；
    * k=20 → 只有 21 条压过它，它升到第 22 名。

    这不是「分数尺度变了」：名次一个都没动，动的是 `1/(k+rank)` 对新旧名次的敏感度。
    """
    recorder = []
    vector = [hit("d1#x", "独有强命中", 0.9)] + [
        hit(f"d1#s{i:02d}", f"共享 {i}", 0.5) for i in range(30)
    ]
    bm25 = [
        hit(f"d1#s{i:02d}", f"共享 {i}", 10.0 - i, retriever="bm25") for i in range(30)
    ]
    ctx, _ = make_ctx(vectors=FakeVectors(vector, recorder), bm25=FakeBm25(bm25, recorder))

    def rank_of_x(rrf_k):
        result = search_documents(ctx, "问题", pool_depth=31, top_k=31, rrf_k=rrf_k)
        return [item.chunk_id for item in result.hits].index("d1#x") + 1

    assert rank_of_x(rrf_k=20) == 22
    assert rank_of_x(rrf_k=60) == 31


def test_rrf_k_and_rerank_lock_are_validated_before_any_retriever_runs():
    """参数打错时不许先花掉一次嵌入：三个检查都要排在检索之前。"""
    ctx, recorder = make_ctx()

    with pytest.raises(ValueError, match="rrf_k"):
        search_documents(ctx, "问题", rrf_k=0)
    with pytest.raises(ValueError, match="rerank_lock"):
        search_documents(ctx, "问题", rerank_lock=9)  # top_k 默认 8
    with pytest.raises(ValueError, match="rerank_lock"):
        search_documents(ctx, "问题", rerank="rrf", rerank_lock=4)  # 没开 cross：不静默忽略

    assert recorder == []


def test_rerank_default_is_the_online_cross_with_a_locked_head():
    """默认档 = `cross` + 锁 4（2026-09-26 拍板开线上默认），模式集合是白名单。

    这条把「线上默认」钉成常数，而不是靠注释描述：改默认值必须同时改这一行与
    内部偏差日志第四十四节的接线记录。
    """
    assert tools_module.RERANK_DEFAULT == "cross"
    assert tools_module.RERANK_LOCK_DEFAULT == 4
    assert tools_module.RERANK_MODES == frozenset({"rrf", "cosine", "cross"})
    assert tools_module.RERANK_DEFAULT in tools_module.RERANK_MODES


def test_default_path_really_calls_the_reranker_on_the_pool_tail():
    """默认档必须**真的**走重排器：只调一次、只拿到尾池（前 4 条由 RRF 独占）。

    这条是「开线上默认」这个动作的效果断言——如果哪天 `RERANK_DEFAULT` 被改回 `rrf`，
    或者锁头把尾池吃光，这里会立刻红。
    """
    recorder: list = []
    calls: list[tuple[str, list[str]]] = []

    def fake_reranker(query, documents):
        calls.append((query, list(documents)))
        return [1.0] * len(documents)

    pool = [hit(f"d1#c{i:02d}", f"第 {i} 段", 0.9) for i in range(12)]
    ctx, _ = make_ctx(
        vectors=FakeVectors(pool, recorder), bm25=FakeBm25([], recorder), reranker=fake_reranker
    )

    # 基线顺序用**显式** rrf 取：默认档现在自己就会调一次重排器，混进来会把 calls 数错。
    plain = [
        item.chunk_id
        for item in search_documents(ctx, "问题", pool_depth=12, top_k=12, rerank="rrf").hits
    ]
    result = search_documents(ctx, "问题", pool_depth=12, top_k=8)

    assert len(calls) == 1  # 默认发一次重排，不是零次
    assert calls[0][1] == [f"第 {i} 段" for i in range(4, 12)]  # 只有尾池 8 条
    assert [item.chunk_id for item in result.hits][:4] == plain[:4]  # 头部由 RRF 独占


def test_default_rerank_lock_saturates_at_top_k_and_then_never_calls_the_backend():
    """`top_k` 比默认锁头条数还小时：锁到 `top_k`（名额全被 RRF 占满），一次都不调后端。

    这是线上 `graph_walker` 那条路（`EVIDENCE_TOP_K = 3`）。它**不许**抛
    「lock > top_k」——那是默认值自己越界，不是调用方配错。
    """
    calls: list[int] = []

    def fake_reranker(query, documents):
        calls.append(len(documents))
        return [1.0] * len(documents)

    pool = [hit(f"d1#c{i:02d}", f"第 {i} 段", 0.9) for i in range(12)]
    ctx, _ = make_ctx(vectors=FakeVectors(pool, []), bm25=FakeBm25([], []), reranker=fake_reranker)

    shallow = search_documents(ctx, "问题", pool_depth=12, top_k=3)
    plain = search_documents(ctx, "问题", pool_depth=12, top_k=3, rerank="rrf")

    assert calls == []
    assert [item.chunk_id for item in shallow.hits] == [
        item.chunk_id for item in plain.hits
    ]


def test_explicit_rrf_still_runs_without_a_reranker():
    """离线口径：显式 `rerank="rrf"` 时缺重排器**不是**错误（一次网络请求都不发）。

    与下一条成对：默认档缺重排器必须报错，「我不想用重排」必须是调用方说出来的话。
    """
    ctx, recorder = make_ctx(reranker=None)

    result = search_documents(ctx, "问题", rerank="rrf")

    assert result.hits == []
    assert [call[0] for call in recorder] == ["vector", "bm25"]


def test_rerank_default_keeps_the_rrf_order():
    """默认档（`cross` + 锁 4）配上**不改顺序**的重排器时，逐字等于 RRF 顺序。

    候选只有 2 条、连头部都填不满，尾池为空 → 重排器一次都不调，顺序与条数都由 RRF 定。
    """
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors(
            [hit("d1#c1", "名次一", 0.2), hit("d1#c2", "名次二", 0.9)], recorder
        ),
        bm25=FakeBm25([], recorder),
    )

    result = search_documents(ctx, "问题")

    assert [item.chunk_id for item in result.hits] == ["d1#c1", "d1#c2"]
    assert result.retriever_stats["fused"] == 2


def test_cosine_rerank_reorders_the_candidates_by_vector_score():
    """同一个候选集：RRF 说 c1 在前，余弦说 c2 在前——重排确实改了顺序。"""
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors(
            [hit("d1#c1", "名次一", 0.2), hit("d1#c2", "名次二", 0.9)], recorder
        ),
        bm25=FakeBm25([], recorder),
    )

    result = search_documents(ctx, "问题", rerank="cosine", rerank_alpha=1.0)

    assert [item.chunk_id for item in result.hits] == ["d1#c2", "d1#c1"]
    assert result.retriever_stats["fused"] == 2


def test_cosine_rerank_keeps_the_return_count_at_top_k():
    """重排改顺序不改编队人数：池子 30 条、返回仍是 8 条。"""
    recorder = []
    ctx, _ = make_ctx(
        vectors=FakeVectors(pool_hits("v", 30), recorder),
        bm25=FakeBm25(pool_hits("b", 30), recorder),
    )

    result = search_documents(ctx, "问题", pool_depth=30, rerank="cosine")

    assert len(result.hits) == 8


def test_cosine_rerank_can_lift_a_candidate_that_only_the_deep_pool_had():
    """池深与重排必须配套：候选只取前 8 条时，第 24 名的块压根没机会被重排捞起来。

    「浅池」这一档**显式**传 `pool_depth=0`：线上默认池深已经从 8（= top_k）改成 50，
    再靠默认值就复现不出「候选里没有那条块」这个前提了（第三十四~三十八节的历史表同理，
    复算它们一律显式传 0）。
    """
    recorder = []
    deep = [hit(f"d1#c{i:03d}", f"第 {i} 段", 1.0 if i == 23 else 0.1) for i in range(24)]
    ctx, _ = make_ctx(vectors=FakeVectors(deep, recorder), bm25=FakeBm25([], recorder))

    shallow = search_documents(ctx, "问题", pool_depth=0, rerank="cosine", rerank_alpha=1.0)
    deep_result = search_documents(ctx, "问题", pool_depth=24, rerank="cosine", rerank_alpha=1.0)

    assert "d1#c023" not in [item.chunk_id for item in shallow.hits]
    assert deep_result.hits[0].chunk_id == "d1#c023"


def test_rerank_mode_is_validated_before_any_retriever_runs():
    """模式名打错时不许先花掉一次嵌入：校验必须排在检索之前。"""
    ctx, recorder = make_ctx()

    with pytest.raises(ValueError, match="rerank"):
        search_documents(ctx, "问题", rerank="magic")

    assert recorder == []


def test_cross_rerank_orders_by_the_reranker_and_hands_it_the_query_and_texts():
    """`rerank="cross"` 把**查询原文 + 候选原文**交给重排器，顺序完全由它的分值决定。

    假重排器刻意把候选反过来打分：RRF 说 c1 在前，重排后 c2 必须在前——只验「重排器被调过」
    是不够的，那验不出顺序真的换过。

    这一条刻意**不给身份表**（`doc_meta=None`）：没有身份时重排文本逐字等于原文，这就是
    「装配层没喂 manifest」的降级口径。
    """
    recorder = []
    seen: list[tuple[str, list[str]]] = []

    def fake_reranker(query, documents):
        seen.append((query, list(documents)))
        # 名次越靠后的候选给分越高：要的就是与 RRF 相反的结论。
        return [0.1 * index for index in range(len(documents))]

    ctx, _ = make_ctx(
        vectors=FakeVectors(
            [hit("d1#c1", "名次一", 0.2), hit("d1#c2", "名次二", 0.9)], recorder
        ),
        bm25=FakeBm25([], recorder),
        reranker=fake_reranker,
    )

    # 显式锁 0 = 整池重排：这条验的是重排器语义，不是默认档（默认锁 4 会把 2 条候选全
    # 留在头部、尾池为空 → 重排器一次都不调）。
    result = search_documents(ctx, "净利润是多少", rerank="cross", rerank_lock=0)

    assert [item.chunk_id for item in result.hits] == ["d1#c2", "d1#c1"]
    assert seen == [("净利润是多少", ["名次一", "名次二"])]
    # score 报的是重排分（这里 c2 得了 0.1、c1 得了 0.0），不再是 RRF 分。
    assert [item.score for item in result.hits] == [pytest.approx(0.1), pytest.approx(0.0)]


def test_cross_rerank_hands_the_reranker_the_identity_prefix_plus_body():
    """有身份表时，重排文本 = **文档身份前缀 + 原文**，不是裸原文。

    这条修的是一个实测出来的坑（`tmp/probe_rerank_text.py`，v3 池深 12）：行级表格块的
    正文里没有公司/证券代码/报告期，裸原文喂进去 21/25、MRR 0.3411；加上身份前缀
    22/25、MRR 0.6133。顺序断言放上面那条，这条只钉**送进去的文本形状**。
    """
    seen: list[list[str]] = []

    def fake_reranker(query, documents):
        seen.append(list(documents))
        return [1.0] * len(documents)

    meta = DocMeta(
        doc_id="d1", company="甲公司", code="002129", period="2024年年度", title="年度报告"
    )
    ctx, _ = make_ctx(
        vectors=FakeVectors([hit("d1#c1", "营业收入 1,234 万元", 0.9)], []),
        bm25=FakeBm25([], []),
        doc_meta={"d1": meta},
        reranker=fake_reranker,
    )

    # 显式锁 0：只有 1 条候选，默认锁 4 会把尾池吃空、重排器一次都不调。
    search_documents(ctx, "营业收入是多少", rerank="cross", rerank_lock=0)

    assert seen == [["甲公司（002129） 2024年年度 年度报告\n营业收入 1,234 万元"]]


def test_rerank_document_text_omits_the_table_lead_line_and_section_path():
    """重排文本只加**文档级**身份前缀：导读行与章节路径实测都不加分（MRR 反而更低）。

    只加导读行甚至比裸原文更差（19/25 vs 21/25）：少了身份，期间行会把重排器推向错年份。
    所以这里钉死「不加」——将来谁想加回来，得先拿出超过 0.6133 的 MRR。
    """
    meta = DocMeta(
        doc_id="d1", company="甲公司", code="002129", period="2024年年度", title="年度报告"
    )
    text = "营业收入 2024年 2023年\n1,234\n5,678"
    row = hit("d1#c1", text, 0.9)

    assert tools_module.rerank_document_text(row, {"d1": meta}) == (
        f"甲公司（002129） 2024年年度 年度报告\n{text}"
    )
    # 没有身份表时逐字退回原文（装配层缺 manifest 的降级口径）
    assert tools_module.rerank_document_text(row, None) == text
    assert tools_module.rerank_document_text(row, {"other": meta}) == text


def test_cross_rerank_scores_the_whole_pool_not_just_the_returned_top_k():
    """重排必须覆盖**整个候选池**：只重排 top-8 就永远捞不出池子深处的 gold。

    这正是「池深 12 + 重排」这条组合的意义（第三十九节的池深曲线：深池里含 gold，但排不进
    前 8）。重排器如果只看到 8 条，池子再深也没用。
    """
    recorder = []
    seen: list[int] = []

    def fake_reranker(query, documents):
        seen.append(len(documents))
        # 只有最后一条（池子里最深的那条）得高分：它必须被捞到第一名。
        return [0.0] * (len(documents) - 1) + [1.0]

    deep = [hit(f"d1#c{i:03d}", f"第 {i} 段", 0.9) for i in range(12)]
    ctx, _ = make_ctx(vectors=FakeVectors(deep, recorder), bm25=FakeBm25([], recorder), reranker=fake_reranker)

    # 显式锁 0：这条的标题就是「整个池子」，默认锁 4 只会送尾巴 8 条。
    result = search_documents(ctx, "问题", pool_depth=12, rerank="cross", rerank_lock=0)

    assert seen == [12]  # 池深 12 全部送进重排器，不是 8
    assert result.hits[0].chunk_id == "d1#c011"
    assert len(result.hits) == 8


def test_cross_rerank_without_a_reranker_fails_before_spending_an_embedding():
    """装配层没给 `ctx.reranker` 时必须当场报错，且**一次嵌入都不许花**。

    静默退回 rrf 会量出一列假的「重排没效果」——那比报错更难发现，所以这条钉在便宜检查里。
    `cross` 现在是**线上默认**，所以这条同时意味着：没装配重排器的离线上下文**必须显式**
    传 `rerank="rrf"` 才能跑（`test_explicit_rrf_still_runs_without_a_reranker`）。
    """
    ctx, recorder = make_ctx(reranker=None)

    with pytest.raises(ValueError, match="reranker"):
        search_documents(ctx, "问题", rerank="cross")

    assert recorder == []


def test_fuse_cross_without_a_reranker_also_fails():
    """`_fuse` 是独立函数，缺重排器时也要自己挡住（与未知模式同一条纪律）。"""
    with pytest.raises(ValueError, match="reranker"):
        tools_module._fuse(
            [hit("d1#c1", "一", 0.9)], [], depth=8, top_k=1, rerank="cross", query="问题"
        )


def test_rerank_alpha_is_validated_before_any_retriever_runs():
    ctx, recorder = make_ctx()

    with pytest.raises(ValueError, match="rerank_alpha"):
        search_documents(ctx, "问题", rerank="cosine", rerank_alpha=1.5)

    assert recorder == []


def test_fuse_rejects_a_pool_shallower_than_the_return_count():
    """池比 top_k 还浅时不可能凑够条数：这是配置错误，不是「返回短一点」。"""
    with pytest.raises(ValueError, match="池深"):
        tools_module._fuse([hit("d1#c1", "一", 0.9)], [], depth=1, top_k=8, rerank="rrf")


def test_fuse_rejects_an_unknown_rerank_mode():
    """`_fuse` 自己也要挡住错模式：它是个独立函数，不能只在调用方那里校验。"""
    with pytest.raises(ValueError, match="rerank"):
        tools_module._fuse([hit("d1#c1", "一", 0.9)], [], depth=8, top_k=8, rerank="magic")


def test_fuse_rejects_a_lock_outside_cross_mode():
    """`lock` 只在 `rerank="cross"` 时有意义；其余模式传非 0 要报错，不静默忽略。

    静默忽略的下场是：调用方以为自己在「锁 4 条 + 重排」，拿到的却是纯 RRF 顺序——
    一列看着像重排结果、其实没有重排的数字。
    """
    with pytest.raises(ValueError, match="跨模式|只在 rerank"):
        tools_module._fuse(
            [hit("d1#c1", "一", 0.9)], [], depth=8, top_k=8, rerank="cosine", lock=4
        )


def test_fuse_rejects_a_lock_deeper_than_the_return_count():
    """锁的条数超过返回条数时，头部就吃掉了全部名额，那是配置错误而不是「锁满」。"""
    with pytest.raises(ValueError, match="lock"):
        tools_module._fuse(
            [hit("d1#c1", "一", 0.9)], [], depth=8, top_k=8, rerank="cross", lock=9
        )


def test_cross_rerank_lock_keeps_the_rrf_head_and_only_reranks_the_tail():
    """锁 4 条 + top_k 8 的语义：前 4 条逐字等于 RRF 的前 4 条，只有尾池进重排器。

    这是 P1（重排池与融合池解耦）的有效形态。整池重排是**负结果**（池越深越差，见
    `RERANK_LOCK_DEFAULT` 上面的注释），头尾分开之后 top-8 反而多 1 题命中——但那条数字
    属于 25 题评测集（`tmp/probe_lock_head.py`），这条用例只钉**语义**。

    假重排器刻意把尾池倒过来打分（越靠后分越高）：尾部名次必须被真正翻转，才验得出
    「重排器只拿到尾巴」而不是「重排器被调了一次但顺序没换」。
    """
    recorder: list = []
    calls: list[tuple[str, list[str]]] = []

    def fake_reranker(query, documents):
        calls.append((query, list(documents)))
        return [0.1 * index for index in range(len(documents))]

    vector_hits = pool_hits("v", 12)
    bm25_hits = pool_hits("b", 12)
    texts = {item.chunk_id: item.text for item in [*vector_hits, *bm25_hits]}
    ctx, _ = make_ctx(
        vectors=FakeVectors(vector_hits, recorder),
        bm25=FakeBm25(bm25_hits, recorder),
        reranker=fake_reranker,
    )

    # 先拿同一批候选的纯 RRF 全长顺序：锁定的头部与尾部取的都是它。
    full = [
        item.chunk_id
        for item in search_documents(
            ctx, "问题", pool_depth=12, top_k=12, rerank="rrf"
        ).hits
    ]
    result = search_documents(
        ctx, "问题", pool_depth=12, top_k=8, rerank="cross", rerank_lock=4
    )
    picked = [item.chunk_id for item in result.hits]

    assert len(full) == 12
    assert picked[:4] == full[:4]  # 头部由 RRF 独占
    assert len(calls) == 1
    assert calls[0] == ("问题", [texts[chunk_id] for chunk_id in full[4:]])
    # 尾部 8 条按重排分倒序取前 4：full[4:] 的第 7/6/5/4 条。
    assert picked[4:] == [full[11], full[10], full[9], full[8]]


def test_cross_rerank_lock_equal_to_top_k_never_calls_the_reranker():
    """锁满 = 名额全被 RRF 占满：重排器**一次都不许调**（也就一分钱不花）。"""
    recorder: list = []
    calls: list[int] = []

    def fake_reranker(query, documents):
        calls.append(len(documents))
        return [1.0] * len(documents)

    ctx, _ = make_ctx(
        vectors=FakeVectors(pool_hits("v", 12), recorder),
        bm25=FakeBm25(pool_hits("b", 12), recorder),
        reranker=fake_reranker,
    )

    # 对比基线显式 rrf：默认档会调一次重排器，`calls == []` 就不再成立。
    plain = search_documents(ctx, "问题", pool_depth=12, top_k=8, rerank="rrf")
    locked = search_documents(
        ctx, "问题", pool_depth=12, top_k=8, rerank="cross", rerank_lock=8
    )

    assert calls == []
    assert [item.chunk_id for item in locked.hits] == [
        item.chunk_id for item in plain.hits
    ]


def test_search_documents_schema_exposes_the_rerank_knobs():
    """两个重排旋钮都要进 schema：否则 MCP 客户端看不到、也没法传。"""
    entry = next(item for item in TOOL_SCHEMAS if item["name"] == "search_documents")
    props = entry["inputSchema"]["properties"]
    assert props["rerank"]["default"] == tools_module.RERANK_DEFAULT
    assert set(props["rerank"]["enum"]) == set(tools_module.RERANK_MODES)
    assert props["rerank_alpha"]["default"] == tools_module.DEFAULT_ALPHA


def sqlite_ctx(
    *,
    vector_hits=(),
    bm25_hits=(),
    chunks=(),
    doc_id="d1",
    others=None,
    doc_meta=None,
):
    """带**真 SQLite** 的 ctx：邻块只能从 `chunks` 表按 ordinal 取，fake 连接会假绿。

    `chunks` 是主文档（`doc_id`）的块、`others` 是 `{doc_id: [块, ...]}` 的额外文档。
    """

    recorder: list = []
    conn = sqlite_store.connect(":memory:")
    sqlite_store.init_schema(conn)
    for one_id, one_chunks in ((doc_id, list(chunks)), *(others or {}).items()):
        sqlite_store.upsert_document(
            conn,
            doc_id=one_id,
            source="s.pdf",
            doc_type="annual_report",
            page_count=1,
            updated_at="2026-09-25T00:00:00+00:00",
        )
        if one_chunks:
            sqlite_store.replace_chunks(conn, one_id, one_chunks)
    return (
        ToolContext(
            conn=conn,
            vectors=FakeVectors(list(vector_hits), recorder),
            bm25=FakeBm25(list(bm25_hits), recorder),
            graph=FakeGraph(),
            artifacts=FakeArtifacts(),
            doc_meta=doc_meta,
            reranker=flat_reranker,
        ),
        recorder,
    )


def run_chunks(count: int, *, doc_id: str = "d1") -> list:
    """一份「ordinal 从 0 连续排到 count-1」的文档；正文由 `chunk()` 工厂写成第 N 段。"""

    return [chunk(f"{doc_id}#c{index}", doc_id=doc_id, ordinal=index) for index in range(count)]


def test_expand_neighbors_default_constant_is_off():
    """默认必须是 0（= 关）；锚点数是常量 3，不跟着参数走。"""
    assert tools_module.EXPAND_NEIGHBORS == 0
    assert tools_module.EXPAND_ANCHORS == 3


def test_expand_neighbors_default_leaves_the_returned_list_untouched():
    """默认关：返回的就是融合顺序的前 top_k，一条邻块也不许插进来。"""
    ctx, _ = sqlite_ctx(vector_hits=[hit("d1#c2", "正文", 0.9)], chunks=run_chunks(5))

    result = search_documents(ctx, "问题")

    assert [item.chunk_id for item in result.hits] == ["d1#c2"]
    assert "neighbours" not in result.retriever_stats


def test_expand_neighbors_inserts_neighbours_after_each_anchor():
    """锚点先占位，再按距离由近到远插邻块——逐条与离线 `expand_window` 相同。"""
    ctx, _ = sqlite_ctx(vector_hits=[hit("d1#c2", "正文", 0.9)], chunks=run_chunks(5))

    result = search_documents(ctx, "问题", expand_neighbors=2)

    assert [item.chunk_id for item in result.hits] == [
        "d1#c2",
        "d1#c1",
        "d1#c3",
        "d1#c0",
        "d1#c4",
    ]


def test_expand_neighbors_truncates_back_to_top_k():
    """返回给模型的条数永远等于 top_k：扩出来的尾巴按 offset 顺序截掉。"""
    ctx, _ = sqlite_ctx(vector_hits=[hit("d1#c2", "正文", 0.9)], chunks=run_chunks(5))

    result = search_documents(ctx, "问题", top_k=3, expand_neighbors=2)

    assert [item.chunk_id for item in result.hits] == ["d1#c2", "d1#c1", "d1#c3"]


def test_expand_neighbors_only_expands_the_first_three_hits():
    """`EXPAND_ANCHORS` 是上限：第 4 条以后的命中不许参与扩窗，否则 prompt 全是邻块。"""
    ctx, _ = sqlite_ctx(
        vector_hits=[hit(f"d1#c{i}", f"第 {i} 段", 0.9) for i in (0, 3, 6, 9, 12)],
        chunks=run_chunks(13),
    )

    result = search_documents(ctx, "问题", expand_neighbors=1)

    ids = [item.chunk_id for item in result.hits]
    assert ids == ["d1#c0", "d1#c1", "d1#c3", "d1#c2", "d1#c4", "d1#c6", "d1#c5", "d1#c7"]
    assert "d1#c9" not in ids and "d1#c8" not in ids


def test_expand_neighbors_never_crosses_the_document_boundary():
    """ordinal 只在文档内有意义：别的文档里 ordinal 恰好接近的块不算邻块。"""
    ctx, _ = sqlite_ctx(
        vector_hits=[hit("d1#c2", "正文", 0.9)],
        chunks=run_chunks(3),
        others={"d2": run_chunks(3, doc_id="d2")},
    )

    result = search_documents(ctx, "问题", expand_neighbors=5)

    assert [item.chunk_id for item in result.hits] == ["d1#c2", "d1#c1", "d1#c0"]


def test_expand_neighbors_keeps_the_anchor_when_sqlite_has_no_rows_for_it():
    """线上与离线**唯一一处**差异：SQLite 里查不到的锚点要留在结果里，只是不扩。"""
    ctx, _ = sqlite_ctx(vector_hits=[hit("d1#c2", "正文", 0.9)])

    result = search_documents(ctx, "问题", expand_neighbors=2)

    assert [item.chunk_id for item in result.hits] == ["d1#c2"]
    assert result.retriever_stats["neighbours"] == 0


def test_neighbours_inherit_the_anchor_retriever_and_score_with_real_provenance():
    """邻块沿用锚点的 `retriever` 与分数（不许发明新取值），溯源字段取自 SQLite 真实行。"""
    ctx, _ = sqlite_ctx(
        vector_hits=[hit("d1#c2", "命中正文", 0.87, retriever="both")],
        chunks=run_chunks(5),
    )

    result = search_documents(ctx, "问题", expand_neighbors=1)

    anchor = result.hits[0]
    neighbour = result.hits[1]
    assert neighbour.chunk_id == "d1#c1"
    # 「继承锚点的分数」= 等于**最终列表里锚点那一行**的分（默认 rerank="rrf" 时它是 RRF 分，
    # 不是向量路的原始余弦 0.87）：邻块自己的分只能来自锚点，不许另算也不许留旧值。
    assert neighbour.retriever == anchor.retriever == "both"
    assert neighbour.score == anchor.score
    assert (neighbour.doc_id, neighbour.source) == ("d1", "s.pdf")
    assert neighbour.text == "第 1 段正文"


def test_expand_neighbors_needs_a_sqlite_connection_before_any_retriever_runs():
    """没连接就取不到邻块：这是配置错误，且必须在花掉一次嵌入之前报出来。"""
    ctx, recorder = make_ctx()

    with pytest.raises(ValueError, match="expand_neighbors"):
        search_documents(ctx, "问题", expand_neighbors=2)

    assert recorder == []


def test_expand_neighbors_rejects_a_negative_width_before_any_retriever_runs():
    ctx, recorder = sqlite_ctx(vector_hits=[hit("d1#c0", "正文", 0.9)], chunks=run_chunks(3))

    with pytest.raises(ValueError, match="expand_neighbors"):
        search_documents(ctx, "问题", expand_neighbors=-1)

    assert recorder == []


def test_expand_neighbors_schema_exposes_the_knob():
    """新旋钮要进 schema：否则 MCP/HTTP 客户端看不到它、也传不进来。"""
    entry = next(item for item in TOOL_SCHEMAS if item["name"] == "search_documents")
    props = entry["inputSchema"]["properties"]
    assert props["expand_neighbors"]["type"] == "integer"
    assert props["expand_neighbors"]["default"] == tools_module.EXPAND_NEIGHBORS


def test_expand_neighbors_records_the_neighbour_count_in_stats():
    """`retriever_stats` 要能回答「这一轮插进来几条新块」——否则收益无从归因。"""
    ctx, _ = sqlite_ctx(vector_hits=[hit("d1#c2", "正文", 0.9)], chunks=run_chunks(5))

    result = search_documents(ctx, "问题", expand_neighbors=2)

    assert result.retriever_stats["neighbours"] == 4
    assert result.retriever_stats["fused"] == 1


def test_expand_neighbors_applies_to_the_final_list_after_two_stage():
    """扩窗作用在**最终列表**上：两段式改了顺序，锚点就跟着换。

    拿第一段的结果去扩，是这条改动最容易埋进去的错——它只在两段式真跑起来时才显形，
    因此必须有一条用例把两个旋钮一起打开。
    """
    recorder: list = []
    conn = sqlite_store.connect(":memory:")
    sqlite_store.init_schema(conn)
    sqlite_store.upsert_document(
        conn,
        doc_id="d1",
        source="s.pdf",
        doc_type="annual_report",
        page_count=1,
        updated_at="2026-09-25T00:00:00+00:00",
    )
    sqlite_store.replace_chunks(conn, "d1", run_chunks(10))
    ctx = ToolContext(
        conn=conn,
        vectors=ScriptedVectors(
            [hit("d1#c5", "第一段", 0.9)], [hit("d1#c7", "第二段", 0.9)], recorder
        ),
        bm25=FakeBm25([], recorder),
        graph=FakeGraph(),
        artifacts=FakeArtifacts(),
        doc_meta=None,
        reranker=flat_reranker,
    )

    result = search_documents(ctx, "问题", two_stage_docs=1, expand_neighbors=1)

    assert [item.chunk_id for item in result.hits] == [
        "d1#c7",
        "d1#c6",
        "d1#c8",
        "d1#c5",
        "d1#c4",
    ]
