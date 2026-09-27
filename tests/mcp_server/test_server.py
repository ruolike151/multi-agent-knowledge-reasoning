import asyncio
import json
from pathlib import Path

import pytest
from mcp import types

from akr.ingest.chunker import Chunk
from akr.mcp_server import server as server_module
from akr.mcp_server.server import (
    SERVER_NAME,
    build_tool_context,
    call_tool_handler,
    create_server,
    list_tools_handler,
)
from akr.mcp_server.tools import ToolContext
from akr.storage import sqlite_store
from akr.storage.artifact_store import ArtifactStore

TOOL_NAMES = {
    "search_documents",
    "walk_graph",
    "resolve_entity",
    "fetch_chunk",
    "list_documents",
}


def make_ctx(tmp_path):
    conn = sqlite_store.connect(tmp_path / "akr.sqlite")
    sqlite_store.init_schema(conn)
    return ToolContext(
        conn=conn,
        vectors=None,
        bm25=None,
        graph=None,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
    )


def test_list_tools_handler_exposes_five_schemas():
    schemas = list_tools_handler()
    assert {schema["name"] for schema in schemas} == TOOL_NAMES
    assert all("description" in schema for schema in schemas)


def test_call_tool_handler_returns_json_text(tmp_path):
    text = call_tool_handler(make_ctx(tmp_path), "list_documents", {})
    assert json.loads(text) == {"documents": []}


def test_call_tool_handler_reports_tool_errors_as_payload(tmp_path):
    text = call_tool_handler(make_ctx(tmp_path), "fetch_chunk", {"chunk_id": "d1#cmissing"})
    payload = json.loads(text)
    assert "error" in payload
    assert "d1#cmissing" in payload["error"]


def test_create_server_registers_list_and_call_handlers(tmp_path):
    server = create_server(make_ctx(tmp_path))
    assert server.name == SERVER_NAME

    listed = asyncio.run(server.get_request_handler("tools/list").handler(None, None))
    assert isinstance(listed, types.ListToolsResult)
    assert {tool.name for tool in listed.tools} == TOOL_NAMES


def test_create_server_call_handler_returns_text_content(tmp_path):
    server = create_server(make_ctx(tmp_path))
    call = server.get_request_handler("tools/call").handler

    result = asyncio.run(call(None, types.CallToolRequestParams(name="list_documents")))
    assert isinstance(result, types.CallToolResult)
    assert json.loads(result.content[0].text) == {"documents": []}


def test_build_tool_context_fails_fast_without_neo4j_uri(tmp_path):
    with pytest.raises(ValueError, match="NEO4J_URI"):
        build_tool_context(db_path=tmp_path / "akr.sqlite", artifacts_root=tmp_path / "artifacts")


# ---- 文档身份的装配（2026-09-25，见 `akr.ingest.doc_meta` 与 `ToolContext.doc_meta`）----
# 身份前缀与查询收窄必须**成对**生效：只加前缀会让 `stage2@8/M3` 从 10/25 掉到 9/25。
# 所以这里断言的不是「参数传下去了」，而是「装配完的上下文真的能分开两家同业公司」。


class FakeGraphStore:
    @staticmethod
    def connect(uri, user, password):  # 签名与真实 GraphStore 对齐
        return object()


def make_chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c0000",
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        ordinal=0,
        text=text,
        content_hash=f"hash-{doc_id}",
        section_path="第一章",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def write_bm25(tmp_path, chunks) -> str:
    from akr.storage.bm25_index import Bm25Index

    index = Bm25Index()
    index.build(chunks)
    path = tmp_path / "bm25.json"
    index.save(path)
    return str(path)


def write_manifest(tmp_path) -> str:
    path = tmp_path / "manifest.json"
    documents = [
        {"doc_id": "doc2", "company": "乙公司", "code": "300274", "period": "2024年年度"}
    ]
    path.write_text(json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8")
    return str(path)


def identical_annual_report_chunks():
    """两家同业公司的同一段文字：只能靠文档身份分开，词法本身分不出。"""

    text = "营业收入为 100 亿元"
    return [make_chunk("doc1", text), make_chunk("doc2", text)]


def build_offline_context(tmp_path, monkeypatch, **overrides):
    """只替掉 Neo4j（它需要真实服务）：其余依赖走真实实现，装配逻辑才被测到。"""

    monkeypatch.setattr(server_module, "GraphStore", FakeGraphStore)
    return build_tool_context(
        db_path=tmp_path / "akr.sqlite",
        artifacts_root=tmp_path / "artifacts",
        chroma_path=None,
        bm25_path=write_bm25(tmp_path, identical_annual_report_chunks()),
        neo4j_uri="bolt://127.0.0.1:7687",  # 已被 FakeGraphStore 接住，不会真的连
        **overrides,
    )


def test_build_tool_context_wires_identity_into_both_bm25_and_tools(tmp_path, monkeypatch):
    ctx = build_offline_context(tmp_path, monkeypatch, manifest_path=write_manifest(tmp_path))
    try:
        assert set(ctx.doc_meta) == {"doc2"}
        # 前缀已进打分矩阵：同一句话在两家措辞相同的年报里优先命中被点名的那家。
        assert ctx.bm25.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc2"
        # 命中里的原文没被前缀污染（引用与 artifact 取的是同一份正文）。
        assert ctx.bm25.search("乙公司（300274）的营业收入", top_k=1)[0].text == "营业收入为 100 亿元"
    finally:
        ctx.conn.close()


def test_build_tool_context_degrades_when_the_manifest_is_missing(tmp_path, monkeypatch):
    ctx = build_offline_context(tmp_path, monkeypatch, manifest_path=tmp_path / "不存在.json")
    try:
        assert ctx.doc_meta == {}
        assert ctx.bm25.search("乙公司（300274）的营业收入", top_k=2)[0].doc_id == "doc1"
    finally:
        ctx.conn.close()


# ---- 向量库接线（2026-09-26，见内部偏差日志第三十九节）----
# 线上装配指向 unit 级库（`data/chroma-v3`）**并且**必须带上 unit → 父块映射：
# 少了映射，查出来的 unit id 落在 gold（父块 id）之外，检索会以「召回率为 0」的形状骗人。
# 所以这里断言两件事不是「参数传下去了」，而是「口径没被静默换掉」。


def test_the_online_vector_library_is_the_unit_level_one_with_its_mapping():
    """线上三个常量：unit 级库 + 映射 + 聚合折叠。改它们等于改线上口径，所以钉住。"""

    from akr.mcp_server.server import (
        DEFAULT_CHROMA_PATH,
        DEFAULT_UNIT_MAP_PATH,
        DEFAULT_VECTOR_FOLD,
    )
    from akr.storage.vector_store import FOLD_MODES

    assert DEFAULT_CHROMA_PATH == Path("data/chroma-v3")
    assert DEFAULT_UNIT_MAP_PATH == Path("data/unit-map.json")
    assert DEFAULT_VECTOR_FOLD == "rrf_sum"
    assert DEFAULT_VECTOR_FOLD in FOLD_MODES


def test_build_tool_context_refuses_a_unit_level_library_without_the_mapping(tmp_path, monkeypatch):
    """unit 级库 + 缺映射必须**当场报错**：静默不折叠只会让整列数字看起来是 0。

    目录不必真存在（也不必真是 unit 级）：映射的存在性检查排在建库之前，所以这条用例
    花不了任何一次嵌入（`embed_fn` 也是假的）。
    """

    monkeypatch.setattr(server_module, "GraphStore", FakeGraphStore)
    with pytest.raises(ValueError, match="unit → 父块"):
        build_tool_context(
            db_path=tmp_path / "akr.sqlite",
            artifacts_root=tmp_path / "artifacts",
            chroma_path=tmp_path / "chroma",
            unit_map_path=tmp_path / "缺.json",
            neo4j_uri="bolt://127.0.0.1:7687",
            embed_fn=lambda texts: [],
        )


def test_build_tool_context_hands_the_mapping_and_fold_to_the_vector_store(tmp_path, monkeypatch):
    """装了映射还不够：`id_map` 与 `fold` 必须真的进 `VectorStore`（少一个就是另一件事）。"""

    from akr.storage import vector_store as vector_store_module

    seen: dict = {}

    class FakeVectorStore:
        def __init__(self, path, embed_fn, *, index_text_fn=None, id_map=None, fold="best"):
            seen.update(path=path, id_map=id_map, fold=fold)

    unit_map = tmp_path / "unit-map.json"
    unit_map.write_text('{"d1#u0": "d1#c0000"}', encoding="utf-8")

    monkeypatch.setattr(server_module, "GraphStore", FakeGraphStore)
    monkeypatch.setattr(vector_store_module, "VectorStore", FakeVectorStore)
    ctx = build_tool_context(
        db_path=tmp_path / "akr.sqlite",
        artifacts_root=tmp_path / "artifacts",
        chroma_path=tmp_path / "chroma",
        unit_map_path=unit_map,
        bm25_path=None,
        manifest_path=tmp_path / "不存在.json",
        neo4j_uri="bolt://127.0.0.1:7687",
        embed_fn=lambda texts: [],
    )
    ctx.conn.close()

    assert seen == {
        "path": tmp_path / "chroma",
        "id_map": {"d1#u0": "d1#c0000"},
        "fold": "rrf_sum",
    }


# ---- BM25 侧表格导读行（2026-09-26 接线，见内部偏差日志第四十节）----
# 导读行是**装配层**旋钮（`Bm25Index(index_text_fn=...)`），走的是与向量侧同一份
# `akr.ingest.table_context` 规则：它只改打分矩阵，`Chunk.text` 一个字不动。
# 离线脚本（`scripts/retrieval_pool_depth.py` 等）必须跟线上用同一个口径，否则「对账」对的是两台机器。


TABLE_CHUNK_TEXT = (
    "2024年 2023年 本年比上年增减\n"
    "营业收入（元） 100 90 11%\n"
    "净利润（元） 20 10 100%"
)


def test_build_tool_context_wires_the_table_lead_into_the_bm25_index(tmp_path, monkeypatch):
    """`Bm25Index.load` 必须拿到导读行那条规则，而且它**不是恒等变换**。"""

    from akr.ingest.doc_meta import index_text
    from akr.ingest.table_context import table_lead

    captured: dict = {}

    class FakeBm25:
        @staticmethod
        def load(path, doc_metas=None, index_text_fn=None):
            captured.update(index_text_fn=index_text_fn)
            return object()

    monkeypatch.setattr(server_module, "GraphStore", FakeGraphStore)
    monkeypatch.setattr(server_module, "Bm25Index", FakeBm25)

    ctx = build_tool_context(
        db_path=tmp_path / "akr.sqlite",
        artifacts_root=tmp_path / "artifacts",
        chroma_path=None,
        bm25_path=write_bm25(tmp_path, identical_annual_report_chunks()),
        manifest_path=tmp_path / "不存在.json",
        neo4j_uri="bolt://127.0.0.1:7687",
    )
    ctx.conn.close()

    render = captured["index_text_fn"]
    assert render is not None
    sample = make_chunk("doc1", TABLE_CHUNK_TEXT)
    assert render(sample).startswith(table_lead(TABLE_CHUNK_TEXT))
    # 与「只用身份前缀 + 原文」的默认口径不同——接线没生效时这两者会一模一样。
    assert render(sample) != index_text(sample, None)
    # 正文块上它是恒等变换：新旋钮不改变非表格块的打分。
    prose = make_chunk("doc1", "营业收入为 100 亿元")
    assert render(prose) == index_text(prose, None)


# ---- cross-encoder 重排器的装配（默认关，只有生产路径才自建）----


def test_an_explicit_reranker_is_passed_through_untouched(tmp_path, monkeypatch):
    """调用方给了重排器就用它：`build_tool_context` 不去动它、也不再自己建一个。"""

    def my_reranker(query, documents):  # pragma: no cover - 只验身份
        return [0.0] * len(documents)

    ctx = build_offline_context(tmp_path, monkeypatch, rerank_fn=my_reranker)
    try:
        assert ctx.reranker is my_reranker
    finally:
        ctx.conn.close()


def test_offline_assembly_does_not_silently_build_an_api_reranker(tmp_path, monkeypatch):
    """自带 `embed_fn` 的调用方（离线脚本 / 单测）拿到的是 `reranker=None`。

    这条是**不花钱**的保证：`rerank="cross"` 在这种上下文里会当场报错，而不是
    悄悄发一次网络请求，让离线对账脚本在别人机器上开始烧额度。
    """

    from akr.storage import rerank as rerank_module
    from akr.storage import vector_store as vector_store_module

    def explode(*args, **kwargs):  # pragma: no cover - 被调用即失败
        raise AssertionError("离线装配不该去建 API 重排器")

    class FakeVectorStore:
        def __init__(self, path, embed_fn, *, index_text_fn=None, id_map=None, fold="best"):
            pass

    unit_map = tmp_path / "unit-map.json"
    unit_map.write_text('{"d1#u0": "d1#c0000"}', encoding="utf-8")

    monkeypatch.setattr(rerank_module, "build_reranker", explode)
    monkeypatch.setattr(vector_store_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(server_module, "GraphStore", FakeGraphStore)

    ctx = build_tool_context(
        db_path=tmp_path / "akr.sqlite",
        artifacts_root=tmp_path / "artifacts",
        chroma_path=tmp_path / "chroma",
        unit_map_path=unit_map,
        bm25_path=None,
        manifest_path=tmp_path / "不存在.json",
        neo4j_uri="bolt://127.0.0.1:7687",
        embed_fn=lambda texts: [],
    )
    try:
        assert ctx.reranker is None
    finally:
        ctx.conn.close()


def test_the_production_path_builds_the_reranker_and_carries_the_usage_path(tmp_path, monkeypatch):
    """本函数自己建嵌入器（= 线上路径）时，顺带用同一把 Key 建重排器，并带上账本路径。"""

    from akr.storage import rerank as rerank_module
    from akr.storage import embeddings as embeddings_module
    from akr.storage import vector_store as vector_store_module

    seen: dict = {}
    sentinel = lambda query, documents: [0.0] * len(documents)  # noqa: E731

    def fake_build_reranker(*, usage_path=None):
        seen["usage_path"] = usage_path
        return sentinel

    class FakeVectorStore:
        def __init__(self, path, embed_fn, *, index_text_fn=None, id_map=None, fold="best"):
            pass

    unit_map = tmp_path / "unit-map.json"
    unit_map.write_text('{"d1#u0": "d1#c0000"}', encoding="utf-8")

    monkeypatch.setattr(server_module, "GraphStore", FakeGraphStore)
    monkeypatch.setattr(vector_store_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(rerank_module, "build_reranker", fake_build_reranker)
    monkeypatch.setattr(embeddings_module, "build_embedder", lambda: (lambda texts: []))

    ctx = build_tool_context(
        db_path=tmp_path / "akr.sqlite",
        artifacts_root=tmp_path / "artifacts",
        chroma_path=tmp_path / "chroma",
        unit_map_path=unit_map,
        bm25_path=None,
        manifest_path=tmp_path / "不存在.json",
        neo4j_uri="bolt://127.0.0.1:7687",
        rerank_usage_path=tmp_path / "rerank-usage.json",
    )
    ctx.conn.close()

    assert ctx.reranker is sentinel
    assert seen["usage_path"] == tmp_path / "rerank-usage.json"
