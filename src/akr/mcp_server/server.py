"""把 Task 20 的 5 个工具挂到 MCP stdio server 上。

进程边界：agent 侧通过 ``agents/tool_gateway.McpStdioGateway`` 以子进程方式调用本 server。
因此本模块不调用任何模型（工具层不许有模型依赖），也不读写 agent 侧的状态。

SDK 版本：本模块按 mcp 2.x 的低层 ``Server`` 写（handler 由构造参数注入）。
mcp 1.x 的 ``@server.list_tools()`` / ``@server.call_tool()`` 装饰器在 2.x 已移除，
``Tool.inputSchema`` 与 ``CallToolResult.isError`` 在 2.x 分别改名
``Tool.input_schema`` / ``CallToolResult.is_error``（pydantic 别名仍接受旧名）。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from akr.mcp_server.schemas import TOOL_SCHEMAS
from akr.mcp_server.tools import ToolContext, dispatch_safe
from akr.ingest.doc_meta import load_doc_meta_or_empty
from akr.storage import sqlite_store
from akr.storage.artifact_store import ArtifactStore
from akr.storage.bm25_index import Bm25Index
from akr.storage.graph_store import GraphStore

if TYPE_CHECKING:  # pragma: no cover - 只为类型标注
    from mcp.server import Server

    from akr.storage.vector_store import EmbedFn

SERVER_NAME = "akr-knowledge-hub"

DEFAULT_DB_PATH = Path("data/akr.sqlite")
DEFAULT_ARTIFACTS_ROOT = Path("data/artifacts")
# 线上向量库：**unit 级**索引（`scripts/build_unit_chunks.py` 建的 400 字表格行级 unit +
# 导读行重嵌）。2026-09-26 从 `data/chroma`（块级、原文嵌入）切过来：深召回 recall@50
# 由 26.0% 提到 74.0%（纯向量 9/25 → 21/25），代价只有建库那一次嵌入（账本
# `tmp/chroma-v3-usage.json`），查询侧不额外花 token。`data/chroma` 仍在原地，是历史数字的基准。
DEFAULT_CHROMA_PATH = Path("data/chroma-v3")
# 上面那份 unit 级索引的 unit → 父块 id 映射（同一脚本落盘）。装配时必须带上：查出来的是
# unit id，不折回父块就落在 gold（父块 id）空间之外，检索会以「召回率为 0」的形状骗过人。
DEFAULT_UNIT_MAP_PATH = Path("data/unit-map.json")
DEFAULT_BM25_PATH = Path("data/bm25.json")
DEFAULT_MANIFEST_PATH = Path("data/manifest.json")

# 线上向量库的折叠口径（白名单在 `akr.storage.vector_store.FOLD_MODES`）：
#   * "best"    = 历史行为：同一父块只留名次最好的那条 unit；
#   * "rrf_sum" = 同一父块的多条 unit 按 1/(60+名次) 累加后再排（`FOLD_WINDOW=100` 条原始行）。
# 为什么线上取后者：v3 的收益有一大块本来被"只看最好名次"吃掉——同一父块有 3 条 unit 落在
# 前 20 是强信号，压成一条候选的名次就没了。离线探针（25 题、只读重放、零 token，
# `tmp/v3_parent_agg_probe.py`）实测纯向量 top-8 12/25 → 19/25、MRR@8 0.1840 → 0.5054；
# 放进「融合 + 问句收窄」那条线上路径 hit@8 20/25 → 21/25、MRR@8 0.4920 → 0.5613。
# 口径仍是在同一批 25 题上挑的（四个候选口径都落在 17–19/25，方向一致、绝对数待更大格子定案）。
DEFAULT_VECTOR_FOLD = "rrf_sum"


def _env_path(name: str, default: str) -> Path | None:
    """读路径型环境变量：未设置 → 默认值；显式设成空串 → 关闭这一路依赖。"""

    raw = os.environ.get(name)
    if raw is None:
        return Path(default)
    raw = raw.strip()
    return Path(raw) if raw else None


def _env_flag(name: str, default: bool = False) -> bool:
    """读布尔环境变量：`1/true/yes/on`（大小写不敏感）为真，其余为假，未设置取默认。"""

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_path(name: str) -> Path | None:
    """读**可选**路径型环境变量：未设置或设成空串都是「这一路不开」（返回 None）。

    与 `_env_path` 的差别只在默认值：`_env_path` 的默认值是「有文件的」（库、清单），
    没设置也得指回默认那份；这里的两份账本（重排用量）默认**不写**，写了反而是多一份状态。
    """

    raw = (os.environ.get(name) or "").strip()
    return Path(raw) if raw else None


def list_tools_handler() -> list[dict[str, Any]]:
    """MCP ``list_tools`` 的实现：唯一事实来源是 Task 20 的 ``TOOL_SCHEMAS``。"""

    return [dict(schema) for schema in TOOL_SCHEMAS]


def call_tool_handler(
    ctx: ToolContext,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """MCP ``call_tool`` 的实现：返回 JSON 文本，工具错误降级为 ``{"error": ...}`` 载荷。

    只有工具层已知的错误（未知工具、缺参数、artifact 不存在……）被 ``dispatch_safe`` 接住；
    未预期异常继续向上抛，避免把 bug 伪装成"工具暂时不可用"。
    """

    payload = dispatch_safe(ctx, name, arguments or {})
    return json.dumps(payload, ensure_ascii=False, default=str)


def build_tool_context(
    *,
    db_path: str | Path,
    artifacts_root: str | Path,
    chroma_path: str | Path | None = None,
    unit_map_path: str | Path | None = DEFAULT_UNIT_MAP_PATH,
    bm25_path: str | Path | None = None,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
    neo4j_uri: str | None = None,
    neo4j_user: str = "neo4j",
    neo4j_password: str = "akr-local-pass",
    embed_fn: "EmbedFn | None" = None,
    vector_doc_prefix: bool = False,
    vector_fold: str = DEFAULT_VECTOR_FOLD,
    rerank_fn: Any = None,
    rerank_usage_path: str | Path | None = None,
) -> ToolContext:
    """装配 5 个工具需要的全部依赖。

    缺 ``neo4j_uri`` 立刻 ``ValueError``：图存储是 ``walk_graph`` 的硬依赖，
    与其在第一次调用时抛 Neo4j 连接错误，不如在启动期就失败。

    ``manifest_path`` 提供文档身份（公司 / 证券代码 / 报告期 / 标题），缺文件只是
    「没有身份前缀、检索退化成全库」——不该让工具进程起不来，所以走容错加载。
    它同时喂给两处，且必须成对生效（2026-09-25 只读 A/B，`scripts/retrieval_ab.py`）：
    只加索引前缀会让 ``stage2@8/M3`` 从 10/25 掉到 9/25，加上查询侧收窄才是 12/25。

    ``unit_map_path``（默认 `data/unit-map.json`）提供 unit → 父块 id 的映射，喂给
    ``VectorStore(id_map=...)``：`chroma_path` 那份库是 **unit 级**的，查出来的 id 是 unit id，
    不折回父块就一律对不上 gold（评测口径是父块）。**缺文件直接失败**，理由与
    `scripts/retrieval_pool_depth.py` 的 `resolve_id_maps` 一样——静默不折叠会让整列数字以
    「召回率为 0」的形状骗过所有检查。要用块级库（`data/chroma`）时显式传
    ``unit_map_path=None`` 并同时传 ``vector_fold="best"``。

    ``vector_fold`` 决定同一父块的多条 unit 怎么合成一个名次（见 ``DEFAULT_VECTOR_FOLD``）。
    ``embed_fn`` 与 ``fold`` 的口径都必须与**建库时**一致：库是带前缀/导读行嵌的，
    这里再叠一层 ``index_text_fn`` 只会让新增 chunk 与库内向量落在两套文本口径上。

    ``vector_doc_prefix=True``（环境变量 ``AKR_VECTOR_DOC_PREFIX``）让**向量侧嵌入文本**
    也带同一套身份前缀，但只影响 ``VectorStore.add``——也就是「这份库是怎么建的」。
    它必须与**建库时**的口径一致：库里是带前缀建的，这里不开就会让新增 chunk 与老向量
    落在两套文本口径上。默认关，因为 v3 的索引文本由 `scripts/build_unit_chunks.py`
    在**建库时**就写进了 dump（前缀 + 导读行 + unit 原文），装配层再叠一层就是双重前缀。

    ``rerank_fn``（默认 None）是 cross-encoder 重排器（`akr.storage.rerank.RerankFn`）。
    与 ``embed_fn`` 同一套规矩：调用方给了就用它；没给且**本函数自己**去建嵌入器（即
    ``embed_fn is None``）时，顺带用同一把硅基流动 Key 建一个——两者同源，多建一个不会
    引入新的失败模式。

    **2026-09-26 起这里必须真的建出重排器**：工具层默认档已经是 ``rerank="cross"`` + 锁 4
    （见 `akr.mcp_server.tools.RERANK_DEFAULT`），而 ``cross`` 缺 ``ctx.reranker`` 会**当场
    报错**（不静默退回 rrf）。所以 ``chroma_path`` 非空 + 本函数自建嵌入器 = 线上路径每次
    `search_documents` 都发一次 ``/v1/rerank``（池深 50 锁 4 → 46 条候选）。
    ``rerank_usage_path``（环境变量 ``AKR_RERANK_USAGE_PATH``，默认 None）决定这次调用
    **记不记账**：不传就是「花了但不落账本」，产物里只会看到它没记。
    """

    if not neo4j_uri:
        raise ValueError("缺少 NEO4J_URI：请设置环境变量 NEO4J_URI（见 .env.example）")

    conn = sqlite_store.connect(db_path)
    sqlite_store.init_schema(conn)
    artifacts = ArtifactStore(artifacts_root)
    graph = GraphStore.connect(neo4j_uri, neo4j_user, neo4j_password)
    doc_metas = load_doc_meta_or_empty(manifest_path)
    # 谁去建嵌入器 == 谁走「读 .env 的生产路径」：本函数自己建嵌入器时，重排器也用同一把
    # Key 建一份；调用方自带 embed_fn（单测 / 离线脚本）时不碰网络，reranker 保持 None。
    own_embedder = embed_fn is None

    vectors = None
    if chroma_path is not None:
        from akr.storage.vector_store import VectorStore, load_id_map

        if embed_fn is None:
            from akr.storage.embeddings import build_embedder

            embed_fn = build_embedder()
        index_text_fn = None
        if vector_doc_prefix:
            if not doc_metas:
                raise ValueError(
                    "AKR_VECTOR_DOC_PREFIX 开着，但 manifest 里没有文档身份："
                    "这会让新入库的 chunk 与库内既有向量落在两套文本口径上，故直接失败"
                )
            from akr.ingest.doc_meta import make_index_text_fn

            index_text_fn = make_index_text_fn(doc_metas)
        id_map = None
        if unit_map_path is not None:
            resolved_unit_map = Path(unit_map_path)
            if not resolved_unit_map.exists():
                raise ValueError(
                    f"缺 unit → 父块 映射：{resolved_unit_map}"
                    "（先跑 scripts\\build_unit_chunks.py）。unit 级的向量库少了这份映射，"
                    "返回的 chunk_id 会落在 gold 之外、检索看起来「召回率为 0」——"
                    "所以这里直接失败，而不是静默不折叠；块级库请显式传 unit_map_path=None。"
                )
            id_map = load_id_map(resolved_unit_map)
        vectors = VectorStore(
            chroma_path,
            embed_fn,
            index_text_fn=index_text_fn,
            id_map=id_map,
            fold=vector_fold,
        )

    bm25 = None
    if bm25_path is not None and Path(bm25_path).exists():
        # 表格导读行（`akr.ingest.table_context`）走 `index_text_fn`：它只改**打分矩阵**，
        # `Chunk.text` 一个字不动，所以不需要重灌语料、不需要重新嵌入。实测（25 道可答题，
        # `data/retrieval-ab.json` → `summary["*+table"]`）：线上配置那一列 12/25 → 14/25、
        # BM25 单路 6/25 → 9/25，**零 token**。缺 manifest 时身份前缀为空、导读行照旧生效
        # （与 `index_text` 的降级口径一致）。
        from akr.ingest.table_context import make_table_index_text_fn

        bm25 = Bm25Index.load(
            bm25_path,
            doc_metas=doc_metas,
            index_text_fn=make_table_index_text_fn(doc_metas),
        )

    reranker = rerank_fn
    if reranker is None and own_embedder and chroma_path is not None:
        from akr.storage.rerank import build_reranker

        reranker = build_reranker(usage_path=rerank_usage_path)

    return ToolContext(
        conn=conn,
        vectors=vectors,
        bm25=bm25,
        graph=graph,
        artifacts=artifacts,
        doc_meta=doc_metas,
        reranker=reranker,
    )


def create_server(ctx: ToolContext) -> "Server":
    """薄适配层：把 ``ctx`` 闭包进 MCP handler，不在这里做任何业务判断。"""

    from mcp import types
    from mcp.server import Server

    async def _list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[types.Tool(**schema) for schema in list_tools_handler()]
        )

    async def _call_tool(
        _ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        text = call_tool_handler(ctx, params.name, params.arguments or {})
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    return Server(SERVER_NAME, on_list_tools=_list_tools, on_call_tool=_call_tool)


async def _serve(ctx: ToolContext) -> None:
    from mcp.server.stdio import stdio_server

    server = create_server(ctx)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """以 stdio 启动：``python -m akr.mcp_server.server``。"""

    ctx = build_tool_context(
        db_path=_env_path("AKR_DB_PATH", str(DEFAULT_DB_PATH)) or DEFAULT_DB_PATH,
        artifacts_root=_env_path("AKR_ARTIFACTS_ROOT", str(DEFAULT_ARTIFACTS_ROOT))
        or DEFAULT_ARTIFACTS_ROOT,
        chroma_path=_env_path("AKR_CHROMA_PATH", str(DEFAULT_CHROMA_PATH)),
        unit_map_path=_env_path("AKR_UNIT_MAP_PATH", str(DEFAULT_UNIT_MAP_PATH)),
        bm25_path=_env_path("AKR_BM25_PATH", str(DEFAULT_BM25_PATH)),
        manifest_path=_env_path("AKR_MANIFEST_PATH", str(DEFAULT_MANIFEST_PATH)),
        neo4j_uri=os.environ.get("NEO4J_URI"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "akr-local-pass"),
        vector_doc_prefix=_env_flag("AKR_VECTOR_DOC_PREFIX"),
        vector_fold=os.environ.get("AKR_VECTOR_FOLD", DEFAULT_VECTOR_FOLD),
        # 重排的**模型名**由 `akr.config.RerankSettings` 读 `SILICONFLOW_RERANK_MODEL`（与嵌入
        # 的模型名同一个位置、同一把 Key）；这里只多给一个可选的账本路径，用来记「重排花了
        # 多少 token」——它和 `AKR_*` 那组是同一类东西（本机路径），不是服务配置。
        rerank_usage_path=_env_optional_path("AKR_RERANK_USAGE_PATH"),
    )
    try:
        asyncio.run(_serve(ctx))
    finally:
        if ctx.graph is not None:
            ctx.graph.close()
        ctx.conn.close()


if __name__ == "__main__":  # pragma: no cover - 进程入口
    main()
