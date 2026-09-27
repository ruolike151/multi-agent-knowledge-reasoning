r"""M1 验收冒烟：真实语料 PDF + 真实嵌入，把两条 M1 验收标准跑成可复现命令。

用法（仓库根目录，先确保 Neo4j 容器在跑）：

    .\.venv\Scripts\python.exe scripts\m1_smoke.py                    # 硅基流动 API 嵌入
    .\.venv\Scripts\python.exe scripts\m1_smoke.py --pages 60 --keep
    .\.venv\Scripts\python.exe scripts\m1_smoke.py --embedder local   # 离线兜底路径

验收点（spec §13 / 计划 README 的 Phase 2 说明）：

1. 给定问题能返回带 `chunk_id` 的检索结果（向量 + BM25 两路 → RRF 融合）。
2. 改动一份文档中 1 页后，只有受影响的 chunk 被重新向量化。

嵌入默认走硅基流动 `BAAI/bge-large-zh-v1.5`（读 `.env` 的 `SILICONFLOW_API_KEY`）；
`--embedder local` 切到本地 `bge-small-zh-v1.5`（需 `.[embed]` 依赖，完全离线）。
两条路径都不写 DeepSeek 的 ¥5 成本表——DeepSeek 只负责 chat，本链路不调用它。
实体抽取（Task 18）显式跳过（`client=None`）——M1 验收范围只是「解析 → 分块 → 三库 → 检索」这条链路。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from neo4j import GraphDatabase

from akr.ingest.incremental import ingest_document
from akr.ingest.parser import parse_pdf
from akr.mcp_server.tools import ToolContext, fetch_chunk, list_documents, search_documents
from akr.storage import sqlite_store
from akr.storage.artifact_store import ArtifactStore
from akr.storage.bm25_index import Bm25Index
from akr.storage.embeddings import build_embedder, build_local_embedder
from akr.storage.graph_store import GraphStore
from akr.storage.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
ANNUAL = ROOT / "data" / "corpus" / "annual" / "600438_通威股份_2025年报.pdf"
DOC_ID = "annual-600438-2025"
QUERY = "公司 2025 年的营业收入和归母净利润是多少"
# 增量验收只允许重嵌「被改到的那一页所覆盖的 chunk」；留一点余量给跨页切分
MAX_REEMBED_RATIO = 0.15


def _pick_embedder(kind: str):
    """`api`（默认）走硅基流动；`local` 走本地 bge-small-zh-v1.5（完全离线）。"""
    return build_local_embedder() if kind == "local" else build_embedder()


def build_context(workspace: Path, embedder: str = "api") -> tuple[ToolContext, object]:
    conn = sqlite_store.connect(workspace / "akr.sqlite")
    sqlite_store.init_schema(conn)
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "akr-local-pass"),
        ),
    )
    driver.verify_connectivity()
    graph = GraphStore(driver)
    graph.init_schema()
    ctx = ToolContext(
        conn=conn,
        vectors=VectorStore(workspace / "chroma", _pick_embedder(embedder)),
        bm25=Bm25Index(),
        graph=graph,
        artifacts=ArtifactStore(workspace / "artifacts"),
    )
    return ctx, driver


def _ingest(ctx: ToolContext, pages, source: str):
    return ingest_document(
        doc_id=DOC_ID,
        source=source,
        pages=pages,
        conn=ctx.conn,
        artifacts=ctx.artifacts,
        vectors=ctx.vectors,
        bm25=ctx.bm25,
        graph=ctx.graph,
        client=None,  # 跳过实体抽取：M1 不验收图链路
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M1 验收冒烟（真实 PDF + 真实嵌入）")
    parser.add_argument("--pages", type=int, default=40, help="只取年报前 N 页，默认 40")
    parser.add_argument(
        "--embedder",
        choices=("api", "local"),
        default="api",
        help="api=硅基流动 BAAI/bge-large-zh-v1.5（默认）；local=本地 bge-small-zh-v1.5",
    )
    parser.add_argument("--keep", action="store_true", help="保留工作目录（默认跑完删除）")
    args = parser.parse_args(argv)

    if not ANNUAL.exists():
        print(f"[FAIL] 语料不存在：{ANNUAL}（先跑 scripts/fetch_corpus.py）", file=sys.stderr)
        return 1

    workspace = Path(tempfile.mkdtemp(prefix="akr-m1-smoke-"))
    print(f"[i] 工作目录：{workspace}")
    print(f"[i] 嵌入后端：{args.embedder}")
    ctx, driver = build_context(workspace, args.embedder)
    source = str(ANNUAL.relative_to(ROOT))
    try:
        print(f"[i] 解析 {ANNUAL.name} 前 {args.pages} 页 …")
        # parse_pdf 返回的 ParsedPage 页码即原文页码；只取前 N 页做冒烟，页码保持真实
        pages = parse_pdf(ANNUAL)[: args.pages]
        first = _ingest(ctx, pages, source)
        print(
            f"[1] 首次入库：chunk {first.total_chunks} 个（新增 {first.added} / 重嵌 {first.embedded}），"
            f"耗时 {first.elapsed_ms:.0f} ms"
        )

        # ---- 验收点 1：带 chunk_id 的检索结果 ----
        # M1 冒烟不装配重排器（它只验「入库 → 检索到带 chunk_id 的片段」这条链路），
        # 所以显式走 `rrf`：工具层的线上默认是 `rerank="cross"`，缺 `ctx.reranker` 会当场报错，
        # 那是有意的（不静默退回 rrf）。
        result = search_documents(ctx, QUERY, top_k=5, rerank="rrf")
        if not result.hits:
            print("[FAIL] 检索没有返回任何结果", file=sys.stderr)
            return 1
        print(f"[2] 检索「{QUERY}」→ {len(result.hits)} 条（{result.retriever_stats}）")
        for rank, hit in enumerate(result.hits, start=1):
            snippet = hit.text.replace("\n", " ")[:60]
            print(f"    #{rank} {hit.chunk_id} score={hit.score:.5f} [{hit.retriever}] {snippet}")
        if any(not hit.chunk_id.startswith(f"{DOC_ID}#") for hit in result.hits):
            print("[FAIL] 检索结果的 chunk_id 与 doc_id 不符", file=sys.stderr)
            return 1
        # fetch_chunk 必须能按 chunk_id 取回正文 → 溯源闭环
        back = fetch_chunk(ctx, result.hits[0].chunk_id)
        if back.text != result.hits[0].text:
            print("[FAIL] fetch_chunk 取回的正文与检索结果不一致", file=sys.stderr)
            return 1
        print(f"[3] fetch_chunk 溯源闭环 OK（{len(back.text)} 字，source={back.source}）")

        # ---- 验收点 2：只重嵌受影响的 chunk ----
        edited = list(pages)
        target = edited[len(edited) // 2]
        edited[len(edited) // 2] = type(target)(
            page=target.page,
            text=target.text + "\n报告期内公司新增一项重大在建工程，预算金额 12.34 亿元。",
            tables=target.tables,
        )
        second = _ingest(ctx, edited, source)
        ratio = second.reembed_ratio
        print(
            f"[4] 改动第 {target.page} 页后重新入库：chunk {second.total_chunks} 个"
            f"（新增 {second.added} / 删除 {second.removed} / 复用 {second.unchanged}），"
            f"重新嵌入比例 {ratio:.2%}，耗时 {second.elapsed_ms:.0f} ms"
        )
        if second.unchanged <= 0:
            print("[FAIL] 没有任何 chunk 被复用，说明差量计算退化成全量", file=sys.stderr)
            return 1
        if ratio > MAX_REEMBED_RATIO:
            print(f"[FAIL] 重新嵌入比例 {ratio:.2%} 超过上限 {MAX_REEMBED_RATIO:.0%}", file=sys.stderr)
            return 1

        overview = list_documents(ctx)
        for item in overview.documents:
            print(f"[5] 知识库文档：{item.doc_id} chunk={item.chunk_count} source={item.source}")

        print("[PASS] M1 冒烟通过：检索可溯源 + 改动 1 页只重嵌受影响的 chunk")
        if args.keep:
            print(f"[i] 已保留工作目录：{workspace}")
        return 0
    finally:
        driver.close()
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
