r"""线上工具层与离线预测的逐题对账（零 chat 调用）。

`scripts/retrieval_ab.py` 的 25 条题是**离线重放**：它在内存里重建一份 BM25、自己拼 RRF，
回答的是「如果线上这么算，能多进多少条 gold」。而对外要汇报的是**线上改动**的收益，
中间还缺一步：线上的 `search_documents` 是不是真的这么算。本脚本补的就是这一步。

做法：用**真实 ToolContext**（真实 Chroma 向量库 + 真实 ``data/bm25.json`` + 从
``data/manifest.json`` 派生的文档身份表），对离线表里的同一批问句各调一次工具，
把「gold 的最好名次」逐题与离线表对比。对照的是离线表自己记下的问句，不另读任务集——
少一个输入就少一处口径漂移。

为什么要成对验证：身份前缀与查询收窄是**两次改动**，只上前缀会让 `stage2@8/M3`
从 10/25 掉到 9/25，成对上线才是 12/25。所以这里对账的配置默认就是成对的那一档。

花费：只有嵌入（每次查询一个短文本），**不调用 chat**，也不写任何状态。

用法（仓库根目录；先跑一次 `retrieval_ab.py --json data/retrieval-ab.json`）：

    .\.venv\Scripts\python.exe scripts\retrieval_tool_smoke.py
    # 想把对账结论留成可引用的证据（「线上逐题一致」就有出处了）：
    .\.venv\Scripts\python.exe scripts\retrieval_tool_smoke.py --json data\retrieval-tool-smoke.json

退出码：0 = 逐题一致；1 = 有差异或前置文件缺失。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 同目录脚本互引：`rank_of` 是离线重放的口径，对账必须用**同一份**实现，
# 复制一份就等着两边各自演化。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from akr.ingest.doc_meta import load_doc_meta  # noqa: E402
from akr.ingest.table_context import make_table_index_text_fn  # noqa: E402
from akr.mcp_server.server import (  # noqa: E402
    DEFAULT_CHROMA_PATH,
    DEFAULT_UNIT_MAP_PATH,
    DEFAULT_VECTOR_FOLD,
)
from akr.mcp_server.tools import (  # noqa: E402
    EXPAND_NEIGHBORS,
    POOL_DEPTH_DEFAULT,
    RERANK_DEFAULT,
    RERANK_LOCK_DEFAULT,
    RERANK_MODES,
    RRF_K_DEFAULT,
    ToolContext,
    search_documents,
)
from akr.storage import sqlite_store  # noqa: E402
from akr.storage.bm25_index import Bm25Index  # noqa: E402
from akr.storage.embeddings import build_embedder  # noqa: E402
from akr.storage.rerank import build_reranker  # noqa: E402
from akr.storage.vector_store import FOLD_MODES, VectorStore, load_id_map  # noqa: E402
from retrieval_ab import AbError, rank_of  # noqa: E402

# 对账的那份向量库必须与**线上装配**指的是同一份：线上 2026-09-26 已切到 unit 级的
# `data/chroma-v3`，这里跟着常量走，不另抄一份路径（`data/chroma` 是历史基准，另行显式传）。
DEFAULT_CHROMA = Path(DEFAULT_CHROMA_PATH)
DEFAULT_UNIT_MAP = Path(DEFAULT_UNIT_MAP_PATH)
DEFAULT_FOLD = DEFAULT_VECTOR_FOLD
DEFAULT_BM25 = Path("data/bm25.json")
DEFAULT_MANIFEST = Path("data/manifest.json")
DEFAULT_AB = Path("data/retrieval-ab.json")
# 邻块只能从 SQLite 的 `chunks` 表按 ordinal 取，所以对账脚本必须指向**线上同一个**
# sqlite 文件（`akr.mcp_server.server.DEFAULT_DB_PATH`，用例会断言相等）。
# 指向别的库＝拿另一套 ordinal 扩窗，对账就变成对两件事。
DEFAULT_DB = Path("data/akr.sqlite")
# 线上首轮检索的真实参数（`akr.orchestration.runner`：查询=原问句、TWO_STAGE_DOCS=3，
# worker 的 top_k 默认 8）。改这里等于改对账口径，必须同步改 runner。
DEFAULT_CONFIG = "stage2@8/M3+meta+narrow"
DEFAULT_TOP_K = 8
DEFAULT_TWO_STAGE_DOCS = 3


def mismatches(
    expected: dict[str, int | None],
    actual: dict[str, int | None],
) -> list[tuple[str, int | None, int | None]]:
    """逐题对账，返回 (task_id, 离线名次, 线上名次) 的差异列表。

    只出现在一边的题也算差异：少算一道题比名次差一名更可疑——那说明两边看的不是同一批题。
    """

    keys = sorted(set(expected) | set(actual))
    return [
        (key, expected.get(key), actual.get(key))
        for key in keys
        if expected.get(key) != actual.get(key)
    ]


def hit_at(ranks: dict[str, int | None], depth: int) -> int:
    """名次表 → 进前 `depth` 的**题数**（离线表与主表共用的命中口径）。"""

    return sum(1 for rank in ranks.values() if rank is not None and rank <= depth)


def evidence(
    *,
    config: str,
    top_k: int,
    two_stage_docs: int,
    expected: dict[str, int | None],
    actual: dict[str, int | None],
    pool_depth: int = POOL_DEPTH_DEFAULT,
    rerank: str = RERANK_DEFAULT,
    rrf_k: int = RRF_K_DEFAULT,
    rerank_lock: int = RERANK_LOCK_DEFAULT,
    expand_neighbors: int = EXPAND_NEIGHBORS,
    fold: str = DEFAULT_FOLD,
) -> dict[str, Any]:
    """对账结论的落盘形状。

    存在的理由：终端里的「逐题一致」明天就找不到了，而对外那句「线上与离线预测逐题一致」
    必须能被翻出来核对。所以把**名次本身**（不是结论）落盘：结论是名次的函数，谁都能复算。

    检索旋钮与**向量库口径**也要落盘：同一份名次表在 `pool_depth=0` / `=24` 下不是一件事，
    在「块级库 + 只留最好名次」与「unit 级库 + 同父块累加」下更不是一件事——少了这几个数，
    名次表就复算不出来。键名分两代：`pool_depth` / `rerank` / `expand_neighbors` 是检索侧，
    `fold` 是向量侧的折叠口径（`akr.storage.vector_store.FOLD_MODES`）。
    `rrf_k` / `rerank_lock` 是 2026-09-26 第二次实测新加的旋钮，也必须落盘：融合分是
    `1/(rrf_k + 名次)`，同一份候选在 `rrf_k=20` 与 `=60` 下不是同一个顺序。
    """

    diff = mismatches(expected, actual)
    return {
        "config": config,
        "top_k": top_k,
        "two_stage_docs": two_stage_docs,
        "pool_depth": pool_depth,
        "rerank": rerank,
        "rrf_k": rrf_k,
        "rerank_lock": rerank_lock,
        "expand_neighbors": expand_neighbors,
        "fold": fold,
        "total": len(expected),
        "offline_hits": hit_at(expected, top_k),
        "online_hits": hit_at(actual, top_k),
        "diffs": [
            {"task_id": task_id, "offline": offline_rank, "online": online_rank}
            for task_id, offline_rank, online_rank in diff
        ],
        "ranks": {"offline": expected, "online": actual},
    }


def build_context(
    *,
    chroma_path: Path,
    bm25_path: Path,
    manifest_path: Path,
    db_path: Path = DEFAULT_DB,
    unit_map_path: Path | None = DEFAULT_UNIT_MAP,
    fold: str = DEFAULT_FOLD,
    reranker: Any = None,
) -> ToolContext:
    """装配工具所需的依赖，身份表的装法与 `akr.mcp_server.server` 逐字相同。

    刻意不复用 `build_tool_context`：它硬要求 Neo4j（`walk_graph` 的依赖），而
    `search_documents` 一个图查询也不发。为了对账一个检索改动去起一个图数据库，
    只会把「跑不起来」当成「对账失败」。

    `conn` 用真 SQLite：邻块扩窗要按 `ordinal` 读 `chunks` 表，假连接会假绿。
    默认关掉扩窗（`expand_neighbors=0`）时这条连接一次也不查，历史对账结果不受影响。

    `unit_map_path` / `fold` 与 `build_tool_context` 的默认值同源（都是 `server.DEFAULT_*`）：
    线上那份库是 unit 级索引，少了映射或折叠口径，返回的 id 就落在 gold 空间之外，
    对账会以「线上命中 0/25」的形状报出来——那是装配错了，不是检索变差了。

    `reranker` 只在 `--rerank cross` 时非空；对账默认的 `rrf` 那一档一行都不碰它，
    所以默认跑法仍然是零重排 token 的。
    """

    if fold not in FOLD_MODES:
        raise AbError(f"未知的折叠口径：{fold!r}；可选 {sorted(FOLD_MODES)}")
    id_map = None
    if unit_map_path is not None:
        if not unit_map_path.exists():
            raise AbError(
                f"缺 unit → 父块 映射：{unit_map_path}（先跑 scripts\\build_unit_chunks.py）"
            )
        id_map = load_id_map(unit_map_path)
    metas = load_doc_meta(manifest_path)
    return ToolContext(
        conn=sqlite_store.connect(db_path),
        vectors=VectorStore(chroma_path, build_embedder(), id_map=id_map, fold=fold),
        # BM25 侧与线上同一份索引文本口径（表格导读行）：它是装配层旋钮、不进工具层参数，
        # 少了这一步，「线上 vs 离线逐题一致」这条对账对的就是两份不同的打分矩阵。
        bm25=Bm25Index.load(
            bm25_path, doc_metas=metas, index_text_fn=make_table_index_text_fn(metas)
        ),
        graph=None,
        artifacts=None,
        doc_meta=metas,
        reranker=reranker,
    )


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["summary"], list(payload["rows"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="线上工具层 vs 离线检索 A/B 的逐题对账")
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument(
        "--unit-map",
        type=Path,
        default=DEFAULT_UNIT_MAP,
        help="unit → 父块 id 映射（线上那份 unit 级库的折回依据）",
    )
    parser.add_argument(
        "--fold",
        choices=sorted(FOLD_MODES),
        default=DEFAULT_FOLD,
        help=f"id_map 折叠口径；默认 {DEFAULT_FOLD!r} = 线上口径",
    )
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite（邻块扩窗的唯一来源）")
    parser.add_argument("--ab", type=Path, default=DEFAULT_AB, help="retrieval_ab.py --json 的产物")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="要对照的离线配置名")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--two-stage-docs", type=int, default=DEFAULT_TWO_STAGE_DOCS)
    parser.add_argument("--pool-depth", type=int, default=POOL_DEPTH_DEFAULT)
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K_DEFAULT,
        help="第一段融合的 RRF 常数 k（默认取工具层口径，不另抄一份）",
    )
    parser.add_argument(
        "--rerank-lock",
        type=int,
        default=RERANK_LOCK_DEFAULT,
        help="rerank=cross 时由 RRF 独占的前 N 条（0 = 整池重排）",
    )
    parser.add_argument(
        "--rerank",
        choices=sorted(RERANK_MODES),
        default=RERANK_DEFAULT,
        help="融合后的排序模式（白名单从工具层 import，不另抄一份）",
    )
    parser.add_argument(
        "--rerank-usage",
        type=Path,
        default=None,
        help="rerank=cross 时的重排用量账本（JSON）；默认 None = 不落盘",
    )
    parser.add_argument("--expand-neighbors", type=int, default=EXPAND_NEIGHBORS)
    parser.add_argument(
        "--json", type=Path, default=None, help="把对账证据（名次表 + 差异列表）落成 JSON"
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    # SQLite 也要先查存在性：`sqlite3.connect` 对不存在的路径**会建一个空库**，
    # 那既是对生产目录的写入，又会让后面的扩窗静默查到 0 条邻块。
    for path in (args.bm25, args.manifest, args.db, args.ab):
        if not path.exists():
            raise AbError(
                f"缺文件：{path}"
                "（先跑 scripts/ingest_corpus.py 与 scripts/retrieval_ab.py --json data/retrieval-ab.json）"
            )
    if not args.chroma.exists():
        raise AbError(
            f"缺向量库目录：{args.chroma}"
            "（线上那份 unit 级库由 scripts/ingest_corpus.py + scripts/build_unit_chunks.py 产出）"
        )

    summary, rows = load_rows(args.ab)
    if args.config not in summary:
        raise AbError(f"{args.ab} 里没有配置 {args.config!r}；有：{sorted(summary)}")
    # `cross` 才建重排器（唯一会发网络请求的一档）：模型与 Key 走
    # `SILICONFLOW_RERANK_MODEL`（`akr.config.RerankSettings`），与嵌入共用同一个网关。
    reranker = build_reranker(usage_path=args.rerank_usage) if args.rerank == "cross" else None
    ctx = build_context(
        chroma_path=args.chroma,
        bm25_path=args.bm25,
        manifest_path=args.manifest,
        db_path=args.db,
        unit_map_path=args.unit_map,
        fold=args.fold,
        reranker=reranker,
    )
    print(
        f"[load] 离线表 {len(rows)} 行、文档身份 {len(ctx.doc_meta)} 份、"
        f"Chroma 计数 {ctx.vectors.count()}（{args.chroma}，折叠口径 {args.fold}）"
    )
    print(
        f"[knobs] pool_depth={args.pool_depth} rerank={args.rerank} "
        f"rrf_k={args.rrf_k} rerank_lock={args.rerank_lock} "
        f"two_stage_docs={args.two_stage_docs} expand_neighbors={args.expand_neighbors}"
    )
    if reranker is not None:
        print(f"[rerank] cross-encoder 已装载，用量账本={args.rerank_usage}")

    expected = {row["task_id"]: row["ranks"][args.config] for row in rows}
    actual: dict[str, int | None] = {}
    for row in rows:
        task_id = row["task_id"]
        result = search_documents(
            ctx,
            str(row["question"]),
            top_k=args.top_k,
            two_stage_docs=args.two_stage_docs,
            pool_depth=args.pool_depth,
            rerank=args.rerank,
            rrf_k=args.rrf_k,
            rerank_lock=args.rerank_lock,
            expand_neighbors=args.expand_neighbors,
        )
        actual[task_id] = rank_of(list(row["gold_chunk_ids"]), result.hits)
        print(f"[{task_id:>14}] 离线={expected[task_id]} 线上={actual[task_id]}")

    diff = mismatches(expected, actual)
    print(
        f"\n[summary] 配置 {args.config}：离线 top-{args.top_k} 命中 "
        f"{hit_at(expected, args.top_k)}/{len(expected)}、线上 "
        f"{hit_at(actual, args.top_k)}/{len(actual)}；逐题差异 {len(diff)} 处"
    )
    for task_id, expected_rank, actual_rank in diff:
        print(f"  [diff] {task_id}: 离线={expected_rank} 线上={actual_rank}")

    if args.json:
        payload = evidence(
            config=args.config,
            top_k=args.top_k,
            two_stage_docs=args.two_stage_docs,
            pool_depth=args.pool_depth,
            rerank=args.rerank,
            rrf_k=args.rrf_k,
            rerank_lock=args.rerank_lock,
            expand_neighbors=args.expand_neighbors,
            fold=args.fold,
            expected=expected,
            actual=actual,
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[write] {args.json}")
    print(f"[done] 耗时 {time.perf_counter() - started:.1f}s")
    return 1 if diff else 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    sys.exit(main())
