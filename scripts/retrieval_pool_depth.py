r"""第一段融合池深度 → top-8 命中的曲线（真实工具层，零 chat 调用）。

**为什么要单独量这条曲线。** 工具层 `search_documents` 里有一个深度选择
`depth = TWO_STAGE_PICK_DEPTH if two_stage else top_k`，而 `two_stage` 的条件带着
`doc_ids is None`——25 道可答题**全部**命中「问句抽码收窄」，`doc_ids` 因此恒非空，
二段分支一次都没跑过（`retriever_stats` 里没有 `stage2_docs` 键）。真正还在动的旋钮
只剩第一段：`rrf_fuse` 拿几个候选去融。

**为什么能用真实工具量「没上线」的配置。** `rrf_fuse` 的 `top_k` 只做截断，所以
`search_documents(top_k=D)` 返回的前 8 条，**逐条等于**「池子取 D、只把前 8 条交给模型」
会返回的东西。于是不必先改生产代码就能量出改动后的返回形状——这也让「离线预测」与
「线上工具层」共用同一条代码路径，少一层口径漂移。

**为什么池深会改变 top-8。** RRF 给「两路都命中」的条目叠加分数，池子越深，越靠后的
共识条目也有机会累到分。池子只有 8 时，某条 gold 只要不在这两路的各前 8 名里就直接出局。

**花费**：只有嵌入（每问一句短文本），**不调用 chat**，不写任何状态。

用法（仓库根目录；`--chroma-v2` 是 `scripts/rebuild_vector_index.py --doc-prefix` 那份库）：

    # v1 基准 + **线上那份**（v3，默认取 `server.DEFAULT_CHROMA_PATH`）、线上折叠口径
    .\.venv\Scripts\python.exe scripts\retrieval_pool_depth.py
    # 复算第三十七节那张表（折叠口径改用「只留最好名次」）
    .\.venv\Scripts\python.exe scripts\retrieval_pool_depth.py --fold best
    .\.venv\Scripts\python.exe scripts\retrieval_pool_depth.py --chroma-v2 data\chroma-v2
    .\.venv\Scripts\python.exe scripts\retrieval_pool_depth.py --chroma-v2 data\chroma-v2 ^
        --json data\pool-depth.json
    # cross-encoder 重排那一格（**唯一会花 token** 的档：每次检索一次 /v1/rerank；
    # 模型 `BAAI/bge-reranker-v2-m3` 在硅基流动免费，但仍要落账本才能报用量）
    .\.venv\Scripts\python.exe scripts\retrieval_pool_depth.py --rerank cross ^
        --pool-depth 12 --rerank-usage tmp\rerank-usage.json --json tmp\pool-depth-cross.json

退出码：0 = 正常；1 = 前置文件缺失（`AbError`）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 同目录脚本互引：`rank_of` / `recall_at` / `AbError` 都是离线重放的口径，
# 曲线要对齐离线表就必须用**同一份**实现。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from akr.ingest.doc_meta import load_doc_meta  # noqa: E402
from akr.ingest.table_context import make_table_index_text_fn  # noqa: E402
from akr.mcp_server.server import (  # noqa: E402
    DEFAULT_CHROMA_PATH,
    DEFAULT_UNIT_MAP_PATH,
    DEFAULT_VECTOR_FOLD,
)
from akr.mcp_server.tools import (  # noqa: E402
    RERANK_DEFAULT,
    RERANK_MODES,
    ToolContext,
    search_documents,
)
from akr.storage.bm25_index import Bm25Index  # noqa: E402
from akr.storage.embeddings import build_embedder  # noqa: E402
from akr.storage.fusion import DEFAULT_ALPHA  # noqa: E402
from akr.storage.rerank import build_reranker  # noqa: E402
from akr.storage.vector_store import FOLD_MODES, VectorStore  # noqa: E402
from build_unit_chunks import load_unit_map  # noqa: E402
from retrieval_ab import AbError, rank_of, recall_at  # noqa: E402

# 第一列刻意钉死在 v1（`data/chroma`）的字面量上，**不跟** `server.DEFAULT_CHROMA_PATH` 漂：
# 线上 2026-09-26 已切到 v3（unit 级），但这一列是第三十四~三十七节所有横向对照的**基准**，
# 跟着线上漂过去，历史表就没法逐格复算了。线上那份（v3）是 `--chroma-v3` 的默认值。
DEFAULT_CHROMA = Path("data/chroma")
DEFAULT_BM25 = Path("data/bm25.json")
DEFAULT_MANIFEST = Path("data/manifest.json")
DEFAULT_AB = Path("data/retrieval-ab.json")
# v3 是 **unit 级**索引（`scripts/build_unit_chunks.py`）：查出来的 id 是 unit id，必须靠这份
# 映射折回父块，返回的 `chunk_id` 才落在 gold（父块 id）空间里。
DEFAULT_UNIT_MAP = Path(DEFAULT_UNIT_MAP_PATH)
# 线上装配的折叠口径（`akr.mcp_server.server.DEFAULT_VECTOR_FOLD`）。
#   * 默认 = 线上口径（同一父块的多条 unit 累加），这样这一列量的是**线上真实返回**；
#   * `--fold best` = 第三十七节那张表的口径（同一父块只留名次最好的 unit），用来复算历史。
DEFAULT_FOLD = DEFAULT_VECTOR_FOLD
# 线上 worker 的真实返回条数：池子可以变深，交给模型看的仍然是 8 条。
DEFAULT_TOP_K = 8
DEFAULT_DEPTHS = (8, 12, 16, 24, 32, 50)


def mrr_at(ranks: dict[str, int | None]) -> float:
    """MRR：每道题最好名次取倒数、对全部题求均值（MISS 记 0）。

    `ranks` 已经在返回的 `top_k` 条里算过，所以这里不再截断——名次口径只有一处。
    """

    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks.values() if rank) / len(ranks)


def metrics(
    rows: list[dict[str, Any]],
    hits_by_task: dict[str, list[Any]],
    top_k: int,
) -> dict[str, Any]:
    """把「每题返回的命中列表」换算成三个口径 + 名次表。

    三个口径缺一不可：命中率的分母是**题数**，标准 recall 的分母是 **gold 条数**，
    MRR 看的是名次。只报一个「命中率」，换个分母就能被问倒。
    """

    ranks: dict[str, int | None] = {}
    recalls: dict[str, float] = {}
    for row in rows:
        gold = list(row["gold_chunk_ids"])
        hits = hits_by_task[row["task_id"]]
        ranks[row["task_id"]] = rank_of(gold, hits[:top_k])
        recalls[row["task_id"]] = round(recall_at(gold, hits, top_k), 4)
    total = len(rows)
    return {
        "top_k": top_k,
        "total": total,
        "hit": sum(1 for rank in ranks.values() if rank is not None),
        "hit_rate": round(sum(1 for r in ranks.values() if r) / total, 4) if total else 0.0,
        "recall": round(sum(recalls.values()) / total, 4) if total else 0.0,
        "mrr": round(mrr_at(ranks), 4),
        "ranks": ranks,
    }


def describe(store: str, depth: int, summary: dict[str, Any]) -> str:
    """终端里一行看懂：池深换了，命中/recall/MRR 各是多少。"""

    return (
        f"[{store} 池深={depth:>2}] top-{summary['top_k']} 命中 "
        f"{summary['hit']:>2}/{summary['total']}（{summary['hit_rate']:.0%}）· "
        f"recall@{summary['top_k']} {summary['recall']:.1%} · MRR {summary['mrr']:.4f}"
    )


def measure_vector_only(
    ctx: ToolContext,
    rows: list[dict[str, Any]],
    depths: tuple[int, ...],
) -> dict[str, Any]:
    """纯向量侧：不经融合、不经问句收窄，直接看嵌入能把 gold 排到第几名。

    与 `measure` 的差别只有取数路径：这里 `top_k=depth` 就是**返回条数**，所以
    「recall@50」是真正的深池召回；`measure` 那一列返回给模型的恒为 top-8，
    两个口径不能互相顶替——深召回好不代表进得了 top-8，反之亦然。

    这一列**只对向量侧**有意义：BM25 侧没有「嵌入改没改好」这回事，所以不在这里量。
    """

    out: dict[str, Any] = {}
    for depth in depths:
        hits_by_task = {
            row["task_id"]: list(ctx.vectors.query(str(row["question"]), top_k=depth))
            for row in rows
        }
        out[str(depth)] = metrics(rows, hits_by_task, depth)
    return out


def describe_vector(store: str, depth: int, summary: dict[str, Any]) -> str:
    """纯向量那一列的终端行：返回条数就是池深，所以标题写 `@depth` 而不是 `top-8`。"""

    return (
        f"[{store} 纯向量@{depth:>2}] 命中 {summary['hit']:>2}/{summary['total']}"
        f"（{summary['hit_rate']:.0%}）· recall@{depth} {summary['recall']:.1%} · "
        f"MRR {summary['mrr']:.4f}"
    )


def measure(
    ctx: ToolContext,
    rows: list[dict[str, Any]],
    depths: tuple[int, ...],
    *,
    rerank: str = RERANK_DEFAULT,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """每个池深各调一次 `search_documents`，返回 {深度字符串: 指标}。

    池深与重排都走**工具层的真实入参**（`pool_depth` / `rerank` / `rerank_alpha`），返回条数
    固定为 `DEFAULT_TOP_K`：脚本与线上从此跑的是同一段代码。（2026-09-25 之前这里是用
    `top_k=depth` 模拟深池——数字等价，但等价性要靠「`rrf_fuse` 的 `top_k` 只做截断」推导一步。）

    刻意**不缓存**命中：池深是工具层的入参，缓存只会把「工具在池深 D 下返回了什么」
    换成「脚本以为它会返回什么」。

    `rerank="cross"` 时每次检索多一次 `/v1/rerank`（`ctx.reranker` 由 `main` 按需建）：
    代价按**题数 × 层数**计（一次检索一条问句 + 一整池候选），所以默认仍是零 token 的 `rrf`。
    """

    out: dict[str, Any] = {}
    for depth in depths:
        hits_by_task = {
            row["task_id"]: list(
                search_documents(
                    ctx,
                    str(row["question"]),
                    top_k=DEFAULT_TOP_K,
                    pool_depth=depth,
                    rerank=rerank,
                    rerank_alpha=alpha,
                ).hits
            )
            for row in rows
        }
        out[str(depth)] = metrics(rows, hits_by_task, DEFAULT_TOP_K)
    return out


def build_context(
    *,
    chroma_path: Path,
    bm25_path: Path,
    manifest_path: Path,
    id_map: dict[str, str] | None = None,
    fold: str = DEFAULT_FOLD,
    reranker: Any = None,
) -> ToolContext:
    """装配工具所需的依赖，装法与 `akr.mcp_server.server` 逐字相同。

    `id_map` 只对 unit 级库（v3）非空：它让 `VectorStore.query` 把 unit 名次折回父块，
    于是 v3 与 v1/v2 在**同一个 gold 空间**里比对——否则 v3 那一列会全是对不上 gold 的
    unit id，看起来像「召回率归零」。

    `fold` 默认取线上口径：只带 `id_map` 不给口径，装出来的就是「块级」的读法（同一父块
    只留名次最好的 unit），量出来的不是线上真正返回的那个顺序。

    BM25 侧也按线上装配带上**表格导读行**（`make_table_index_text_fn`）：它是装配层旋钮，
    不进工具层参数，脚本少了这一步，量出来的就是「另一份索引文本打出来的分」。

    `reranker` 只在 `--rerank cross` 时非空（`main` 里按需建，见那一行的账本说明）。
    """

    metas = load_doc_meta(manifest_path)
    return ToolContext(
        conn=None,
        vectors=VectorStore(chroma_path, build_embedder(), id_map=id_map, fold=fold),
        bm25=Bm25Index.load(
            bm25_path, doc_metas=metas, index_text_fn=make_table_index_text_fn(metas)
        ),
        graph=None,
        artifacts=None,
        doc_meta=metas,
        reranker=reranker,
    )


def resolve_stores(
    chroma: Path, chroma_v2: Path | None, chroma_v3: Path | None
) -> list[tuple[str, Path]]:
    """要量哪几个向量库：v1（线上在用的那份）永远量，v2/v3 只有显式给了路径才加。"""

    stores: list[tuple[str, Path]] = [("v1", chroma)]
    if chroma_v2 is not None:
        stores.append(("v2", chroma_v2))
    if chroma_v3 is not None:
        stores.append(("v3", chroma_v3))
    return stores


def resolve_id_maps(
    stores: list[tuple[str, Path]], unit_map_path: Path
) -> dict[str, dict[str, str]]:
    """哪几个库要折叠：只有 v3 是 unit 级索引，所以只有它需要映射（缺了就报错，不静默）。"""

    if not any(name == "v3" for name, _ in stores):
        return {}
    if not unit_map_path.exists():
        raise AbError(f"缺 unit → 父块 映射：{unit_map_path}（先跑 scripts\\build_unit_chunks.py）")
    return {"v3": load_unit_map(unit_map_path)}


def parse_depths(text: str) -> tuple[int, ...]:
    """`8,16,24` → (8, 16, 24)；去重且升序，免得同一档被量两遍。"""

    depths: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise AbError(f"池深必须是正整数，收到 {value}")
        if value not in depths:
            depths.append(value)
    if not depths:
        raise AbError("--depths 不能为空")
    return tuple(sorted(depths))


def resolve_depths(pool_depth: int | None, depths_text: str) -> tuple[int, ...]:
    """`--pool-depth` 给了就只量这一档，否则按 `--depths` 量整条曲线。

    「只量一档」是对账线上要上线的那个值用的：曲线看趋势，单点看结论。
    """

    if pool_depth is None:
        return parse_depths(depths_text)
    if pool_depth <= 0:
        raise AbError(f"池深必须是正整数，收到 {pool_depth}")
    return (pool_depth,)


def build_parser() -> argparse.ArgumentParser:
    """命令行参数定义。抽出来是为了让测试能直接问「默认值是什么」，不用拼 argv。"""

    parser = argparse.ArgumentParser(description="工具层第一段融合池深度 → top-8 命中曲线")
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument(
        "--chroma-v2",
        type=Path,
        default=None,
        help="对照向量库（向量侧带文档身份前缀重嵌的那份）",
    )
    parser.add_argument(
        "--chroma-v3",
        type=Path,
        default=Path(DEFAULT_CHROMA_PATH),
        help=(
            "第二份对照向量库（unit 级切块 + 导读行重嵌的那份）；默认就是"
            "**线上装配的那份**（`akr.mcp_server.server.DEFAULT_CHROMA_PATH`）"
        ),
    )
    parser.add_argument(
        "--fold",
        choices=sorted(FOLD_MODES),
        default=DEFAULT_FOLD,
        help=(
            f"id_map 折叠口径；默认 {DEFAULT_FOLD!r} = 线上口径，"
            "'best' = 第三十七节那张历史表的口径"
        ),
    )
    parser.add_argument(
        "--rerank",
        choices=sorted(RERANK_MODES),
        default=RERANK_DEFAULT,
        help="融合后的排序模式；默认 rrf = 历史行为",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="rerank=cosine 时余弦先验的权重（0~1）",
    )
    parser.add_argument(
        "--rerank-usage",
        type=Path,
        default=None,
        help=(
            "rerank=cross 时的重排用量账本（JSON）；不传就不落盘"
            "（token 仍会随响应回来，只是不记账）"
        ),
    )
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ab", type=Path, default=DEFAULT_AB, help="题目与 gold 的来源")
    parser.add_argument(
        "--unit-map",
        type=Path,
        default=DEFAULT_UNIT_MAP,
        help="v3（unit 级库）的 unit → 父块 id 映射；只有给了 --chroma-v3 时才读",
    )
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument(
        "--pool-depth",
        type=int,
        default=None,
        help="只量这一个池深（覆盖 --depths），用来对账要上线的那个值",
    )
    parser.add_argument("--json", type=Path, default=None, help="把曲线落成 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    started = time.perf_counter()
    for path in (args.bm25, args.manifest, args.ab):
        if not path.exists():
            raise AbError(f"缺文件：{path}（先跑 scripts/retrieval_ab.py --json data/retrieval-ab.json）")
    stores = resolve_stores(args.chroma, args.chroma_v2, args.chroma_v3)
    for _, chroma in stores:
        if not chroma.exists():
            raise AbError(f"缺向量库目录：{chroma}（先跑 scripts/rebuild_vector_index.py）")
    id_maps = resolve_id_maps(stores, args.unit_map)

    depths = resolve_depths(args.pool_depth, args.depths)
    rows = list(json.loads(args.ab.read_text(encoding="utf-8"))["rows"])
    payload: dict[str, Any] = {
        "top_k": DEFAULT_TOP_K,
        "depths": list(depths),
        "rerank": args.rerank,
        "alpha": args.alpha,
        "rerank_usage": str(args.rerank_usage) if args.rerank_usage else None,
        "fold": args.fold,
        "stores": {},
        "note": "池深 = rrf_fuse 拿几路候选去融；返回给模型的仍是 top_k 条",
        "measured_with": "search_documents(query, top_k=8, pool_depth=D, rerank=R)",
        "vector_only_measured_with": "VectorStore.query(question, top_k=D)（深池召回，不经融合/收窄）",
    }
    # 重排器**按需**建：`cross` 是唯一会发网络请求的一档（其余两档一行都不碰它）。模型名与
    # Key 走 `SILICONFLOW_RERANK_MODEL`（`akr.config.RerankSettings`），与嵌入共用同一个网关。
    reranker = build_reranker(usage_path=args.rerank_usage) if args.rerank == "cross" else None
    if reranker is not None:
        print(f"[rerank] cross-encoder 已装载，用量账本={args.rerank_usage}")
    for name, chroma in stores:
        ctx = build_context(
            chroma_path=chroma,
            bm25_path=args.bm25,
            manifest_path=args.manifest,
            id_map=id_maps.get(name),
            fold=args.fold,
            reranker=reranker,
        )
        folded = f"，折叠映射 {len(id_maps[name])} 条" if name in id_maps else ""
        print(
            f"[load] {name} = {chroma}，Chroma 计数 {ctx.vectors.count()}{folded}，"
            f"折叠口径 {args.fold}，题目 {len(rows)} 条"
        )
        measured = measure(ctx, rows, depths, rerank=args.rerank, alpha=args.alpha)
        vector_only = measure_vector_only(ctx, rows, depths)
        payload["stores"][name] = {
            "chroma": str(chroma),
            "depths": measured,
            "vector_only": vector_only,
        }
        for depth in depths:
            print(describe(name, depth, measured[str(depth)]))
            print(describe_vector(name, depth, vector_only[str(depth)]))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {args.json}")
    print(f"[done] 耗时 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    sys.exit(main())
