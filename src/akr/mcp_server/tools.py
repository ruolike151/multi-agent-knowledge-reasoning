"""5 个检索类工具的实现。

硬约束：本模块不得导入 `src/akr/model/` 下的任何模块（工具是纯函数，不调用 LLM），
所有返回值都带 chunk_id / source / score 溯源元数据（spec §6.2）。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from akr.mcp_server.schemas import (
    ChunkResult,
    DocumentItem,
    DocumentOverview,
    EntityCandidateOut,
    EntityRefOut,
    EntityResult,
    GraphPathOut,
    GraphResult,
    Hit,
    RelRefOut,
    SearchResult,
)
from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import DocMeta
from akr.ingest.table_context import compose_index_text
from akr.storage import sqlite_store
from akr.storage.fusion import DEFAULT_ALPHA, cosine_rerank, rrf_fuse, score_rerank
from akr.storage.hits import RetrievalHit
from akr.storage.query_hints import QueryHints, extract_hints, narrow_doc_ids


@dataclass(frozen=True)
class ToolContext:
    """工具所需的全部依赖。测试传 fake，生产由 Task 21 的 server 构造。"""

    conn: Any  # sqlite3.Connection
    vectors: Any  # VectorStore
    bm25: Any  # Bm25Index
    graph: Any  # GraphStore
    artifacts: Any  # ArtifactStore
    # 文档身份表（doc_id → 公司/代码/报告期/标题）。有它才开启「查询侧收窄」；
    # 没有就是历史行为（全库检索）。见 `akr.storage.query_hints` 与下面的
    # `_narrowed_doc_ids`：这是一条**确定性规则**，不增加任何模型调用。
    doc_meta: Mapping[str, DocMeta] | None = None
    # cross-encoder 重排器（`akr.storage.rerank.RerankFn`）：`(query, documents) -> 分值列表`。
    # 只有 `rerank="cross"` 会用它，且**用一次就是一次网络往返**；装配层不给（None）时，
    # 工具层收到 `rerank="cross"` 会当场报错，不会静默退回 rrf——否则「重排有没有用」
    # 这个问题会被一个假的对照组骗过去。默认为 None = 历史路径一行都不碰它。
    reranker: Any = None

_ALLOWED_FILTER_KEYS = frozenset({"doc_ids"})

# 两段式检索的三个护栏：
#   * `two_stage_docs` 由调用方给（Retriever worker 用 3）——第一段前 N 个**不同文档**；
#   * 第一段「定文档」要从**深池**里取名次：A/B 实测（`scripts/retrieval_ab.py`，
#     `stage2@8/M3`）的那 3 个文档取自 depth=50 的融合结果，不是返回给模型的 top-8。
#     两者不是一回事：top-8 里常常只有 2~3 个不同文档，定出来的集合与深池差很远
#     （2026-09-25 实测：浅池定文档 6/25，深池定文档 10/25）；
#   * 第二段已经给出 top_k 条，第一段的额外命中最多再补这么多条，用来控制进 prompt 的
#     片段数（每条约 600 字符，条数直接等于 token 成本）。
TWO_STAGE_EXTRA_HITS = 4
TWO_STAGE_PICK_DEPTH = 50

# 邻块扩窗（同一文档内按 `ordinal` 取邻近块）：
#   * `EXPAND_NEIGHBORS` = **宽度**，0 = 关闭 = 历史行为；2 = 「向两侧各取 2 块」；
#   * `EXPAND_ANCHORS` = 只对返回列表的前 3 条命中做扩窗。
# 为什么锚点数是常量、不进参数：Task 4 已经离线把「宽度 × 锚点数」四个格子全量过了，
# 线上只放一个旋钮，就没人能配出一格**没量过**的组合。宽度与锚点的实测对照见
# `data/retrieval-ab.json` → `summary["expand@8/prod±2/3anchors"]`（本任务上线的那一格）。
EXPAND_NEIGHBORS = 0
EXPAND_ANCHORS = 3

# 第一段融合的候选池深：`pool_depth > 0` 时两条检索路各取这么多候选去融合，融合结果仍截回
# top_k。用 0 当哨兵而不是 `None`：MCP 的 JSON Schema 里 `["integer","null"]` 对客户端不友好，
# 而「0 = 不放大池子」在整数域里不会与任何合法池深冲突（池深 0 本来就是「不检索」的意思）。
#
# 2026-09-26 接线：默认值从 0（= 池深等于 top_k，历史行为）改成 12，线上装配同时切到
# `data/chroma-v3`（`akr.mcp_server.server.DEFAULT_CHROMA_PATH`）。实测（25 道可答题、
# v3、unit→父块累加折叠）融合 top-8 命中：池深 8 → 22、12 → 23、16 → 23、24 → 22、
# 50 → 21；MRR 0.6267 / 0.5867 / 0.5967 / 0.6004 / 0.5550。表见内部偏差日志
# 第三十九节，复算命令 `scripts\retrieval_pool_depth.py --fold rrf_sum`。
#
# 两个结论写在这里，免得后人把「深池 = 更准」当默认假设：
#   * **越深并不越好**：24/50 比 12/16 低，MRR 更是从池深 8 就开始降。机制是 RRF 的池子越深，
#     靠后的「同名次共识」条目越多，越容易把本来排在前面的 gold 顶下去（配 unit 级累加折叠时
#     更明显：同一父块的单元越多，累加分越容易压过单条强命中）。
#   * **12 与 16 并列峰值，取更浅的 12**：峰值与池深 8 只差 1 题（n=25，属噪声），所以这一档
#     是「不亏」而不是「显著更准」；要定案需要更大的评测格子。要复算历史基线显式传
#     `pool_depth=0`（逐字等于改动前的行为）。
#
# 2026-09-26 第二次实测（`tmp/probe_lock_blend.py` / `tmp/probe_gain_attribution.py`，同一批
# 25 题、只读重放、零 token）把上面那条「深池 MRR 更低」的结论**限定到了 k=60 这一档**：
# 池深 50 配 `RRF_K_DEFAULT=20`，top-8 是 23/25 · recall@8 80.0% · MRR 0.6567，
# 而线上旧口径（池深 12 · k=60）是 23/25 · 78.0% · 0.6490——**命中不降、召回 +1 份 gold、
# MRR +0.008**。深池掉 MRR 的机制是「池越深，靠后的同名次共识越多」：k 越小，`1/(k+rank)`
# 对名次越敏感，靠后的共识条目拿到的分越小，就不会顶掉前面的强命中。所以**池深与 k 是一对
# 参数，必须一起调**；只动池深（k=60）确实会把 MRR 从 0.6490 拉到 0.6247。
#   * 逐题归属：+1 份 gold 来自 `core-md-007` 的第二条 gold 进到第 8（recall 1.0）；
#     MRR 的 8 处名次移动里有 4 处只是相邻名次（`core-sh2-002` 4→2、`core-mh-004` 4→3、
#     `core-mh-009` 2→1、`core-md-001` 7→4）。
#   * n=25 的评测集上 1 道题翻转就值 4pp，这一档是「不亏」而不是「显著更准」；要拍板需要
#     更大的格子。要复算改动前的线上基线显式传 `pool_depth=12, rrf_k=60`。
POOL_DEPTH_DEFAULT = 50

# 第一段融合之后的排序模式：
#   * "rrf"    = 历史行为：谁的名次和靠前谁在前（`rrf_fuse` 的输出顺序）；
#   * "cosine" = 在 RRF 候选集上叠一层有界的余弦先验（`akr.storage.fusion.cosine_rerank`）。
#   * "cross"  = 拿 **cross-encoder**（`akr.storage.rerank`，硅基流动 `/v1/rerank`）对整个
#     候选池逐条打分，顺序完全由它决定（`akr.storage.fusion.score_rerank`）。与前两条的
#     本质差别：前两条**不新增任何模型调用**（余弦分是向量路已经算出来的东西），这一条
#     每检索一次就多一次网络往返 + 一次按候选数计费的打分。
#
# **2026-09-26 用户拍板：把 `"cross"` 开成线上默认**（此前是可选开关、默认 `"rrf"`）。
# 依据（工具层 25 题、只读重放、`tmp/verify-p50k20.json`）：池深 50 · k=20 下锁头 4 条
# 是 24/25 · recall@8 80.0% · MRR 0.6617，比零重排那一档多 1 题命中（23/25 · 0.6567）。
#   * **代价写在这里，别忘**：每次检索多一次 `/v1/rerank`（池深 50 、锁 4 → 46 条候选，
#     按候选数计费；参考第四十节一次 12 条候选 = 2,839 token / 约 0.46s）。所以它是**唯一
#     一条会发网络请求的排序路径**，也是唯一一条「后端抽风就会拖垮检索」的路径。
#   * 缺 `ctx.reranker` 时**当场报错、不静默退回 rrf**：静默退回只会在「没有重排」的情况下
#     量出一列看着正常的假数字（`_fuse` 里同一条纪律）。离线脚本 / 单测要么装配重排器，
#     要么**显式**传 `rerank="rrf"`——「我没装重排器」必须是调用方说出来的话。
#   * 要复算 2026-09-26 之前的线上基线（第四十节那批）显式传 `rerank="rrf"`。
# 白名单而不是「不是 rrf 就当 cosine」：拼错一个字母必须报错，不能静默换一条检索路径。
RERANK_DEFAULT = "cross"
RERANK_MODES = frozenset({"rrf", "cosine", "cross"})

# 第一段融合的 RRF 常数 k（`rrf_fuse` 的 `1/(k + rank)`）。
# **刻意不复用 `akr.storage.fusion.DEFAULT_K`（=60）**：那个常数还被向量侧的 `rrf_sum`
# 折叠（`VectorStore._aggregate`，线上口径 `server.DEFAULT_VECTOR_FOLD`）当作 `1/(k+名次)` 的
# 累加权重用。两处公式同形，优化目标却相反——折叠要的是「同一父块的多条 unit 累加起来顶上去」，
# 融合要的是「靠后的同名次共识别把前面的强命中顶掉」。共用一个常数就会让「调融合」顺带改掉
# 「检索候选集合」，量到的数字与线上行为立刻脱钩（这次就是先撞上这一点才拆开的）。
# 实测（25 题、只读重放、`tmp/probe_lock_blend.py`）：池深 50 下 k=20 / 40 / 60 的
# top-8 命中都是 23/25，MRR 0.6567 / 0.6447 / 0.6247——**认准 20**。
RRF_K_DEFAULT = 20

# `rerank="cross"` 时的「锁定头部条数」：前 `lock` 条由 RRF 独占，重排器只在剩下的
# `top_k - lock` 个名额里挑（`lock == top_k` 时一次重排都不调）。
#
# 为什么需要它：把**整个深池**交给 cross-encoder 是负结果——池越深越差（`tmp/pool-depth-cross-full.json`
# 在 v3 上量到顶八 8/12/16/24/50 = 22/22/21/20/18，MRR 0.6613 → 0.4523），因为深池里大量候选是
# 同父块的其它 unit，逐条打分看不到父块聚合那一层。把「名次的归属权」拆开就修好了：实测
# （`tmp/probe_lock_head.py --fine` + `tmp/probe_gain_attribution.py`，池深 50 · k=20 · 锁 4）
# top-8 = **24/25 · recall@8 80.0% · MRR 0.6617**，比零重排那一档多 1 题命中
# （`core-mh-001` 的 gold 从池内第 20 名被重排抬到第 8）；锁 6 换到 recall 82.0%（34/42 份 gold）。
#   * 同一批分数下「锁 4 但尾池取 12」只有 23/25（`tmp/probe_lock_blend.py` 的 `tail=24/50` 段），
#     说明**头池与尾池本来就要分开**：头部按浅池的共识定，尾部才有深池的余地。
#   * 代价是每题一次 `/v1/rerank`（按候选数计费）。2026-09-26 用户拍板连它一起开成默认，
#     所以这个常数就是**线上口径**，不再是「可选旋钮的说明书」。
#   * 「锁满」仍然免费：`lock >= top_k` 时尾池没有名额，一次重排都不调（见 `_fuse`）。
#     默认值**按 `top_k` 饱和**（`min(4, top_k)`），因为线上的 `graph_walker` 就是
#     `top_k=3`——锁 4 会当场炸；饱和到 3 等于那一档退回纯 RRF，语义完整且不白花一次往返。
#     显式传的 `lock` 仍严格校验（`lock > top_k` 报错，不静默夹取）。
RERANK_LOCK_DEFAULT = 4


def _doc_ids(doc_filters: dict | None) -> tuple[str, ...] | None:
    if not doc_filters:
        return None
    unknown = set(doc_filters) - _ALLOWED_FILTER_KEYS
    if unknown:
        raise ValueError(f"不支持的过滤条件：{sorted(unknown)}；只支持 {sorted(_ALLOWED_FILTER_KEYS)}")
    doc_ids = doc_filters.get("doc_ids")
    return tuple(doc_ids) if doc_ids else None


def _to_hit(hit) -> Hit:
    return Hit(**dataclasses.asdict(hit))


def _top_docs(hits: list, count: int) -> tuple[str, ...]:
    """按名次取前 `count` 个**不同文档**。第二段的过滤条件就是它。"""

    ordered: list[str] = []
    for hit in hits:
        if hit.doc_id not in ordered:
            ordered.append(hit.doc_id)
        if len(ordered) >= count:
            break
    return tuple(ordered)


def _query_hints(query: str, ctx: ToolContext) -> QueryHints:
    """抽问句里的证券代码 / 年份 / 公司名；没有身份表就返回空线索（不收窄）。"""

    metas = getattr(ctx, "doc_meta", None)
    return extract_hints(query, metas) if metas else QueryHints()


def _narrowed_doc_ids(query: str, ctx: ToolContext) -> tuple[str, ...]:
    """把候选文档收窄成 1~3 份；抽不到线索或没有身份表时返回空元组（= 不收窄）。

    为什么值得单开一步（2026-09-25 只读 A/B，`scripts/retrieval_ab.py`）：光把文档身份
    拼进 BM25 索引文本，top-8 命中 10/25 → **9/25**（反而退一步）——身份 token 在自己
    文档的几百个块里都出现，idf 被同族块摊薄了。先用代码把候选文档收到 1~3 份再检索，
    同一批题变成 **12/25**、标准 recall@8 24.0% → 32.0%。两步必须成对上线。
    """

    metas = getattr(ctx, "doc_meta", None)
    if not metas:
        return ()
    return narrow_doc_ids(extract_hints(query, metas), metas)


def rerank_document_text(
    hit: RetrievalHit, doc_meta: Mapping[str, DocMeta] | None
) -> str:
    """喂给 cross-encoder 的文本：**文档身份前缀 + 原文**（没有身份表时退回原文）。

    为什么必须带前缀。v3 是**行级** unit 块，正文常常只剩几行数字：没有公司、没有证券
    代码、也没有报告期，于是「2023 年」和「2024 年」两张表在重排器眼里长得一模一样。
    实测（25 道可答题、v3、池深 12、`tmp/probe_rerank_text.py`，同一批候选池只换渲染）：

    | 喂给重排器的文本 | top-8 命中 | recall@8 | MRR |
    |---|---|---|---|
    | 原文（未带前缀） | 21/25 | 64.0% | 0.3411 |
    | 只加表格导读行 | 19/25 | 58.0% | 0.3158 |
    | **身份前缀 + 原文（本函数）** | **22/25** | **70.0%** | **0.6133** |
    | 身份前缀 + 导读行 | 22/25 | 70.0% | 0.5220 |
    | 身份前缀（含章节路径）+ 导读行 | 22/25 | 66.0% | 0.5374 |

    两个负结果照写：只加导读行**比**原文还差（缺了身份，期间行反而把重排器推向错年份）；
    在身份前缀之上再叠章节路径与导读行，命中数不加、MRR 更低。所以这里只拼**文档级**
    身份前缀——顺带也就不必为每个候选多查一次 chunk 的 `section_path`。

    拼装复用 `compose_index_text`：索引文本的拼法与检索侧保持同一份实现，不另开一套分隔符。
    """

    meta = doc_meta.get(hit.doc_id) if doc_meta else None
    if meta is None:
        return hit.text
    return compose_index_text(header=meta.prefix(""), lead="", text=hit.text)


def _fuse(
    vector_hits: list,
    bm25_hits: list,
    *,
    depth: int,
    top_k: int,
    rerank: str,
    alpha: float = DEFAULT_ALPHA,
    k: int = RRF_K_DEFAULT,
    lock: int | None = None,
    query: str = "",
    reranker: Any = None,
    doc_meta: Mapping[str, DocMeta] | None = None,
) -> list[RetrievalHit]:
    """第一段融合：两路候选先按 RRF 融成 `depth` 条，再按 `rerank` 决定最终顺序。

    `depth` 必须 ≥ `top_k`：候选比返回条数还少时不可能凑够一支队列，那是配置错误。

    `rerank="rrf"`（**显式**传 = 历史口径 / 复算旧基线用；工具层默认档已是 `cross`）逐字
    等于历史行为：RRF 排序后截前 `top_k` 条。注意这里先融 `depth` 条再截 `top_k`，
    与「直接融 `top_k` 条」在 `depth == top_k` 时完全等值（截断只砍尾巴，不重排），池深
    放大后才开始有差别。

    `rerank="cosine"` 在 RRF 候选集上叠一层**有界**余弦先验，再截前 `top_k`：谁进候选由
    RRF 决定，谁排前面由向量路的**原始余弦分**决定（`vector_hits` 的 `score` 就是向量库的
    直接输出，不是融合分）。为什么需要它——RRF 只吃名次，余弦 0.86 与 0.41 的两条命中只要
    名次相同就完全等价，而本轮的损失恰恰是**排序**：深池 20/25 的题已含 gold、前 8 只进
    12/25（README「证据基线」§2）。

    `rerank="cross"` 把候选池的**尾部**逐条送给 cross-encoder 打分，再截前 `top_k`：候选的
    集合仍由两路召回 + RRF 决定，顺序交给重排器。它比 "cosine" 多花一次网络往返
    （`akr.storage.rerank`），所以 `reranker=None` 时**当场报错**——没有重排器还叫 cross
    重排，只会量出一列假的「没效果」。

    `lock` 只在 `rerank="cross"` 时有意义（其余模式传非 0 会报错，不静默忽略）：前 `lock`
    条由 RRF 独占，重排器只在剩下的 `top_k - lock` 个名额里挑；`lock == top_k` 时一次重排
    都不调。动机是「整池重排」的负结果与修法见 `RERANK_LOCK_DEFAULT`。

    `lock=None`（默认）表示「用工具层默认值」：`RERANK_LOCK_DEFAULT`，并**按 `top_k` 饱和**
    （`min(4, top_k)`）——线上的 `graph_walker` 传 `top_k=3`，锁 4 会直接越界；饱和到 3 等于
    那一档退回纯 RRF，语义完整且一次网络往返都不花。显式传值仍严格校验，不静默夹取。

    送给重排器的文本**不是** `hit.text` 原文，而是 `rerank_document_text`（身份前缀 + 原文）：
    实测只喂原文时，池深越深越糟（v3、池深 24：RRF 23/25 → cross 10/25，见
    `tmp/probe_rerank_text.py`）。
    """

    if depth < top_k:
        raise ValueError(f"融合池深不能小于返回条数：depth={depth} < top_k={top_k}")
    if k < 1:
        raise ValueError(f"RRF 常数 k 必须是正整数，收到 {k}")
    if lock is None:
        lock = min(RERANK_LOCK_DEFAULT, top_k) if rerank == "cross" else 0
    else:
        if not 0 <= lock <= top_k:
            raise ValueError(f"锁定条数必须在 [0, top_k]：lock={lock} > top_k={top_k}")
        if lock and rerank != "cross":
            raise ValueError(
                f"lock={lock} 只在 rerank='cross' 时有意义，当前 rerank={rerank!r}"
                "（不静默忽略：忽略等于用 rrf 的顺序冒充『重排过』）"
            )
    fused = rrf_fuse([vector_hits[:depth], bm25_hits[:depth]], k=k, top_k=depth)
    if rerank == "rrf":
        return fused[:top_k]
    if rerank == "cosine":
        return cosine_rerank(fused, vector_hits, alpha=alpha, top_k=top_k)
    if rerank == "cross":
        if reranker is None:
            raise ValueError(
                "rerank='cross' 需要装配重排器：ctx.reranker 为 None"
                "（见 akr.storage.rerank.build_reranker）"
            )
        head = fused[:lock]
        tail = fused[lock:]
        if lock >= top_k or not tail:
            # lock == top_k：名额全被 RRF 占满，重排一次都不调（也就一分钱不花）。
            # `lock >= top_k` 必须单独判，不能只靠 `not tail`：池深大于 top_k 时 tail 非空，
            # 但 `score_rerank(top_k=0)` 一条也选不出来——那一次网络往返纯属白花。
            return head[:top_k]
        scores = reranker(query, [rerank_document_text(hit, doc_meta) for hit in tail])
        return [*head, *score_rerank(tail, scores, top_k=top_k - lock)]
    raise ValueError(f"未知的 rerank 模式：{rerank!r}；可选 {sorted(RERANK_MODES)}")


def _neighbour_offsets(width: int) -> tuple[int, ...]:
    """邻块偏移顺序：锚点自己（0）先，然后按距离由近到远，同距离先取 ordinal **更小**那条。

    为什么 0 必须排最前：扩窗后的列表要截回 `top_k`，锚点本身是「检索路已经认可过的证据」，
    先占位才不会被自己扩出来的邻块挤掉。同距离先取下一条（`-1` 先于 `+1`），是因为年报
    表格块的「指标名 + 期间」导读行排在数值行**之前**——救表格题时上一条比下一条更值钱。

    这里与离线脚本的 `scripts/retrieval_ab.py::_window_offsets`（Task 4）**逐字同构**：
    两边同序，逐题对账脚本才可能比对出 0 处差异。
    """

    offsets = [0]
    for distance in range(1, width + 1):
        offsets.extend((-distance, distance))
    return tuple(offsets)


def _doc_chunks(
    ctx: ToolContext, cache: dict[str, dict[str, Chunk]], doc_id: str
) -> dict[str, Chunk]:
    """一份文档的 `{chunk_id: Chunk}`，来自 SQLite 的 `chunks` 表（按 `ordinal` 升序）。

    `cache` 由调用方持有，一次扩窗内同一文档只查一次库：一次 `search_documents` 的锚点
    通常挤在 1~3 份文档里，不缓存就会对同一份文档查 3 遍（每份几百行）。

    为什么必须走 SQLite：邻块的 `doc_id` / `source` / `text` 只能来自真实入库的行。
    从锚点拼一个「文档内序号」再回头去找向量库，等于凭空造证据。
    """

    cached = cache.get(doc_id)
    if cached is None:
        cached = {row.chunk_id: row for row in sqlite_store.list_chunks(ctx.conn, doc_id)}
        cache[doc_id] = cached
    return cached


def _neighbourhood(
    ctx: ToolContext,
    hits: list,
    *,
    width: int,
    top_k: int,
) -> list[RetrievalHit]:
    """把前 `EXPAND_ANCHORS` 条命中换成「锚点 + 同文档内 ordinal 邻近的块」，再截回 `top_k`。

    三条不放宽的规矩（对应 README 的全局约束）：

    * **邻块沿用锚点的 `retriever` 与 `score`**——它没有自己的检索分，编一个就是造假；
      `Hit.retriever` 的 `Literal` 也一个字不改（只有 vector / bm25 / both 三个取值）；
    * **不跨文档**：`ordinal` 只在文档内有意义，别的文档里序号接近的块不算邻块；
    * **截回 `top_k`**：返回给模型的条数永远等于 `top_k`，扩窗不许把 prompt 撑大。

    与离线 `expand_window`（Task 4）**唯一**一处刻意不同：锚点所在文档在 SQLite 里没有行时，
    这里**保留锚点**、只是不扩（宁可少扩，也不许因为本地库缺一行就把一条真命中从证据里删掉）；
    离线那份会把这条锚点丢掉（它判的是「两条路看的不是同一批数据」）。Step 5 的逐题对账
    脚本兜住这处差异：`diffs` 非空时，先看差异 id 是不是锚点本身。
    """

    if width <= 0 or not hits:
        return list(hits)
    cache: dict[str, dict[str, Chunk]] = {}
    ordinal_index: dict[str, dict[int, Chunk]] = {}
    offsets = _neighbour_offsets(width)
    expanded: list[RetrievalHit] = []
    seen: set[str] = set()
    for index, hit in enumerate(hits):
        if hit.chunk_id not in seen:
            seen.add(hit.chunk_id)
            expanded.append(hit)
        if index >= EXPAND_ANCHORS:
            # 第 4 条以后的命中只按原顺序留在列表里，不参与扩窗：否则 prompt 里全是邻块，
            # 而模型看到的条数是固定的（top_k），等于用邻块把检索路的真命中挤出去。
            continue
        doc_chunks = _doc_chunks(ctx, cache, hit.doc_id)
        anchor = doc_chunks.get(hit.chunk_id)
        if anchor is None:
            continue
        ordered = ordinal_index.get(hit.doc_id)
        if ordered is None:
            ordered = {row.ordinal: row for row in doc_chunks.values()}
            ordinal_index[hit.doc_id] = ordered
        for offset in offsets:
            if offset == 0:
                continue
            neighbour = ordered.get(anchor.ordinal + offset)
            if neighbour is None or neighbour.chunk_id in seen:
                continue
            seen.add(neighbour.chunk_id)
            expanded.append(
                RetrievalHit(
                    chunk_id=neighbour.chunk_id,
                    doc_id=neighbour.doc_id,
                    source=neighbour.source,
                    text=neighbour.text,
                    score=hit.score,
                    retriever=hit.retriever,
                )
            )
    return expanded[:top_k]


def search_documents(
    ctx: ToolContext,
    query: str,
    top_k: int = 8,
    doc_filters: dict | None = None,
    two_stage_docs: int = 0,
    pool_depth: int = POOL_DEPTH_DEFAULT,
    rerank: str = RERANK_DEFAULT,
    rerank_alpha: float = DEFAULT_ALPHA,
    rrf_k: int = RRF_K_DEFAULT,
    rerank_lock: int | None = None,
    expand_neighbors: int = EXPAND_NEIGHBORS,
) -> SearchResult:
    """混合检索：两路各取候选，再用 RRF 融合后截回 `top_k`（spec §6.3）。

    `pool_depth > 0` 时把两路各取 `pool_depth` 条候选去融合（默认 50 是线上口径；
    `0` = 各取 `top_k` 条，逐字等于 2026-09-26 之前的历史行为）。
    动机（2026-09-25 只读 A/B）：深池里 20/25 的题已经含有 gold，而线上前 8 只有 12/25，
    差的是**排序**不是**召回**——多给 `rrf_fuse` 一些候选，靠后的共识条目才有机会累到分。
    代价只有本地计算：向量库仍然只查一次（一次嵌入），BM25 也仍然只查一次。
    **深池必须配小 `rrf_k`**：只动池深（k=60）时 12/16 是峰值（23/25）、24/50 反而更低
    （22/21）；配 `rrf_k=20` 后池深 50 反而是最好的一档（23/25 · recall@8 80.0% · MRR 0.6567，
    线上旧口径 12·60 是 23/25 · 78.0% · 0.6490）——见 `POOL_DEPTH_DEFAULT` 与
    `RRF_K_DEFAULT` 上面的注释、内部偏差日志第三十九 / 四十三节。

    `rerank="cosine"` 时在 RRF 候选集上叠一层有界余弦先验（`rerank_alpha` 是它的权重，
    0 = 等价于 rrf、1 = 完全交给余弦），再截回 `top_k`。
    `rerank="cross"` 时把候选池的尾部交给 cross-encoder（`ctx.reranker`）逐条打分后再截回
    `top_k`，前 `rerank_lock` 条由 RRF 独占。
    **默认值（2026-09-26 起）是 `rerank="cross"` + 锁 4**，也就是线上路径每次检索都发一次
    `/v1/rerank`（池深 50 时 46 条候选）——它是**唯一一条会新增网络调用的排序路径**。
    缺 `ctx.reranker`（没装配重排器的离线/单测上下文）时这里**当场报错，不静默退回 rrf**：
    静默退回只会在「没有重排」的情况下量出一列看着正常的假数字。离线口径请显式传
    `rerank="rrf"`。送给重排器的文本是「文档身份前缀 + 原文」（`rerank_document_text`），
    不是 `hit.text` 原文。
    `rerank_lock=None`（默认）表示用工具层默认值，并按 `top_k` **饱和**：`top_k=3`（线上
    `graph_walker` 就是这条）时锁到 3，等于那一档退回纯 RRF、一次重排都不调。显式传值仍
    严格校验。
    `rerank_lock` 的实测（池深 50 · k=20）：锁 0 = 22/25（整池重排是负结果）、锁 4 = 24/25 ·
    recall@8 80.0% · MRR 0.6617、锁 6 = 23/25 · recall@8 82.0%（见 `RERANK_LOCK_DEFAULT`）。

    `expand_neighbors > 0` 时把**最终列表**的前 `EXPAND_ANCHORS` 条命中扩成「锚点 +
    同文档内 ordinal 邻近的块」，再截回 `top_k`（详见 `_neighbourhood`）。取值**就是宽度**：
    `2` = 向两侧各取 2 块。动机（2026-09-25 只读 A/B）：gold 落在某个命中的 ±2 之内时，
    窗口能覆盖 22/25 条题（`data/retrieval-ab.json` → `window_summary["window±2"]`），
    但那是**上界**；真扩完还要和锚点抢前 8 个名额，Task 4 离线量的就是后者。

    `two_stage_docs > 0` 时追加一段「文档内重搜」：把第一段融合结果里前 N 个不同文档
    当过滤条件，对同一个问句重搜一次；第二段结果排在最前，第一段的额外命中按序补在后面。
    动机与实测（2026-09-25 只读 A/B，`scripts/retrieval_ab.py`）：25 条可答题里 gold chunk
    进入前 8 的条数，纯第一段是 6/25；改成「原问句 + 前 3 个文档内重搜」是 **10/25**。
    代价只有一次嵌入 + 一次 BM25（**不增加任何模型调用**）。

    `expand_neighbors=0` / `two_stage_docs=0` 仍等于历史行为；**线上口径在 2026-09-26 这天
    连改了三处**：`pool_depth=50`、`rrf_k=20`、`rerank="cross" + 锁 4`。**要复算 2026-09-26
    之前的线上基线就显式传 `pool_depth=0, rerank="rrf"`**——历史那几张表（第三十四~三十八、
    第四十节）都是在这个口径下量的；要复算「池深 12 + k=60」那一档传
    `pool_depth=12, rrf_k=60`。
    """
    if top_k <= 0:
        raise ValueError("top_k 必须是正整数")
    if two_stage_docs < 0:
        raise ValueError("two_stage_docs 不能为负")
    if pool_depth < 0:
        raise ValueError("pool_depth 不能为负")
    if rrf_k < 1:
        raise ValueError(f"rrf_k 必须是正整数，收到 {rrf_k}")
    # 「便宜的检查」全部排在检索之前：参数打错时不许先花掉一次嵌入。
    if rerank not in RERANK_MODES:
        raise ValueError(f"未知的 rerank 模式：{rerank!r}；可选 {sorted(RERANK_MODES)}")
    if rerank_lock is None:
        # 「用工具层默认」：锁 4，但按 `top_k` 饱和（线上 `graph_walker` 传 `top_k=3`，
        # 锁 4 会越界；饱和到 3 = 名额全被 RRF 占满 = 那一档一次重排都不调）。
        lock = min(RERANK_LOCK_DEFAULT, top_k) if rerank == "cross" else 0
    else:
        if not 0 <= rerank_lock <= top_k:
            raise ValueError(
                f"rerank_lock 必须在 [0, top_k]：lock={rerank_lock} > top_k={top_k}"
            )
        if rerank_lock and rerank != "cross":
            raise ValueError(
                f"rerank_lock={rerank_lock} 只在 rerank='cross' 时有意义，当前 rerank={rerank!r}"
            )
        lock = rerank_lock
    if rerank == "cross" and ctx.reranker is None:
        raise ValueError(
            "rerank='cross' 需要装配 ctx.reranker（akr.storage.rerank.build_reranker）；"
            "缺它只会在「没有重排」的情况下量出一列假的对照"
        )
    if not 0.0 <= rerank_alpha <= 1.0:
        raise ValueError(f"rerank_alpha 必须在 [0, 1]，收到 {rerank_alpha}")
    if expand_neighbors < 0:
        raise ValueError(f"expand_neighbors 不能为负，收到 {expand_neighbors}")
    if expand_neighbors > 0 and ctx.conn is None:
        raise ValueError(
            "expand_neighbors 需要 SQLite 连接（ctx.conn）才能取邻块；"
            "线上由 akr.mcp_server.server 装配，跑对账请用 scripts/retrieval_tool_smoke.py --db"
        )
    doc_ids = _doc_ids(doc_filters)
    # 调用方自己给了 `doc_filters` 就尊重它（过滤意图是它的）；只有没给时才用
    # 问句里抽出的证券代码 / 年份收窄候选文档。
    narrowed = () if doc_ids is not None else _narrowed_doc_ids(query, ctx)
    if narrowed:
        doc_ids = narrowed
    two_stage = two_stage_docs > 0 and doc_ids is None
    # 池深与「取候选的深度」是两个数，必须分开：
    #   * `pool`  = 两条检索路各拿多少条**去融合**（本任务的旋钮）；
    #   * `depth` = 两条检索路各**查**多少条（两段式定文档要用深池，历史上固定 50）。
    # pool_depth=0 时 pool == top_k、depth 走历史分支，逐字等于改动前的默认路径
    # （线上默认现在是 50，见 `POOL_DEPTH_DEFAULT`）。
    pool = pool_depth if pool_depth > 0 else top_k
    depth = pool_depth if pool_depth > 0 else (TWO_STAGE_PICK_DEPTH if two_stage else top_k)
    vector_hits = list(ctx.vectors.query(query, top_k=depth, doc_ids=doc_ids))
    bm25_hits = list(ctx.bm25.search(query, top_k=depth, doc_ids=doc_ids))
    fused = _fuse(
        vector_hits,
        bm25_hits,
        depth=pool,
        top_k=top_k,
        rerank=rerank,
        alpha=rerank_alpha,
        k=rrf_k,
        lock=lock,
        query=query,
        reranker=ctx.reranker,
        doc_meta=getattr(ctx, "doc_meta", None),
    )
    stats = {"vector": len(vector_hits), "bm25": len(bm25_hits), "fused": len(fused)}
    hits = list(fused)
    if two_stage and fused:
        picked = _top_docs(
            rrf_fuse([vector_hits, bm25_hits], k=rrf_k, top_k=depth), two_stage_docs
        )
        stage2_vector = list(ctx.vectors.query(query, top_k=top_k, doc_ids=picked))
        stage2_bm25 = list(ctx.bm25.search(query, top_k=top_k, doc_ids=picked))
        stage2 = rrf_fuse([stage2_vector, stage2_bm25], k=rrf_k, top_k=top_k)
        already = {hit.chunk_id for hit in stage2}
        extras = [hit for hit in fused if hit.chunk_id not in already][:TWO_STAGE_EXTRA_HITS]
        hits = [*stage2, *extras]
        stats.update({"stage2_docs": len(picked), "stage2": len(stage2)})
    if expand_neighbors > 0:
        # 扩窗作用在**最终列表**上（两段式跑完之后的顺序）：拿第一段的结果去扩，
        # 锚点就会和模型真正看到的证据错位。计数口径 = 返回列表里新插进来的邻块条数。
        anchored = {hit.chunk_id for hit in hits}
        hits = _neighbourhood(ctx, hits, width=expand_neighbors, top_k=top_k)
        stats["neighbours"] = sum(1 for hit in hits if hit.chunk_id not in anchored)
    return SearchResult(
        hits=[_to_hit(hit) for hit in hits],
        retriever_stats=stats,
    )


def walk_graph(
    ctx: ToolContext,
    entity: str,
    hops: int = 2,
    relation_types: list[str] | None = None,
) -> GraphResult:
    """先用 mention 解析实体，再做多跳遍历；解析不到就返回空路径（不是异常）。"""
    candidates = ctx.graph.resolve(entity, limit=1)
    if not candidates:
        return GraphResult()
    paths = ctx.graph.walk(candidates[0].entity_id, hops=hops, relation_types=relation_types)
    return GraphResult(
        paths=[
            GraphPathOut(
                nodes=[EntityRefOut(**dataclasses.asdict(node)) for node in path.nodes],
                relations=[RelRefOut(**dataclasses.asdict(rel)) for rel in path.relations],
                text=path.text,
            )
            for path in paths
        ]
    )


def resolve_entity(ctx: ToolContext, mention: str, context: str = "") -> EntityResult:
    candidates = ctx.graph.resolve(mention, context=context, limit=5)
    return EntityResult(candidates=[EntityCandidateOut(**dataclasses.asdict(c)) for c in candidates])


def _source_of(ctx: ToolContext, doc_id: str) -> str:
    """source 只存在 SQLite 元数据里（artifact 只存正文），因此这里必须能查到。"""
    for row in sqlite_store.list_documents(ctx.conn):
        if row.doc_id == doc_id:
            return row.source
    raise ValueError(f"找不到 doc_id={doc_id} 的元数据，无法提供 source 溯源")


def fetch_chunk(ctx: ToolContext, chunk_id: str) -> ChunkResult:
    if "#" not in chunk_id:
        raise ValueError(f"非法 chunk_id：{chunk_id!r}（应为 '<doc_id>#c<hash>'）")
    doc_id = chunk_id.split("#", 1)[0]
    artifact = ctx.artifacts.get(doc_id, chunk_id)  # 不存在时抛 ArtifactNotFound
    return ChunkResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=_source_of(ctx, doc_id),
        text=artifact.text,
        char_range=list(artifact.char_range),
    )


def list_documents(ctx: ToolContext) -> DocumentOverview:
    return DocumentOverview(
        documents=[
            DocumentItem(
                doc_id=row.doc_id,
                source=row.source,
                doc_type=row.doc_type,
                chunk_count=row.chunk_count,
                updated_at=row.updated_at,
            )
            for row in sqlite_store.list_documents(ctx.conn)
        ]
    )


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "search_documents": search_documents,
    "walk_graph": walk_graph,
    "resolve_entity": resolve_entity,
    "fetch_chunk": fetch_chunk,
    "list_documents": list_documents,
}


def dispatch(ctx: ToolContext, name: str, arguments: dict) -> dict:
    """工具名 → 纯函数 → JSON-ready dict。未知工具名直接报错，不做兜底。"""
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"未知工具 {name!r}；可用工具：{sorted(TOOL_FUNCTIONS)}")
    return TOOL_FUNCTIONS[name](ctx, **arguments).model_dump()


def dispatch_safe(ctx: ToolContext, name: str, arguments: dict) -> dict:
    """给 MCP server / Agent 工具循环用的包装：把可预期的错误变成 error 载荷，不让进程崩。"""
    try:
        return dispatch(ctx, name, arguments)
    except (ValueError, KeyError, LookupError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
