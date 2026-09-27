from __future__ import annotations

import math
from collections.abc import Sequence

from akr.storage.hits import RetrievalHit

DEFAULT_K = 60
DEFAULT_ALPHA = 0.5


def rrf_fuse(
    result_lists: Sequence[Sequence[RetrievalHit]],
    *,
    k: int = DEFAULT_K,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """Reciprocal Rank Fusion：跨检索路的融合只依赖排名，不依赖分数尺度。

    向量余弦相似度落在 [-1, 1]，BM25 分数无上界，两者量纲不同，加权相加没有数学意义
    （参考项目正是这么做的，spec §6.3 记为对照素材 R-3）。RRF 用 1/(k+rank) 只吃排名，
    天然免归一化，并对单路召回质量差异鲁棒。

    溯源字段（doc_id / source / text）取该 chunk_id 的首次出现：同一个 chunk_id 的两条
    命中都由同一个 Chunk 派生，这些字段必然一致，因此首次出现即可，无需逐字段比对。
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalHit] = {}
    retrievers: dict[str, set[str]] = {}

    for hits in result_lists:
        for rank, item in enumerate(hits, start=1):
            if item.chunk_id not in first_seen:
                first_seen[item.chunk_id] = item
                scores[item.chunk_id] = 0.0
                retrievers[item.chunk_id] = set()
            scores[item.chunk_id] += 1.0 / (k + rank)
            retrievers[item.chunk_id].add(item.retriever)

    labels = {chunk_id: sorted(found) for chunk_id, found in retrievers.items()}
    fused = [
        RetrievalHit(
            chunk_id=chunk_id,
            doc_id=first_seen[chunk_id].doc_id,
            source=first_seen[chunk_id].source,
            text=first_seen[chunk_id].text,
            score=score,
            retriever="both" if len(labels[chunk_id]) > 1 else labels[chunk_id][0],
        )
        for chunk_id, score in scores.items()
    ]
    # 确定性：同分按 chunk_id 升序，保证同一批输入两次融合顺序完全一致。
    fused.sort(key=lambda item: (-item.score, item.chunk_id))
    return fused[:top_k] if top_k is not None else fused


def cosine_rerank(
    fused: Sequence[RetrievalHit],
    cosine: Sequence[RetrievalHit],
    *,
    alpha: float = DEFAULT_ALPHA,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """在 RRF 排名分上叠一层**有界**的余弦先验，重排后再截断。

    为什么需要它：`rrf_fuse` 的 1/(k+rank) 只看名次，向量路与 BM25 路各自的置信度在融合时
    被抹平——名次相同的两条命中，余弦 0.86 与 0.41 在融合后完全等价。而这一轮要收的是
    **排序**损失：深池里 20/25 的题已含 gold，线上前 8 只进 12/25（README「证据基线」§2）。

    三条口径（写死在这里，改一条就要改 `tests/storage/test_rrf_fusion.py`）：

    * `fused` 已按名次排好（`rrf_fuse` 的输出），本函数只在其上加权，不改它的相对序；
    * 余弦相似度**截断**到 [0, 1]：负相似度当作「没有信号」而不是「负分」——语料是年报，
      反向语义不代表这一句更该排在后面；
    * `cosine` 里没有的 chunk 不给分、也不当作 0 分：它们排在所有有信号的命中之后，且保持
      各自在 `fused` 里的相对序。缺一路输入（例如某块没进向量候选）不该被一个凑出来的
      0 分决定名次。

    混合公式：`(1 - alpha) * rrf_norm + alpha * cosine`，其中 `rrf_norm = score / max(score)`。
    RRF 分数在 [1/(k+N), 1/(k+1)] 之间，先按本次候选里的最大值归一到 (0, 1]；不做这一步，
    `alpha` 的单位会随 `k` 与候选数漂移，调参结果没法复现。

    `fused` 的 `chunk_id` 必须唯一（`rrf_fuse` 的输出天然满足）：重复会让「按 chunk_id 查
    余弦分」变成二义，这里宁可报错也不猜。

    只在 `rerank="cosine"` 时被工具层调用（Task 3）；默认的 `rerank="rrf"` 一行都不走这条
    路径，历史基线因此不受影响。
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1]，收到 {alpha}")
    ordered = list(fused)
    seen: set[str] = set()
    for item in ordered:
        if item.chunk_id in seen:
            raise ValueError(f"fused 里的 chunk_id 必须唯一，重复出现：{item.chunk_id!r}")
        seen.add(item.chunk_id)
    if not ordered:
        return []

    cosine_by_id: dict[str, float] = {}
    for item in cosine:
        # 同一条命中出现在多份余弦列表里时取更高分：那是它最好的证据；取最后一次会让
        # 结果取决于调用方拼列表的顺序。
        cosine_by_id[item.chunk_id] = max(cosine_by_id.get(item.chunk_id, 0.0), item.score)

    top_score = max(item.score for item in ordered)
    scale = top_score if top_score > 0 else 1.0

    def blended(item: RetrievalHit) -> float:
        prior = min(1.0, max(0.0, cosine_by_id[item.chunk_id]))
        return (1.0 - alpha) * (item.score / scale) + alpha * prior

    present = [item for item in ordered if item.chunk_id in cosine_by_id]
    absent = [item for item in ordered if item.chunk_id not in cosine_by_id]
    scores = {item.chunk_id: blended(item) for item in present}
    reranked = sorted(present, key=lambda item: (-scores[item.chunk_id], item.chunk_id))
    result = [
        RetrievalHit(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            source=item.source,
            text=item.text,
            score=scores[item.chunk_id],
            retriever=item.retriever,
        )
        for item in reranked
    ]
    result.extend(absent)
    return result[:top_k] if top_k is not None else result


def score_rerank(
    fused: Sequence[RetrievalHit],
    scores: Sequence[float],
    *,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """用一路**外部打分**（cross-encoder 重排分，`akr.storage.rerank`）重排融合候选。

    与 `cosine_rerank` 的两处刻意不同，都源于「这条分数从哪来」：

    * **逐条对齐、不参杂**：`scores[i]` 必须就是 `fused[i]` 的分。长度不等直接报错——
      `cosine_rerank` 允许候选缺一路输入，是因为那是**两路检索的并集**；而重排器的候选集
      就是 `fused` 本身，少一条只可能是调用方把两份列表错配了。
    * **不混合**：结果顺序**完全**由重排分决定，RRF 名次一个数都不参与。`cosine_rerank` 的
      `(1-alpha) * rrf_norm + alpha * cosine` 是在两路信号之间调和；这里没有两路信号——
      重排器已经看过「查询 + 候选原文」，它的排序就是结论，再掺 RRF 只会让「重排到底有没有
      用」这个问题量不出来。

    同分时保持 `fused` 里的先后（那是 RRF 的结论），而不是按 chunk_id——比较键里的 `index`
    在语义上是「原顺序当兜底」，不是实现细节。

    `RetrievalHit.score` 写的是**重排分**（与 `cosine_rerank` 写融合分同一口径：score 描述
    「这条为什么排在这」，重排后它就不再是任何一路的原始分）。非有限的分数（NaN / inf）
    直接报错：排序键里出现 NaN 会让结果取决于比较次数，那是不可复现的名次。
    """

    if len(scores) != len(fused):
        raise ValueError(
            f"重排分必须与候选一一对应：候选 {len(fused)} 条、分值 {len(scores)} 个"
        )
    for position, value in enumerate(scores):
        if not math.isfinite(float(value)):
            raise ValueError(f"第 {position} 条候选的重排分不是有限数：{value!r}")
    order = sorted(range(len(fused)), key=lambda index: (-float(scores[index]), index))
    picked = order[:top_k] if top_k is not None else order
    return [
        RetrievalHit(
            chunk_id=fused[index].chunk_id,
            doc_id=fused[index].doc_id,
            source=fused[index].source,
            text=fused[index].text,
            score=float(scores[index]),
            retriever=fused[index].retriever,
        )
        for index in picked
    ]
