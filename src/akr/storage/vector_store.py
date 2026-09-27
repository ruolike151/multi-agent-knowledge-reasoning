from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Mapping
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from akr.ingest.chunker import Chunk
from akr.storage.fusion import DEFAULT_K as RRF_K
from akr.storage.hits import RetrievalHit

EmbedFn = Callable[[list[str]], list[list[float]]]
IndexTextFn = Callable[[Chunk], str]

# 折叠口径：`id_map` 把索引行的 id 折回它的归属 id 时，**同一个归属的多条行怎么合成一个名次**。
#   * "best"    = 历史行为：名次最好的那条留下，同归属的后续行直接丢掉（等价于「只看该父块
#                 最靠前的那个 unit」）；
#   * "rrf_sum" = 聚合：前 `fold_window` 条原始行按 `1/(RRF_K + 原始名次)` 累加成一个分，再按
#                 分排序。同一个父块被多条 unit 命中时它会被**顶上去**，而不是只按它最好的
#                 那条 unit 的名次站队。
# 聚合的实测动机（2026-09-26，`tmp/v3_parent_agg_probe.py`；同一批 25 道题、只读重放、零 token）：
# v3 的 unit 级命中折回父块时，「只留最好名次」把「同一父块有 3 条 unit 都在前 20」这种强信号
# 压成了一条候选的名次——纯向量 top-8 只有 12/25；改成累加后 19/25、MRR@8 0.1840 → 0.5054。
# 探针里四个候选口径（分数求和 / RRF 求和 / 只取前 100 条 / ≥2 条命中优先）都落在 17–19/25，
# 方向一致，但**具体取哪个口径要在更大的格子上定案**，别把 19/25 当成定案数字。
FOLD_DEFAULT = "best"
FOLD_MODES = frozenset({"best", "rrf_sum"})

# 聚合折叠时参与打分的原始行数上限。探针里 100 条与全量 400 条在同一批题上只差 1 题
# （19/25 vs 18/25），属于噪声；100 条便宜 4 倍，所以取 100。`top_k` 比它还大时按 `top_k`
# 放大，否则返回条数会不够。
FOLD_WINDOW = 100


def load_id_map(path: str | Path) -> dict[str, str]:
    """读「索引行 id → 归属 id」的 JSON 映射（`scripts/build_unit_chunks.py` 落的那份）。

    这是映射的**唯一**读取口径：装配层（`akr.mcp_server.server`）、评测脚本
    （`scripts/retrieval_pool_depth.py` / `retrieval_ab.py`）都从这里进来，免得三处各写
    一份解析、各自对「什么算坏文件」有不同答案。

    缺文件或结构不对**必须报错**：静默退化成「不折叠」会让 unit 级的命中 id 落在 gold
    空间之外，整列数字会以「召回率为 0」的形状骗过所有检查。
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} 里没有 unit → 父块 的映射：先跑 scripts\\build_unit_chunks.py")
    return {str(key): str(value) for key, value in payload.items()}


class VectorStore:
    def __init__(
        self,
        path: str | Path,
        embed_fn: EmbedFn,
        *,
        index_text_fn: IndexTextFn | None = None,
        id_map: Mapping[str, str] | None = None,
        fold: str = FOLD_DEFAULT,
        fold_window: int = FOLD_WINDOW,
    ) -> None:
        """``index_text_fn`` 只改**送进嵌入模型的文本**，不改库里存的 ``documents``。

        2026-09-25 加这个钩子，是为了让「文档身份前缀」能进向量侧：前缀依赖 chunk 的
        ``doc_id`` / ``section_path``，只有拿到 ``Chunk`` 的调用方才拼得出来
        （``akr.ingest.doc_meta.make_index_text_fn``）。刻意**不**把带前缀的文本写进
        ``documents``——``query()`` 拿 ``documents`` 当命中原文回给模型，若前缀写进去，
        每一条证据都会多出一行「公司（代码）报告期」，污染引用文本。

        ``id_map`` 把**索引行的 id 折回它的归属 id**（`data/chroma-v3` 的 unit → 父块）。
        用途是让「细粒度索引」与「块级索引」在同一个口径下比对：unit 级库命中若干条同属
        一个父块的 unit 时，返回的 `chunk_id` 一律是父块 id，因此评测的 gold（父块 id）
        不必重造、RRF 也不会把同一个父块算成两支候选。默认 ``None`` = 索引 id 即命中 id，
        与历史行为逐字一致。

        ``fold`` 决定「同归属的多条行怎么合成一个名次」，取值在 `FOLD_MODES` 里（白名单，
        拼错一个字母必须报错，不能静默换一条检索路径）：

        * ``"best"``（默认）= 历史行为：名次最好的那条留下，后续同归属的行丢掉；
        * ``"rrf_sum"`` = 前 ``fold_window`` 条原始行按 ``1/(RRF_K + 名次)`` 累加后排序
          （见模块顶部 `FOLD_MODES` 的实测动机）。没有 ``id_map`` 时每个索引行各自成归属，
          累加分严格递减，因此退化成与 ``"best"`` 逐字相同的顺序——不是「关掉聚合就报错」。

        折叠会少返回行，所以 ``best`` 口径带 ``id_map`` 时 `query` 会**多取**候选直到凑够
        `top_k` 个不同归属（见 `_fetch_rows`）；多取只会往后补，不会重排已经取到的归属。
        """

        if fold not in FOLD_MODES:
            raise ValueError(f"未知的折叠口径：{fold!r}；可选 {sorted(FOLD_MODES)}")
        if fold_window < 1:
            raise ValueError(f"fold_window 必须是正整数，收到 {fold_window}")
        self._embed = embed_fn
        self._index_text: IndexTextFn = index_text_fn or (lambda chunk: chunk.text)
        self._id_map = dict(id_map) if id_map else None
        self._fold = fold
        self._fold_window = fold_window
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        index_texts = [self._index_text(chunk) for chunk in chunks]
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=self._embed(index_texts),
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "section_path": chunk.section_path,
                    "ordinal": chunk.ordinal,
                    "content_hash": chunk.content_hash,
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def query(
        self,
        text: str,
        *,
        top_k: int = 8,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievalHit]:
        if self.count() == 0:
            return []
        where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
        embedding = self._embed([text])
        if self._fold == "rrf_sum":
            # 聚合要的是「固定深度的原始名次表」，所以按行数取而不是按不同归属取：
            # 取到 100 个不同归属会变成几百行，把头部信号摊薄。
            rows = self._fetch_rows(
                embedding,
                where=where,
                want=max(self._fold_window, top_k),
                distinct_owners=False,
            )
            return self._aggregate(rows, top_k=top_k)
        rows = self._fetch_rows(embedding, where=where, want=top_k)
        return self._fold_best(rows, top_k=top_k)

    def _fold_best(
        self, rows: list[tuple[str, str | None, dict, float]], *, top_k: int
    ) -> list[RetrievalHit]:
        """历史口径：名次最好的那条留下，同归属的后续行丢掉。"""

        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for row in rows:
            owner = self._owner(row[0])
            if owner in seen:
                continue  # 同一归属的后续 unit：名次更好的那条已经占位
            seen.add(owner)
            hits.append(self._hit(owner, row))
            if len(hits) >= top_k:
                break
        return hits

    def _aggregate(
        self, rows: list[tuple[str, str | None, dict, float]], *, top_k: int
    ) -> list[RetrievalHit]:
        """聚合口径：前 `fold_window` 条原始行按 `1/(RRF_K + 名次)` 累加，按分排序。

        同分（只在「两条行名次相同」这种构造下出现）按该归属**最好名次**排，
        所以顺序完全由输入名次表决定，不依赖 dict 的插入序。

        命中的 `score` 与 `text` 仍取该归属**名次最好**的那条原始行：`score` 在两种折叠
        口径下必须是同一个语义（余弦相似度），排序口径的差别只体现在列表顺序上。
        """

        scores: dict[str, float] = {}
        best: dict[str, tuple[int, tuple[str, str | None, dict, float]]] = {}
        for rank, row in enumerate(rows, start=1):
            owner = self._owner(row[0])
            scores[owner] = scores.get(owner, 0.0) + 1.0 / (RRF_K + rank)
            best.setdefault(owner, (rank, row))
        ordered = sorted(scores, key=lambda owner: (-scores[owner], best[owner][0]))
        return [self._hit(owner, best[owner][1]) for owner in ordered[:top_k]]

    def _owner(self, chunk_id: str) -> str:
        """索引行的归属 id：没有 `id_map` 时索引 id 就是命中 id（历史行为）。"""

        return self._id_map.get(chunk_id, chunk_id) if self._id_map is not None else chunk_id

    @staticmethod
    def _hit(owner: str, row: tuple[str, str | None, dict, float]) -> RetrievalHit:
        """`(归属 id, 原始行)` → 命中：排序按归属走，回给模型的原文与分数跟着最好的那条行。"""

        _, document, metadata, distance = row
        return RetrievalHit(
            chunk_id=owner,
            doc_id=str(metadata["doc_id"]),
            source=str(metadata["source"]),
            text=document or "",
            score=1.0 - float(distance),
            retriever="vector",
        )

    def _fetch_rows(
        self,
        embedding: list[list[float]],
        *,
        where: dict | None,
        want: int,
        distinct_owners: bool = True,
    ) -> list[tuple[str, str | None, dict, float]]:
        """按距离升序取索引行，返回 `(索引 id, documents, metadata, distance)` 四元组。

        `distinct_owners=True`（默认）：带 `id_map` 时一直取到 `want` 条**不同归属**为止——
        `best` 折叠会丢掉同归属的后续行，不往后多取就凑不够返回条数。Chroma 的返回顺序
        就是距离升序，折叠只是丢行，所以「前 want 个不同归属」在本轮取多少都成立：多取只会
        往后补，不会把已经排好的往前挪。空库/过滤后行数不足时就是取完了，返回手里这些。

        `distinct_owners=False`：只取 `want` 条原始行——`rrf_sum` 聚合吃的是「固定深度的
        名次表」，按不同归属去凑会把它撑成几百行，反而摊薄头部信号。
        """

        total = self.count()
        limit = min(max(want, 1), total)
        while True:
            rows = self._query_rows(embedding, where=where, limit=limit)
            if (
                not distinct_owners
                or self._id_map is None
                or len(rows) < limit
                or limit >= total
            ):
                return rows
            if len({self._owner(row[0]) for row in rows}) >= want:
                return rows
            limit = min(limit * 2, total)

    def _query_rows(
        self, embedding: list[list[float]], *, where: dict | None, limit: int
    ) -> list[tuple[str, str | None, dict, float]]:
        """Chroma 查询的薄封装：只负责把列式返回转成行式四元组。"""

        result = self._collection.query(
            query_embeddings=embedding,
            n_results=limit,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        return [
            (chunk_id, document, metadata, float(distance))
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def delete_by_doc(self, doc_id: str) -> int:
        ids = list(self._collection.get(where={"doc_id": doc_id}, include=[])["ids"])
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def delete_ids(self, chunk_ids: Sequence[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self._collection.count()
