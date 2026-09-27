from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import jieba
from rank_bm25 import BM25L

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import DocMeta, index_text
from akr.storage.hits import RetrievalHit

_STOPWORDS = frozenset(
    {"的", "了", "和", "与", "及", "是", "在", "对", "为", "有", "等", "其中", "同比", "较"}
)

# jieba 会把 `-9,818,364,927.98` 切成 `9` / `818` / `364` / `927.98`，把 `1,234 万元`
# 切成 `1` / `234`：这些「数字碎片」没有语义（既不是年份、也不是比率、更不是金额），
# 却各自进 BM25 的词频与文档长度归一化，等于往语料里灌噪声——碎片越多，文档越长，
# 长度归一化把真正命中的行越压越平。判据刻意只管**纯整数且 ≤3 位**：`927.98` / `0.35`
# 这类比率、`2024` / `002129` 这类年份与公司代码全部保留。
# 2026-09-26 的 A/B（`tmp/probe_tokenizer_variants.py`，BM25 单路、25 可答题、语料
# `data/chunks.jsonl`）：只丢这一类，top-8 命中 14/25 → 16/25、recall@8 44.0% → 48.0%、
# MRR 0.3413 → 0.3980，且 recall@50 一条不动（84.0%）——**是排序收益不是召回收益**；
# 再激进一档（连小数点碎片一起丢）指标逐格相同，语料 token 反而更少，所以保守取这一档。
_NUMERIC_FRAGMENT = re.compile(r"[+-]?\d{1,3}\Z")

# 年报领域词表：jieba 默认词典把「营业收入」切成「营业 / 收入」，
# 「营业成本」与「营业收入」因此共享「营业」，排序精度受损。
# 这里不做全局 add_word，而是在分词结果上额外「回并」出短语，避免污染 jieba 的全局状态。
_DOMAIN_TERMS = frozenset(
    {
        "营业收入",
        "营业成本",
        "营业利润",
        "净利润",
        "毛利率",
        "净利率",
        "资产负债率",
        "经营活动",
        "投资活动",
        "筹资活动",
        "现金流量",
        "应收账款",
        "存货周转率",
        "股东权益",
        "每股收益",
        "研发投入",
        "员工人数",
        "主营业务",
    }
)
_MAX_TERM_CHARS = 6


def _merge_domain_terms(tokens: Sequence[str]) -> list[str]:
    """把被分词器切碎的领域词按最长优先回并成短语 token（只返回合并结果）。"""
    phrases: list[str] = []
    index = 0
    while index < len(tokens):
        for size in range(_MAX_TERM_CHARS, 1, -1):
            if index + size <= len(tokens):
                candidate = "".join(tokens[index : index + size])
                if candidate in _DOMAIN_TERMS:
                    phrases.append(candidate)
                    index += size
                    break
        else:
            index += 1
    return phrases


def tokenize_zh(text: str) -> list[str]:
    """中文分词 + 归一化：丢弃空白、单字符虚词与停用词，保留细粒度片段并追加领域短语。

    片段与短语刻意共存，缺一不可：
    - 只留短语：查询「员工」永远匹配不到 chunk 里的「员工人数」，召回直接归零；
    - 只留片段：「营业成本」与「营业收入」共享「营业」，排序无法区分。
    两个 token 各自贡献一次词频，属于可接受的轻微重复。

    数字侧只丢**纯整数碎片**（`_NUMERIC_FRAGMENT`，理由见那一行的实测）：千分位把
    一个金额切成好几段 token 时，这些段既不表达量纲也不表达指标，留着只会摊薄 idf。
    """
    pieces = [
        token
        for token in (raw.strip().lower() for raw in jieba.lcut(text))
        if token
        and not (len(token) == 1 and not token.isalnum())
        and token not in _STOPWORDS
        and not _NUMERIC_FRAGMENT.match(token)
    ]
    return pieces + _merge_domain_terms(pieces)


class Bm25Index:
    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        doc_metas: Mapping[str, DocMeta] | None = None,
        index_text_fn: Callable[[Chunk], str] | None = None,
    ) -> None:
        self._k1 = k1
        self._b = b
        # 文档身份只影响**索引文本**，不影响返回给模型的 `text`：artifact / 引用展示 /
        # 向量侧嵌入拿到的都是 `Chunk.text` 原文（见 `akr.ingest.doc_meta` 的动机说明）。
        self._doc_metas: dict[str, DocMeta] = dict(doc_metas or {})
        # 索引文本口径是**装配层**旋钮（见 `akr.ingest.table_context` 的 docstring）：
        # 传了就用它，没传才回落到「身份前缀 + 原文」这条历史行为。
        # 存**绑定方法**而不是当初算好的字符串：`set_doc_metas` 改表后要立刻生效。
        self._index_text: Callable[[Chunk], str] = index_text_fn or self._default_index_text
        self._chunks: dict[str, Chunk] = {}
        self._matrix: BM25L | None = None

    def _default_index_text(self, chunk: Chunk) -> str:
        """默认口径：身份前缀 + 原文（该文档没有身份时就是原文本身）。"""

        return index_text(chunk, self._doc_metas.get(chunk.doc_id))

    def set_doc_metas(self, doc_metas: Mapping[str, DocMeta]) -> None:
        """整体替换文档身份表；表变了必须重建打分矩阵。"""

        self._doc_metas = dict(doc_metas)
        self._matrix = None

    def index_text_of(self, chunk: Chunk) -> str:
        """该 chunk 参与打分时用的文本；口径由装配层决定（默认 = 身份前缀 + 原文）。"""

        return self._index_text(chunk)

    def _ordered(self) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in sorted(self._chunks)]

    def _ensure_matrix(self) -> BM25L | None:
        if self._matrix is None and self._chunks:
            corpus = [tokenize_zh(self.index_text_of(chunk)) or ["_"] for chunk in self._ordered()]
            # 用 BM25L 而非 BM25Okapi：后者的 idf = log(N-df+0.5)-log(df+0.5) 在 df = N/2 时
            # 恰为 0，十几份年报的小语料下高频词会被静默清零、检索空手而归；
            # BM25L 的 idf = log(N+1)-log(df+0.5) 恒为正，且未命中文档的分数仍为 0。
            self._matrix = BM25L(corpus, k1=self._k1, b=self._b)
        return self._matrix

    def build(self, chunks: Sequence[Chunk]) -> int:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._matrix = None
        return len(self._chunks)

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._matrix = None
        return len(self._chunks)

    def remove(self, chunk_ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if self._chunks.pop(chunk_id, None) is not None:
                removed += 1
        if removed:
            self._matrix = None
        return removed

    def count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        doc_ids: Sequence[str] | None = None,
    ) -> list[RetrievalHit]:
        matrix = self._ensure_matrix()
        tokens = tokenize_zh(query)
        if matrix is None or not tokens:
            return []
        allowed = set(doc_ids) if doc_ids else None
        scored = list(zip(self._ordered(), matrix.get_scores(tokens)))
        ranked = [
            (chunk, float(score))
            for chunk, score in scored
            if score > 0 and (allowed is None or chunk.doc_id in allowed)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                text=chunk.text,
                score=score,
                retriever="bm25",
            )
            for chunk, score in ranked[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        # payload 永远只有这三键：`doc_metas` 与 `index_text_fn` 都是**装配期的派生量**，
        # 落盘会让 bm25.json 出现第二份事实来源（改一次 manifest 就得重灌数据）。
        payload = {
            "k1": self._k1,
            "b": self._b,
            "chunks": [asdict(chunk) for chunk in self._ordered()],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        doc_metas: Mapping[str, DocMeta] | None = None,
        index_text_fn: Callable[[Chunk], str] | None = None,
    ) -> "Bm25Index":
        """从落盘的分块语料恢复索引。

        `doc_metas` 与 `index_text_fn` 都不落盘：身份与索引文本口径都是派生量，落盘会让
        bm25.json 出现第二份事实来源，改一次 manifest 就得重灌数据。传进来即生效（服务端由
        `build_tool_context` 装配），不传就是「只用原文」的历史行为。
        """

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(
            k1=payload["k1"],
            b=payload["b"],
            doc_metas=doc_metas,
            index_text_fn=index_text_fn,
        )
        index.build([Chunk(**item) for item in payload["chunks"]])
        return index
