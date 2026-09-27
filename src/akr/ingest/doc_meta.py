"""文档级身份元数据：把「这条 chunk 属于谁、什么时候、哪份报告」变成可检索的文本与可过滤的字段。

动机（2026-09-25 只读实测）：25 道可答题的 **21 个唯一 gold chunk 里 0 个含证券代码**、
只有 8 个含公司名；语料 6176 个 chunk 里正文含证券代码的只有 36 个。而问句几乎都点名
「公司 + 证券代码 + 年份」——查询里最具区分度的 token 在目标 chunk 里根本不存在，
BM25 只能靠「硅片 / 净利润」这类全行业通用词去和另外四家同业年报的近重复段落硬碰。
这就是「文档级召回 25/25、chunk 级 top-8 只有 10/25」的直接原因：**文档有身份，块没有**。

本模块只做一件事：把 manifest 里的身份拼成**检索前缀**，交给 `Bm25Index` 在**建索引时**
拼接。`Chunk.text` 一律不动——artifact 里的原文、回给模型的片段都是同一份原始正文，
检索前缀是纯派生量（`prefix` 也可单独取用，便于脚本复现与断言）。

向量侧自 2026-09-25 起**可选**地复用同一个 `index_text`（`make_index_text_fn` →
`VectorStore(index_text_fn=...)`）：它只改「送进嵌入模型的文本」，Chroma 里存的 `documents`
仍是原文，所以命中回给模型的片段不会多出一行前缀。该开关默认关闭，因为加前缀会挤占
`MAX_INPUT_CHARS=500` 的预算（实测 3588/6176 个 chunk 原文超 500 字符），
收益必须由 A/B 实测决定，不能假设。

事实来源是 `data/manifest.json`（`scripts/fetch_corpus.py` 产出）：15 份文档的
`company` / `code` / `period` / `announcement_title` 全都齐备，不需要从 doc_id 猜。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from akr.ingest.chunker import Chunk

_YEAR = re.compile(r"(?:19|20)\d{2}")


@dataclass(frozen=True)
class DocMeta:
    """一份文档的身份：谁（公司 / 证券代码）+ 什么时候（报告期）+ 是什么（报告标题）。"""

    doc_id: str
    company: str = ""
    code: str = ""
    period: str = ""
    title: str = ""

    @property
    def years(self) -> tuple[str, ...]:
        """报告期里出现的四位年份（`2024年年度` → `('2024',)`）。"""

        return tuple(_YEAR.findall(self.period))

    def prefix(self, section_path: str = "") -> str:
        """检索前缀：`TCL中环（002129） 2024年年度 2024年年度报告 第四节 董事会报告`。

        章节路径由 chunk 提供（它是块级信息），公司 / 代码 / 报告期 / 标题由文档提供。
        顺序刻意是「身份 → 报告 → 章节」：身份 token 放在文本最前，配合 BM25 的
        文档长度归一化，同业年报里那些措辞雷同的段落才不会互相盖过。
        """

        identity = f"{self.company}（{self.code}）" if self.code else self.company
        parts = [identity, self.period, self.title, section_path]
        return " ".join(part for part in parts if part)


def load_doc_meta(path: str | Path) -> dict[str, DocMeta]:
    """读 manifest 的 `documents` 数组，按 doc_id 建索引；结构不对立刻报错。

    与 `scripts/ingest_corpus.load_documents` 同一条口径：不做静默兜底。缺文件的场景
    （例如只跑单测的临时目录）由调用方决定怎么降级——本函数只负责「有就解析对」。
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("documents") if isinstance(payload, dict) else None
    if not records:
        raise ValueError(f"{path} 里没有 documents 数组")

    metas: dict[str, DocMeta] = {}
    for record in records:
        doc_id = str(record.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(f"{path} 里有缺 doc_id 的记录：{record!r}")
        metas[doc_id] = DocMeta(
            doc_id=doc_id,
            company=str(record.get("company") or "").strip(),
            code=str(record.get("code") or "").strip(),
            period=str(record.get("period") or "").strip(),
            title=str(record.get("announcement_title") or "").strip(),
        )
    return metas


def load_doc_meta_or_empty(path: str | Path | None) -> dict[str, DocMeta]:
    """容错版：路径为空或文件不存在时返回空表（检索退化为只用原文，不报错）。

    用于服务端与评测入口——缺 manifest 只是「没有身份前缀」，不该让整个工具进程起不来。
    """

    if path is None:
        return {}
    if not Path(path).exists():
        return {}
    return load_doc_meta(path)


def index_text(chunk: Chunk, meta: DocMeta | None) -> str:
    """BM25 的索引文本 = 身份前缀 + 换行 + 原文；没有元数据时就是原文本身。"""

    if meta is None:
        return chunk.text
    header = meta.prefix(chunk.section_path)
    return f"{header}\n{chunk.text}" if header else chunk.text


def make_index_text_fn(
    metas: Mapping[str, DocMeta],
) -> Callable[[Chunk], str]:
    """`Chunk → 索引文本` 的可调用对象：给 `VectorStore(index_text_fn=...)` 用。

    返回的是**闭包而不是查表结果**，因为 `VectorStore.add` 每批都要按当前 chunk 现算；
    缺 meta 的 chunk 退化成原文（与 BM25 同口径：没有身份就只按正文检索）。
    """

    def render(chunk: Chunk) -> str:
        return index_text(chunk, metas.get(chunk.doc_id))

    return render


def doc_headers(metas: Mapping[str, DocMeta]) -> dict[str, str]:
    """`doc_id → 文档级前缀`（不含章节）。给「只有文档、没有 chunk」的场景用。"""

    return {doc_id: meta.prefix() for doc_id, meta in metas.items() if meta.prefix()}
