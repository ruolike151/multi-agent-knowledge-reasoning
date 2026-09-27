"""unit 级切块：把 800 字父块再切成「行边界上的小块」，专治表格块的截断与粒度。

动机（2026-09-25 只读实测，内部偏差日志与内部探针脚本）：

* **截断**：`akr.storage.embeddings.ApiEncoder.encode` 是硬截断 `text[:MAX_INPUT_CHARS]`
  （500 字符），语料 6176 个父块里 3588 个（58.1%）的正文有一部分**从未进过嵌入**，
  被截掉的尾巴上正是答案（`…归属于上市公司股东的净利润 -6,419,556,843.85` 落在第 500 字之后）。
* **粒度**：gold 不是找不到，是被埋在 800 字的大块里——BM25 受控探针（零 token）里，
  同样的问句在 800 字块上 top-8 命中 6/25，换成 400 字 unit 后 9/25，单行 unit 12/25；
  而「父块级有没有」在细粒度下是 25/25。
* **表格扁平化**：16.53% 的父块至少有一处「行标签被数字行劈开」（`经营活动产生的现金流量
  净额` 整串只在 75 个块里、断头版 `经营活动产生的现金` 在 115 个块里），23.6% 的表格块
  以纯数字行开头。

所以 unit 切块做三件事：**按行切**（不切在指标名与数值之间）、**把索引文本压进嵌入窗口**
（宁可切小，也不让尾巴被截掉）、**继承父块的导读行**（`table_lead` 抽出的「期间行 + 指标行」
排在 unit 原文之前，否则它会被同一个 500 字符预算截掉）。

三条刻意的纪律：

* **不碰生产分块器**：`chunk_pages` 的默认值、`chunk_id` 口径一律不变，本模块是新的一层，
  「新参数默认值 = 历史行为」这条全局约束因此不破。
* **unit id 独立命名空间**：`{doc_id}#u{digest}`，与父块的 `#c{digest}` 天然不冲突，
  两份索引可以共处一个 Chroma 集合、也可以各写一份（v3 库走后者）。
* **命中可折回父块**：每个 unit 都带 `parent_id`（`chunk_pages` 会给出的那个 id），
  评测因此不必重造 gold——命中 unit 就等于命中它所属的父块。

`index_text` 与 BM25 侧（`akr.ingest.table_context`）**同一份拼装口径**：身份前缀 → 导读行
→ 原文。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from akr.ingest.chunker import (
    Chunk,
    Span,
    _hard_split,
    content_hash,
    parent_blocks,
)
from akr.ingest.doc_meta import DocMeta
from akr.ingest.parser import ParsedPage
from akr.ingest.table_context import compose_index_text, table_lead

# 嵌入侧单条输入上限（`akr.storage.embeddings.MAX_INPUT_CHARS` 的副本）。
# 为什么复制而不 import：ingest 层不该依赖 storage 层（那会把 chromadb 拖进解析路径）。
# 两份常量的一致性由 `tests/ingest/test_unit_chunks.py` 的等值断言锁住。
EMBED_INPUT_CHARS = 500

# 表格块的目标 unit 大小。表格行短、指标名与数值紧邻，切小一点才能让「净利润 -64.19 亿」
# 这类行自己占一个可检索单元。
TABLE_UNIT_CHARS = 400
# 正文块的目标大小＝父块大小（800）：正文是连续叙述，切碎只会丢上下文。两者最终都会被
# 嵌入窗口（`EMBED_INPUT_CHARS` 减去前缀与导读行）反推收紧，见 `unit_window`。
PROSE_UNIT_CHARS = 800
# unit 下限：前缀 + 导读行最长能吃掉 400 字符预算，但 50 字的碎片没有任何检索价值。
# 触到下限时那一条会超预算（尾巴被嵌入截断）——这是刻意的取舍，不是漏算。
MIN_UNIT_CHARS = 120


@dataclass(frozen=True)
class UnitChunk:
    """一个 unit：`chunk` 是它自己的行块，`parent_id` 是它折回去的父块，`index_text` 是索引文本。"""

    chunk: Chunk
    parent_id: str
    index_text: str

    def __len__(self) -> int:  # 便于测试里直接量「这条索引文本多长」
        return len(self.index_text)


def unit_window(
    *,
    header: str,
    lead: str,
    target: int,
    min_chars: int = MIN_UNIT_CHARS,
    embed_chars: int = EMBED_INPUT_CHARS,
) -> int:
    """按嵌入窗口反推 unit 的目标大小：`min(target, 预算)`，再夹到 `min_chars` 以上。

    预算 = `embed_chars` 减去「身份前缀 + 导读行」占掉的字符（含分隔换行）。预算为负
    （前缀本身就超窗口）时取 `min_chars`：切得再小也塞不进去，此时保住可读的下限更有用。
    """

    overhead = len(compose_index_text(header=header, lead=lead, text=""))
    return max(min_chars, min(target, embed_chars - overhead))


def _slice_spans(spans: Sequence[Span], *, window: int) -> list[list[Span]]:
    """把父块的 span 序列按 `window` 聚成若干组，组边界只落在**行边界**上。

    单行超过 `window` 时用 `chunker._hard_split` 硬切（与父块分块器同一套兜底），
    所以每个 unit 的原文都不会超过 `window`。
    """

    groups: list[list[Span]] = []
    current: list[Span] = []
    used = 0
    for span in spans:
        for piece in _hard_split(span.text, window):
            cost = len(piece) + 1  # 行尾换行与父块口径一致
            if current and used + cost >= window:
                groups.append(current)
                current, used = [], 0
            current.append(span if piece == span.text else replace(span, text=piece))
            used += cost
    if current:
        groups.append(current)
    return groups


def unit_chunks(
    pages: Sequence[ParsedPage],
    *,
    doc_id: str,
    source: str,
    meta: DocMeta | None = None,
    max_chars: int = 800,
    overlap: int = 100,
    table_chars: int = TABLE_UNIT_CHARS,
    prose_chars: int = PROSE_UNIT_CHARS,
    min_chars: int = MIN_UNIT_CHARS,
    embed_chars: int = EMBED_INPUT_CHARS,
    section_path: str = "",
) -> list[UnitChunk]:
    """一份文档 → 全部 unit（按父块顺序、父块内按行顺序）。

    切法是「父块 → 行块」，因此 unit 一定落在父块覆盖的文本内：父块自身用
    `chunker.parent_blocks` 现算，与生产分块器共用分组口径（父块 id 逐字相同）。
    """

    blocks = parent_blocks(
        pages,
        doc_id=doc_id,
        max_chars=max_chars,
        overlap=overlap,
        section_path=section_path,
    )
    header_of = (lambda section: meta.prefix(section)) if meta is not None else (lambda _: "")

    units: list[UnitChunk] = []
    seen: dict[str, int] = {}
    for block in blocks:
        # 导读行取自**父块**：unit 自己可能整段都不含期间行/数值行，而它存在的意义正是把
        # 「指标名 + 期间 + 数值」的线索带到每一个单元里。
        lead = table_lead(block.text)
        header = header_of(block.section_path)
        window = unit_window(
            header=header,
            lead=lead,
            target=table_chars if lead else prose_chars,
            min_chars=min_chars,
            embed_chars=embed_chars,
        )
        for group in _slice_spans(block.spans, window=window):
            text = "\n".join(span.text for span in group).strip()
            if not text:
                continue
            digest = content_hash(text)
            seen[digest] = seen.get(digest, 0) + 1
            suffix = "" if seen[digest] == 1 else f"#{seen[digest]}"
            units.append(
                UnitChunk(
                    chunk=Chunk(
                        chunk_id=f"{doc_id}#u{digest}{suffix}",
                        doc_id=doc_id,
                        source=source,
                        ordinal=len(units),
                        text=text,
                        content_hash=digest,
                        section_path=group[0].section,
                        page_start=group[0].page,
                        page_end=group[-1].page,
                        char_start=group[0].start,
                        char_end=group[-1].end,
                    ),
                    parent_id=block.block_id,
                    index_text=compose_index_text(header=header, lead=lead, text=text),
                )
            )
    return units
