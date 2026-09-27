from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from akr.ingest.parser import ParsedPage

_CHAPTER = re.compile(r"^(第[一二三四五六七八九十百零〇]+[章节])\s*(.*)$")
_CN_NUMBER = re.compile(r"^([一二三四五六七八九十]+)、\s*(.*)$")
_PAREN = re.compile(r"^[（(]([一二三四五六七八九十]+)[）)]\s*(.*)$")
_DOTTED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})[、\s]+(.*)$")

MAX_HEADING_CHARS = 40


@dataclass(frozen=True)
class Heading:
    level: int
    title: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    ordinal: int
    text: str
    content_hash: str
    section_path: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Span:
    """文档流里的一段连续文本（通常是整行，超长行被硬切成若干段）。

    `start` / `end` 是**文档流**偏移（跳过空行、行间距算 1），不是页内偏移。
    """

    text: str
    section: str
    page: int
    start: int
    end: int


@dataclass(frozen=True)
class ParentBlock:
    """父块：`chunk_pages` 的中间形态，也是 unit 级切块（`akr.ingest.unit_chunks`）的输入。

    抽出来是为了让「怎么分组」只有一份实现：父块的 `block_id` 与 `chunk_pages` 产出的
    `Chunk.chunk_id` **逐字相同**，所以 unit 命中可以无歧义地折回父块（`tests/ingest/
    test_unit_chunks.py` 用同一条断言锁住这件事）。`spans` 是分组后的原始片段，
    unit 切块只在**这一层内部**再切，不会跨父块边界。
    """

    block_id: str
    content_hash: str
    text: str
    section_path: str
    ordinal: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    spans: tuple[Span, ...]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def detect_heading(line: str) -> Heading | None:
    """识别年报里的章节标题。长度阈值用于排除正文中的编号句子。"""
    if len(line) > MAX_HEADING_CHARS:
        return None
    if match := _CHAPTER.match(line):
        return Heading(1, f"{match.group(1)} {match.group(2)}".strip())
    if match := _CN_NUMBER.match(line):
        return Heading(2, f"{match.group(1)}、{match.group(2)}".strip())
    if match := _PAREN.match(line):
        return Heading(3, f"（{match.group(1)}）{match.group(2)}".strip())
    if match := _DOTTED.match(line):
        level = match.group(1).count(".") + 1
        return Heading(level, f"{match.group(1)} {match.group(2)}".strip())
    return None


def _hard_split(line: str, max_chars: int) -> list[str]:
    if len(line) <= max_chars:
        return [line]
    return [line[i : i + max_chars] for i in range(0, len(line), max_chars)]


def _spans(
    pages: Sequence[ParsedPage], stack: tuple[str, ...], *, max_chars: int
) -> list[Span]:
    """把页面文本摊平成带章节、页码与文档流偏移的片段序列。"""
    spans: list[Span] = []
    offset = 0
    for page in pages:
        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            heading = detect_heading(line)
            if heading is not None:
                stack = (*stack[: max(heading.level - 1, 0)], heading.title)
            section = " / ".join(stack)
            cursor = offset
            for piece in _hard_split(line, max_chars):
                spans.append(Span(piece, section, page.page, cursor, cursor + len(piece)))
                cursor += len(piece)
            offset += len(line) + 1  # 行间换行计入文档流长度
    return spans


def _merge(spans: Sequence[Span], *, max_chars: int, overlap: int) -> list[list[Span]]:
    groups: list[list[Span]] = []
    current: list[Span] = []
    pending_tail: Span | None = None
    for span in spans:
        if current:
            same_section = current[-1].section == span.section
            # 预算按「每行字符数 + 行尾换行」计。用 >= 而非 >：预算恰好用满时即断块，
            # 否则 `max_chars=10` 下的两行 4 字文本会被合并，重复文本拿不到 `#2` 后缀。
            projected = sum(len(item.text) + 1 for item in current) + len(span.text) + 1
            if not same_section or projected >= max_chars:
                groups.append(current)
                text = "\n".join(item.text for item in current)
                tail = text[-overlap:] if overlap else ""
                last = current[-1]
                pending_tail = (
                    Span(tail, span.section, last.page, last.end - len(tail), last.end)
                    if tail
                    else None
                )
                current = []
        if not current and pending_tail is not None:
            tail = pending_tail
            pending_tail = None
            # 重叠尾巴无条件带入下一块：块预算只约束「本块新增内容」，
            # 否则 overlap=100/max_chars=800 这类配置的重叠永远不会生效。
            current = [tail]
        current.append(span)
    if current:
        groups.append(current)
    return groups


def parent_blocks(
    pages: Sequence[ParsedPage],
    *,
    doc_id: str,
    max_chars: int = 800,
    overlap: int = 100,
    section_path: str = "",
) -> list[ParentBlock]:
    """把页面切成父块（带内容寻址的 `block_id` 与它覆盖的 `spans`）。

    这是分块的**唯一**口径：`chunk_pages` 只是给每个父块补上 `source` 并转成 `Chunk`。
    参数校验也放在这里，`chunk_pages` 不再重复一遍。
    """

    if max_chars <= 0:
        raise ValueError("max_chars 必须为正整数")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap 必须满足 0 <= overlap < max_chars")

    stack = tuple(part for part in section_path.split(" / ") if part)
    groups = _merge(_spans(pages, stack, max_chars=max_chars), max_chars=max_chars, overlap=overlap)

    blocks: list[ParentBlock] = []
    seen: dict[str, int] = {}
    for group in groups:
        text = "\n".join(span.text for span in group).strip()
        if not text:
            continue
        digest = content_hash(text)
        seen[digest] = seen.get(digest, 0) + 1
        suffix = "" if seen[digest] == 1 else f"#{seen[digest]}"
        blocks.append(
            ParentBlock(
                block_id=f"{doc_id}#c{digest}{suffix}",
                content_hash=digest,
                text=text,
                section_path=group[0].section,
                ordinal=len(blocks),
                page_start=group[0].page,
                page_end=group[-1].page,
                char_start=group[0].start,
                char_end=group[-1].end,
                spans=tuple(group),
            )
        )
    return blocks


def chunk_pages(
    pages: Sequence[ParsedPage],
    *,
    doc_id: str,
    source: str,
    max_chars: int = 800,
    overlap: int = 100,
    section_path: str = "",
) -> list[Chunk]:
    """页面 → 生产分块：逐字等于「`parent_blocks` 的每个块套上一个 `Chunk`」。"""

    return [
        Chunk(
            chunk_id=block.block_id,
            doc_id=doc_id,
            source=source,
            ordinal=block.ordinal,
            text=block.text,
            content_hash=block.content_hash,
            section_path=block.section_path,
            page_start=block.page_start,
            page_end=block.page_end,
            char_start=block.char_start,
            char_end=block.char_end,
        )
        for block in parent_blocks(
            pages,
            doc_id=doc_id,
            max_chars=max_chars,
            overlap=overlap,
            section_path=section_path,
        )
    ]
