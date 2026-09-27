from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pdfplumber

_NEWLINE = re.compile(r"\r\n?")
_SPACES = re.compile(r"[ \t\u3000]+")


class ParseError(RuntimeError):
    """PDF 缺失、损坏或无法解析。禁止静默返回空结果。"""


class PageLike(Protocol):
    def extract_text(self) -> str | None: ...

    def extract_tables(self) -> list[list[list[str | None]]]: ...


@dataclass(frozen=True)
class ParsedPage:
    page: int
    text: str
    tables: list[list[list[str]]]


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    unified = _NEWLINE.sub("\n", text)
    lines = [_SPACES.sub(" ", line).strip() for line in unified.split("\n")]
    return "\n".join(line for line in lines).strip()


def _clean_tables(raw: Iterable[list[list[str | None]]]) -> list[list[list[str]]]:
    return [
        [[cell if cell is not None else "" for cell in row] for row in table]
        for table in raw
    ]


def pages_from_pages(pages: Iterable[PageLike], *, start_page: int = 1) -> list[ParsedPage]:
    return [
        ParsedPage(
            page=start_page + offset,
            text=_normalize(page.extract_text()),
            tables=_clean_tables(page.extract_tables() or []),
        )
        for offset, page in enumerate(pages)
    ]


def parse_pdf(path: str | Path) -> list[ParsedPage]:
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise ParseError(f"PDF 文件不存在：{pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as doc:
            return pages_from_pages(doc.pages)
    except ParseError:
        raise
    except Exception as exc:
        # pdfplumber 底层的异常类型不稳定，统一转成显式错误并保留原因链
        raise ParseError(f"解析失败：{pdf_path}：{exc}") from exc


def has_text_layer(pages: Sequence[ParsedPage], *, min_chars: int = 200) -> bool:
    """扫描件页面上没有文本层：这类文档需要走 spec §7.6 的原生视觉通道。"""
    return sum(len(page.text) for page in pages) >= min_chars
