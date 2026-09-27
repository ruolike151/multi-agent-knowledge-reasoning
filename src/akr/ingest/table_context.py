"""表格导读行：把年报表格块里「指标名 + 期间 + 数值」的行压成前几行，供索引文本前置。

动机（2026-09-25 只读实测，`scripts/retrieval_ab.py` 的 `+table` 三列）：年报的财务表格经
PDF 抽取后是「一列一个 token 的散行」，指标名常常被排版切成两行——
`归属于上市公司股东` / `-9,818,364,927.98 …` / `的净利润（元）`——问句里最关键的
「净利润」「同比增减」要么落在离数值很远的那一行，要么根本不在表格行上。`table_lead`
把这类行**原样摘出来**，按「期间行 / 数值行」的顺序拼成一段最多 `TABLE_LEAD_LINES` 行的
导读（真实形状见 `tests/ingest/test_table_context.py` 的 `REAL_TABLE_EXCERPT`）。

它只改**索引文本**（`table_index_text`）：`Chunk.text`、artifact 原文、回给模型的
`SearchResult.hits[].text` 一律不动——与 `akr.ingest.doc_meta` 的身份前缀同一条纪律。

两个刻意的设计选择：

* **前插**：索引文本是「身份前缀 → 导读行 → 原文」。向量侧的截断是
  `text[:MAX_INPUT_CHARS]`（`akr.storage.embeddings:208`），语料里多数块的原文本来就超了
  这个预算，挂在末尾的导读行会被整段截掉；放在前面才能在 Task 7 重嵌时为向量侧所用。
  BM25 是词袋、不看顺序，前插对它无害。
* **正文块恒等**：`table_lead` 在「没有表格行」的块上返回空串，此时 `table_index_text`
  返回的就是 `index_text(chunk, meta)` 本身——对大多数 chunk 是恒等变换，
  「新参数默认值 = 历史行为」这条全局约束因此不破。

`index_text_fn` 是**装配层**旋钮（`Bm25Index(index_text_fn=...)`），不是工具层参数，
所以它不进 `TOOL_SCHEMAS`：同一份库用两种索引文本打分会让结果无法复现。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Mapping

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import DocMeta, index_text

# 导读行最多几行：3 行刚好够「期间行 + 前两条指标」，再多就是给 prompt 与嵌入预算加压。
TABLE_LEAD_LINES = 3
# 人类可读的分隔符。`tokenize_zh` 会丢空白、也会丢单字符非字母数字 token（含 `|`），
# 所以它只影响日志可读性，不产生任何检索 token。
_LEAD_SEPARATOR = " | "
# 指标名被排版切开后的碎片行：单格、够短、不含数字（`归属于上市公司股东`、`的净利润（元）`）。
# 12 个字符的上界把「六、主要会计数据和财务指标」（13 字）这类**章节标题**挡在外面。
_NAME_FRAGMENT_MAX_CHARS = 12
# 「是数字」的最小判据：可带千分位、小数点、正负号、百分号。
_NUMBER = re.compile(r"[-+]?[\d,]+(?:\.\d+)?%?")
# 期间行里的格子：`2024年` / `2024年末`。
_PERIOD_CELL = re.compile(r"^\d{4}年(?:末)?$")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _cells(line: str) -> list[str]:
    return line.split()


def _is_number(token: str) -> bool:
    return bool(_NUMBER.fullmatch(token))


def _is_numeric_row(cells: list[str]) -> bool:
    """数值行：格子 ≥3、数字格子 ≥2、且非数字格子 ≤1。

    「非数字格子 ≤1」是关键那一条：正文句子按空格拆开也会出现好几个纯数字
    （`根据中国证券监督管理委员会于 2021 年 11 月 9 日签发的证监许可` 拆出 7 格、3 格是数字），
    只要求「≥2 个数字格子」就会把这些句子误判成表格行。
    """

    numeric = sum(1 for token in cells if _is_number(token))
    return len(cells) >= 3 and numeric >= 2 and numeric >= len(cells) - 1


def _is_period_row(cells: list[str]) -> bool:
    """期间行：≥2 个格子形如 `2024年`/`2024年末`（`2024年 2023年 本年比上年增减 2022年`）。"""

    return len(cells) >= 2 and sum(1 for token in cells if _PERIOD_CELL.fullmatch(token)) >= 2


def _is_name_fragment(line: str, cells: list[str]) -> bool:
    """指标名被排版切开后的碎片：单格、够短、不含数字（含数字的碎片是数值行）。"""

    return (
        len(cells) == 1
        and len(line) <= _NAME_FRAGMENT_MAX_CHARS
        and not any(char.isdigit() for char in line)
    )


def table_lead(text: str, *, max_lines: int = TABLE_LEAD_LINES) -> str:
    """表格块的导读行：按「期间行 / 数值行」摘出前 `max_lines` 行，用 ` | ` 连成一段。

    紧贴数值行上/下的一行短碎片会被拼进同一行（`归属于上市公司股东` + 数值行 +
    `的净利润（元）` → `归属于上市公司股东的净利润（元） …`）。只有**行内没有汉字**的
    数值行才吃「行尾碎片」——行内已经有汉字的行，指标名已经在这行上了，再接一行就是把
    下一行的名字偷过来。没有表格行的块返回空串。
    """

    rows: list[str] = []
    pending = ""
    lines = [line.strip() for line in text.splitlines()]
    index = 0
    while index < len(lines) and len(rows) < max_lines:
        line = lines[index]
        cells = _cells(line)
        if not cells:
            pending = ""
        elif _is_period_row(cells) or _is_numeric_row(cells):
            name = pending
            if not _CJK.search(line) and index + 1 < len(lines):
                tail = lines[index + 1]
                if _is_name_fragment(tail, _cells(tail)):
                    name = f"{name}{tail}" if name else tail
                    index += 1
            rows.append(f"{name} {line}" if name else line)
            pending = ""
        elif _is_name_fragment(line, cells):
            pending = line
        else:
            pending = ""
        index += 1
    return _LEAD_SEPARATOR.join(rows)


def table_index_text(chunk: Chunk, meta: DocMeta | None) -> str:
    """索引文本 = 身份前缀 → 导读行 → 原文；没有导读行时**逐字等于** `index_text(chunk, meta)`。"""

    lead = table_lead(chunk.text)
    if not lead:
        return index_text(chunk, meta)
    header = meta.prefix(chunk.section_path) if meta is not None else ""
    return compose_index_text(header=header, lead=lead, text=chunk.text)


def compose_index_text(*, header: str, lead: str, text: str) -> str:
    """索引文本的**唯一**拼装口径：身份前缀 → 导读行 → 原文（空段直接丢）。

    抽成函数是为了让 unit 级切块（`akr.ingest.unit_chunks`）复用同一份拼法：unit 的导读行
    来自**父块**（单元自己可能整段都不含期间行/数值行），顺序与分隔符必须与块级逐字一致，
    否则「同一个库两种索引文本」又会出现——那正是本轮要消掉的那类问题。

    顺带一个便宜的用法：`compose_index_text(header=h, lead=l, text="")` 返回的正好是
    「前缀 + 导读行」这一段的字符数（含分隔换行），嵌入预算因此不必自己数分隔符。
    """

    return "\n".join(part for part in (header, lead, text) if part)


def make_table_index_text_fn(metas: Mapping[str, DocMeta]) -> Callable[[Chunk], str]:
    """`Chunk → 索引文本` 的可调用对象：BM25 侧（`Bm25Index(index_text_fn=...)`）与
    向量侧（Task 7 的 `VectorStore(index_text_fn=...)`）共用同一份规则。

    缺 meta 的 chunk 只丢身份前缀，导读行照旧生效（与 `index_text` 的降级口径一致：
    没有身份就只按正文检索）。
    """

    def render(chunk: Chunk) -> str:
        return table_index_text(chunk, metas.get(chunk.doc_id))

    return render
