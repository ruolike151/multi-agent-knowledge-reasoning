"""查询理解：从问句里抽出「证券代码 / 年份 / 公司名」，把候选文档从 15 份收到 1~2 份。

为什么需要这一步（2026-09-25 只读实测）：把身份前缀写进 BM25 的索引文本之后，
纯词法通道的 top-8 命中从 5/25 到 6/25——**只有 +1**。原因是前缀把「区分度 token」
从「完全不存在」变成「在 400 条同文档 chunk 里都出现」，idf 被自己的同族块摊薄了；
真正让命中跳到 11/25 的是**先用代码把候选文档收窄**，让这些 token 只在 1~2 份文档里较量。

两条约束：

* **零模型**：全是确定性规则，不调用任何 LLM（工具层硬约束，见 `akr.mcp_server.tools`）；
* **抽不到就不收窄**：宁可退化成全库检索，也不能因为抽错年份把 gold 所在文档过滤掉。
  年份过滤只在「过滤后还有候选」时才生效，否则回退到不带年份的候选集。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from akr.ingest.doc_meta import DocMeta

# 六位证券代码：前后都不能贴着别的数字（`2024` 年、`100000` 万元不会命中）。
_CODE = re.compile(r"(?<!\d)\d{6}(?!\d)")
# 四位年份：只认 19xx / 20xx。证券代码里不会出现以 19/20 开头的四位片段。
_YEAR = re.compile(r"(?:19|20)\d{2}")


@dataclass(frozen=True)
class QueryHints:
    """一条问句里抽出的收窄线索。空元组表示「这一路没抽到」。"""

    codes: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    companies: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.codes or self.companies)


def extract_hints(
    question: str,
    metas: Mapping[str, DocMeta],
    *,
    known_codes: frozenset[str] | None = None,
) -> QueryHints:
    """抽证券代码 / 年份 / 公司名。

    * 代码只在**语料里真实存在**时才算线索（`known_codes`）——问句里出现的六位数字
      也可能是金额（`842123 万元`），不在语料里的代码收窄不出任何文档，不如不抽。
    * 公司名用「语料里的公司名是不是问句的子串」来判定，不需要别名表；问句普遍写的
      就是 manifest 里的简称（实测 25/25 道可答题的问句都含公司简称）。
    """

    codes_pool = known_codes if known_codes is not None else frozenset(
        meta.code for meta in metas.values() if meta.code
    )
    codes = tuple(sorted({code for code in _CODE.findall(question) if code in codes_pool}))
    years = tuple(sorted(set(_YEAR.findall(question))))
    companies = tuple(
        sorted({meta.company for meta in metas.values() if meta.company and meta.company in question})
    )
    return QueryHints(codes=codes, years=years, companies=companies)


def narrow_doc_ids(
    hints: QueryHints,
    metas: Mapping[str, DocMeta],
) -> tuple[str, ...]:
    """把候选文档收窄成 doc_id 元组；抽不到线索时返回空元组（= 不收窄）。

    优先级：证券代码 > 公司名（代码是唯一键，公司名可能被问句里的同业比较带出多个）。
    年份只做**二次过滤**，且过滤后为空就撤销它——问句说「2024 年」而语料只有
    「2025 年度」公告时，正确行为是回到代码命中的那几份文档，而不是清空候选。
    """

    if hints.codes:
        basis = [meta for meta in metas.values() if meta.code and meta.code in hints.codes]
    elif hints.companies:
        basis = [meta for meta in metas.values() if meta.company in hints.companies]
    else:
        return ()

    if hints.years:
        narrowed = [
            meta
            for meta in basis
            if any(year in meta.years for year in hints.years)
        ]
        if narrowed:
            basis = narrowed
    return tuple(sorted(meta.doc_id for meta in basis))
