"""零成本探针：把文档元数据注入 chunk 的检索文本后，纯 BM25 的召回会怎么变。

动机（2026-09-25 实测）：25 道可答题的 21 个唯一 gold chunk 里，
**0 个含证券代码、只有 8 个含公司名**；而语料 6176 个 chunk 里，
正文含证券代码的只有 36 个、含公司名的有 2824 个。
问句却几乎都点名「公司 + 证券代码 + 年份」——也就是说，
查询里最具区分度的 token 在目标 chunk 的正交文里根本不存在，
BM25 只能靠「硅片 / 净利润」这类全行业通用词去和 4 家同业年报的近重复段落硬碰。

本脚本只做本地计算（jieba + rank_bm25），**不调用任何 API、零花费**。
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from akr.ingest.chunker import Chunk  # noqa: E402
from akr.storage.bm25_index import Bm25Index  # noqa: E402

CHUNKS = ROOT / "data/chunks.jsonl"
MANIFEST = ROOT / "data/manifest.json"
TASKSET = ROOT / "eval/taskset/core.jsonl"

_CHUNK_DEFAULTS = {"content_hash": "", "char_start": 0, "char_end": 0}
_CHUNK_FIELDS = {field.name for field in dataclasses.fields(Chunk)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunks() -> list[Chunk]:
    chunks = []
    for row in load_jsonl(CHUNKS):
        merged = {**_CHUNK_DEFAULTS, **row}
        chunks.append(Chunk(**{key: merged[key] for key in _CHUNK_FIELDS}))
    return chunks


def load_doc_meta() -> dict[str, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {doc["doc_id"]: doc for doc in manifest["documents"]}


def header_of(chunk: Chunk, meta: dict[str, dict]) -> str:
    doc = meta.get(chunk.doc_id, {})
    parts = [
        str(doc.get("company", "")),
        f"（{doc['code']}）" if doc.get("code") else "",
        str(doc.get("period", "")),
        str(doc.get("announcement_title", "")),
        str(chunk.section_path or ""),
    ]
    return " ".join(part for part in parts if part).strip()


def rank_of(gold_ids: list[str], hits: list) -> int | None:
    wanted = set(gold_ids)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk_id in wanted:
            return rank
    return None


def summarize(rows: list[dict], key: str, depth: int) -> tuple[int, int, float]:
    hit8 = sum(1 for row in rows if (row[key] or 9999) <= 8)
    hitd = sum(1 for row in rows if (row[key] or 9999) <= depth)
    mrr = sum(1.0 / row[key] for row in rows if row[key]) / len(rows)
    return hit8, hitd, mrr


def docs_for_question(question: str, meta: dict[str, dict]) -> tuple[str, ...]:
    """从问句里抽证券代码与年份，缩出候选文档（模拟「查询理解」这一步）。"""

    codes = re.findall(r"(?<!\d)(\d{6})(?!\d)", question)
    years = re.findall(r"(20\d{2})\s*年", question)
    picked = []
    for doc_id, doc in meta.items():
        if doc.get("code") not in codes:
            continue
        if years and not any(year in str(doc.get("period", "")) for year in years):
            continue
        picked.append(doc_id)
    return tuple(sorted(picked))


def main() -> int:
    started = time.perf_counter()
    chunks = load_chunks()
    meta = load_doc_meta()
    tasks = [task for task in load_jsonl(TASKSET) if task.get("answerable") and task.get("gold_chunk_ids")]
    print(f"[load] chunk {len(chunks)}、文档 {len(meta)}、可答题 {len(tasks)}")

    plain = Bm25Index()
    plain.build(chunks)

    enriched = [
        dataclasses.replace(chunk, text=f"{header_of(chunk, meta)}\n{chunk.text}") for chunk in chunks
    ]
    with_meta = Bm25Index()
    with_meta.build(enriched)

    sample = next(chunk for chunk in chunks if chunk.doc_id == "annual-002129-2024" and chunk.ordinal == 10)
    print(f"[样例前缀] {header_of(sample, meta)!r}")

    depth = 50
    rows = []
    for task in tasks:
        question = str(task["question"])
        gold = list(task["gold_chunk_ids"])
        docs = docs_for_question(question, meta)
        rows.append(
            {
                "task_id": task["task_id"],
                "type": task["type"],
                "docs": docs,
                "plain@8": rank_of(gold, plain.search(question, top_k=8)),
                "plain@50": rank_of(gold, plain.search(question, top_k=depth)),
                "meta@8": rank_of(gold, with_meta.search(question, top_k=8)),
                "meta@50": rank_of(gold, with_meta.search(question, top_k=depth)),
                "meta_doc@8": rank_of(gold, with_meta.search(question, top_k=8, doc_ids=docs)) if docs else None,
                "meta_doc@50": rank_of(gold, with_meta.search(question, top_k=depth, doc_ids=docs)) if docs else None,
            }
        )
        print(
            f"[{task['task_id']:>14}] plain@8={rows[-1]['plain@8']} plain@50={rows[-1]['plain@50']} "
            f"meta@8={rows[-1]['meta@8']} meta@50={rows[-1]['meta@50']} "
            f"meta_doc@8={rows[-1]['meta_doc@8']} docs={len(docs)}"
        )

    print("\n[summary] 25 道可答题，查询 = 原问句")
    print(f"  {'配置':<16} {'top-8':>7} {'top-50':>7} {'MRR':>7}")
    for key in ("plain@8", "plain@50", "meta@8", "meta@50", "meta_doc@8", "meta_doc@50"):
        hit8, hitd, mrr = summarize(rows, key, depth)
        print(f"  {key:<16} {hit8:>4}/25 {hitd:>5}/25 {mrr:>7.3f}")

    print("\n[by type] meta_doc@8 / meta@8 / plain@50")
    for kind in ("single_hop", "multi_hop", "multi_doc"):
        group = [row for row in rows if row["type"] == kind]
        if not group:
            continue
        print(
            f"  {kind:<12} meta_doc@8={sum(1 for r in group if (r['meta_doc@8'] or 9999) <= 8)}/{len(group)}"
            f"  meta@8={sum(1 for r in group if (r['meta@8'] or 9999) <= 8)}/{len(group)}"
            f"  plain@50={sum(1 for r in group if (r['plain@50'] or 9999) <= depth)}/{len(group)}"
        )

    print(f"\n[done] 耗时 {time.perf_counter() - started:.1f}s，零 API 调用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
