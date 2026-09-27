r"""把 SQLite 分块表导出成 `data/chunks.jsonl`（Task 22 `build_candidates` 的输入）。

这是交接文档 §6 第 6 条记的第二个缺口脚本：`eval/taskset/build_candidates.py` 要求每行至少含
`chunk_id / doc_id / source / text`，但分块只躺在 SQLite / Chroma / artifact 三个存储里，
没有导出器，所以「30 条核心任务集的草稿」一直造不出来。

用法（先跑 `scripts/ingest_corpus.py` 入库）：

    .\.venv\Scripts\python.exe scripts\export_chunks.py                    # → data/chunks.jsonl
    .\.venv\Scripts\python.exe scripts\export_chunks.py --doc-id annual-600438-2025
    .\.venv\Scripts\python.exe scripts\export_chunks.py --out tmp/head.jsonl --limit-per-doc 20

每行字段：`chunk_id / doc_id / source / ordinal / page_start / page_end / section_path / text`。
后四个是给人看与给后续抽样用的溯源字段，`read_chunks` 只强制前四个。顺序固定为
`(doc_id, ordinal)`，同一份库重复导出得到逐字节相同的文件。

退出码：0 = 导出成功；1 = 库不存在 / 点名的 doc_id 不在库里 / 库里一条分块都没有。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from akr.storage import sqlite_store

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "akr.sqlite"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

# 与 build_candidates.read_chunks 的强制字段保持一致，另外带上溯源列。
FIELDS = (
    "chunk_id",
    "doc_id",
    "source",
    "ordinal",
    "page_start",
    "page_end",
    "section_path",
    "text",
)


def collect_rows(
    conn: Any, *, doc_ids: tuple[str, ...] = (), limit_per_doc: int | None = None
) -> list[dict[str, Any]]:
    """按 `(doc_id, ordinal)` 顺序收集分块；点名的 doc_id 不在库里立刻报错。"""

    documents = sqlite_store.list_documents(conn)
    known = {row.doc_id for row in documents}
    missing = sorted(set(doc_ids) - known)
    if missing:
        raise ValueError(f"库里没有这些 doc_id：{missing}（先跑 scripts/ingest_corpus.py）")

    rows: list[dict[str, Any]] = []
    for document in documents:
        if doc_ids and document.doc_id not in doc_ids:
            continue
        chunks = sqlite_store.list_chunks(conn, document.doc_id)
        if limit_per_doc is not None:
            chunks = chunks[:limit_per_doc]
        for chunk in chunks:
            rows.append({field: getattr(chunk, field) for field in FIELDS})
    return rows


def dump_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """写 JSONL；`ensure_ascii=False` 保留中文原文，行尾统一 LF。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 SQLite 分块表导出成 chunks JSONL")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite 路径，默认 data/akr.sqlite")
    parser.add_argument("--out", default=str(CHUNKS_PATH), help="输出 JSONL，默认 data/chunks.jsonl")
    parser.add_argument("--doc-id", action="append", default=[], help="只导出这些 doc_id（可重复）")
    parser.add_argument("--limit-per-doc", type=int, default=None, help="每份文档最多导出 N 条")
    args = parser.parse_args(argv)

    if args.limit_per_doc is not None and args.limit_per_doc < 1:
        print("[FAIL] --limit-per-doc 必须 >= 1", file=sys.stderr)
        return 1
    db_path = Path(args.db)
    if not db_path.is_file():
        print(f"[FAIL] 库不存在：{db_path}（先跑 scripts/ingest_corpus.py）", file=sys.stderr)
        return 1

    conn = sqlite_store.connect(db_path)
    try:
        try:
            rows = collect_rows(
                conn, doc_ids=tuple(args.doc_id), limit_per_doc=args.limit_per_doc
            )
        except ValueError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()

    if not rows:
        print(f"[FAIL] {db_path} 里没有任何分块（先跑 scripts/ingest_corpus.py）", file=sys.stderr)
        return 1

    target = dump_rows(rows, args.out)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["doc_id"]] = counts.get(row["doc_id"], 0) + 1
    for doc_id, count in sorted(counts.items()):
        print(f"[ok] {doc_id}：{count} 条")
    print(f"[done] 共 {len(rows)} 条 / {len(counts)} 份文档 → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
