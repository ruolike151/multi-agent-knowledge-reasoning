from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from akr.ingest.chunker import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id      TEXT PRIMARY KEY,
  source      TEXT NOT NULL,
  doc_type    TEXT NOT NULL,
  page_count  INTEGER NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     TEXT PRIMARY KEY,
  doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  source       TEXT NOT NULL,
  ordinal      INTEGER NOT NULL,
  text         TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  section_path TEXT NOT NULL DEFAULT '',
  page_start   INTEGER NOT NULL DEFAULT 0,
  page_end     INTEGER NOT NULL DEFAULT 0,
  char_start   INTEGER NOT NULL DEFAULT 0,
  char_end     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS costs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id        TEXT NOT NULL,
  node          TEXT NOT NULL,
  hit_tokens    INTEGER NOT NULL,
  miss_tokens   INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cost_cny      REAL NOT NULL,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_costs_run ON costs(run_id);
"""

_CHUNK_COLUMNS = (
    "chunk_id, doc_id, source, ordinal, text, content_hash, "
    "section_path, page_start, page_end, char_start, char_end"
)


@dataclass(frozen=True)
class DocumentRow:
    doc_id: str
    source: str
    doc_type: str
    page_count: int
    chunk_count: int
    updated_at: str


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    source: str,
    doc_type: str,
    page_count: int,
    updated_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, source, doc_type, page_count, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
          source = excluded.source,
          doc_type = excluded.doc_type,
          page_count = excluded.page_count,
          updated_at = excluded.updated_at
        """,
        (doc_id, source, doc_type, page_count, updated_at),
    )
    conn.commit()


def replace_chunks(conn: sqlite3.Connection, doc_id: str, chunks: Sequence[Chunk]) -> None:
    with conn:  # 删除与插入必须原子，避免中途失败留下半份文档
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            f"INSERT INTO chunks ({_CHUNK_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [_row(chunk) for chunk in chunks],
        )


def list_chunks(conn: sqlite3.Connection, doc_id: str) -> list[Chunk]:
    rows = conn.execute(
        f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE doc_id = ? ORDER BY ordinal",
        (doc_id,),
    ).fetchall()
    return [
        Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source=row["source"],
            ordinal=row["ordinal"],
            text=row["text"],
            content_hash=row["content_hash"],
            section_path=row["section_path"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            char_start=row["char_start"],
            char_end=row["char_end"],
        )
        for row in rows
    ]


def list_documents(conn: sqlite3.Connection) -> list[DocumentRow]:
    rows = conn.execute(
        """
        SELECT d.doc_id, d.source, d.doc_type, d.page_count, d.updated_at,
               COUNT(c.chunk_id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.doc_id = d.doc_id
        GROUP BY d.doc_id
        ORDER BY d.doc_id
        """
    ).fetchall()
    return [
        DocumentRow(
            doc_id=row["doc_id"],
            source=row["source"],
            doc_type=row["doc_type"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


def add_cost(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    node: str,
    cost_cny: float,
    hit_tokens: int,
    miss_tokens: int,
    output_tokens: int,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO costs (run_id, node, hit_tokens, miss_tokens, output_tokens, cost_cny, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, node, hit_tokens, miss_tokens, output_tokens, cost_cny, created_at),
    )
    conn.commit()


def total_cost_cny(conn: sqlite3.Connection, *, run_id: str | None = None) -> float:
    if run_id is None:
        row = conn.execute("SELECT COALESCE(SUM(cost_cny), 0.0) AS total FROM costs").fetchone()
    else:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_cny), 0.0) AS total FROM costs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return float(row["total"])


def _row(chunk: Chunk) -> tuple:
    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.source,
        chunk.ordinal,
        chunk.text,
        chunk.content_hash,
        chunk.section_path,
        chunk.page_start,
        chunk.page_end,
        chunk.char_start,
        chunk.char_end,
    )
