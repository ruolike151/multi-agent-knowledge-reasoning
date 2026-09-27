r"""全语料入库：`data/manifest.json` 里的 15 份 PDF → artifact / Chroma / BM25 / SQLite / 图谱。

这是交接文档 §6 第 6 条记的两个缺口脚本之一：`akr.ingest.incremental.ingest_document` 只是库函数，
`scripts/m1_smoke.py` 又只跑 1 份 PDF 的前 40 页且 `client=None` 不建图，所以「一次把 15 份灌进去」
一直没有可执行入口。Task 35 / 36 / 37 与 Task 22 的候选任务构造都卡在这上面。

用法（仓库根目录，先起 Neo4j）：

    # 零花费预演：只解析 + 分块并报数，不写库、不调任何模型
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py --dry-run

    # 真入库：建 artifact / Chroma / BM25，跳过图谱抽取（只花嵌入的钱，不花 DeepSeek 的 ¥5）
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py

    # 带图谱抽取：每个新增 chunk 一次 DeepSeek 调用，用 --max-calls 硬闸门封顶
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py --extract --max-calls 200

    # 挑着灌 / 只灌前 N 页 / 强制重灌
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py --doc-id annual-600438-2025
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py --limit 1 --pages 40
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py --force

    # 真解析 + 真嵌入的冒烟：四库整包挪到临时目录，不污染生产库
    .\.venv\Scripts\python.exe scripts\ingest_corpus.py --doc-id ann-002129-ir --data-root $env:TEMP\akr-smoke

重复执行是幂等的：默认跳过 SQLite 里已有分块的 doc_id（`--force` 才重灌）。即使重灌也不会重复
累计内容——`ingest_document` 按 chunk_id 内容寻址做差量，未变的 chunk 只复用不重嵌。

退出码：0 = 全部成功；1 = 语料缺失 / 参数自相矛盾 / 有文档入库失败 / 撞到调用闸门。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from akr.config import Settings
from akr.ingest.chunker import chunk_pages
from akr.ingest.incremental import IngestResult, ingest_document
from akr.ingest.parser import ParsedPage, parse_pdf
from akr.mcp_server.tools import ToolContext
from akr.model.client import LLMClient
from akr.model.cost import Usage
from akr.storage import sqlite_store
from akr.storage.artifact_store import ArtifactStore
from akr.storage.bm25_index import Bm25Index
from akr.storage.embeddings import build_embedder, build_local_embedder
from akr.storage.graph_store import GraphStore
from akr.storage.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
DATA_ROOT = ROOT / "data"


@dataclass(frozen=True)
class DataPaths:
    """四库的落盘位置。默认 `data/`，`--data-root` 可整包重定向。

    重定向的用途是跑「真解析 + 真嵌入」的冒烟而不污染生产库（`scripts/m1_smoke.py`
    的做法是临时目录 + 跑完删除，这里给一个更省事的旋钮）。
    """

    root: Path = DATA_ROOT

    @property
    def db(self) -> Path:
        return self.root / "akr.sqlite"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def chroma(self) -> Path:
        return self.root / "chroma"

    @property
    def bm25(self) -> Path:
        return self.root / "bm25.json"


class CallBudgetExceeded(RuntimeError):
    """抽取调用数或花费撞到 `--max-calls` / `--max-cost-cny` 闸门。

    闸门在**发请求之前**判定，所以撞门时不会再多花一次钱；已入库的文档保持完整
    （`ingest_document` 是文档级完整写入），重跑本脚本即从下一份继续。
    """


@dataclass
class CallMeter:
    """抽取调用的计数与成本累加（¥5 预算的执行侧闸门）。"""

    max_calls: int | None = None
    max_cost_cny: float | None = None
    peak: bool = True
    calls: int = 0
    cost_cny: float = 0.0

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise CallBudgetExceeded(
                f"已达 --max-calls={self.max_calls}（累计 {self.calls} 次调用、"
                f"≈¥{self.cost_cny:.4f}）"
            )
        if self.max_cost_cny is not None and self.cost_cny >= self.max_cost_cny:
            raise CallBudgetExceeded(
                f"已达 --max-cost-cny={self.max_cost_cny}（累计 {self.calls} 次调用、"
                f"≈¥{self.cost_cny:.4f}）"
            )

    def charge(self, usage: Usage) -> float:
        cost = usage.cost_cny(peak=self.peak)
        self.calls += 1
        self.cost_cny += cost
        return cost


class MeteredClient:
    """给 `LLMClient` 套一层调用计数与成本累加；只实现 `chat`，够 `extract_chunk` 用。

    刻意不继承 `LLMClient`：继承会把「未计数」的其它入口也带进来，闸门就漏了。
    """

    def __init__(self, inner: Any, meter: CallMeter) -> None:
        self._inner = inner
        self._meter = meter

    def chat(self, **kwargs: Any) -> Any:
        self._meter.check()
        result = self._inner.chat(**kwargs)
        self._meter.charge(result.usage)
        return result


@dataclass(frozen=True)
class DocPreview:
    doc_id: str
    pages: int
    chunks: int
    chars: int

    @property
    def chars_per_chunk(self) -> float:
        return self.chars / self.chunks if self.chunks else 0.0


@dataclass
class RunStats:
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    chunks: int = 0
    added: int = 0
    embedded: int = 0
    extracted: int = 0


def load_documents(path: str | Path = MANIFEST) -> list[dict[str, Any]]:
    """读 manifest 的 `documents` 数组；结构不对立刻报错（不做静默兜底）。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("documents") if isinstance(payload, dict) else None
    if not records:
        raise ValueError(f"{path} 里没有非空的 documents 数组")
    return list(records)


def select_documents(
    records: list[dict[str, Any]], *, doc_ids: tuple[str, ...] = (), limit: int | None = None
) -> list[dict[str, Any]]:
    """按 `--doc-id` 过滤、再按 `--limit` 截断；点名了不存在的 doc_id 直接报错。"""

    selected = [r for r in records if not doc_ids or r["doc_id"] in doc_ids]
    missing = sorted(set(doc_ids) - {r["doc_id"] for r in selected})
    if missing:
        raise ValueError(f"manifest 里没有这些 doc_id：{missing}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit 必须 >= 1")
        selected = selected[:limit]
    return selected


def resolve_pdf(record: dict[str, Any], *, root: Path = ROOT) -> Path:
    """按 manifest 的 `file` 定位 PDF；对不上时按「公司代码 + 年份」在语料目录里回退匹配。

    manifest 的 `file` 字段与磁盘文件名曾经不一致（例如写成「2024年年度报告.pdf」，实体是
    「2024年报.pdf」），所以不能盲信拼出来的路径；回退路径命中数不为 1 时显式报错，
    不猜一个「看起来最像」的文件。
    """

    declared = record.get("file")
    if declared:
        path = Path(root) / declared
        if path.is_file():
            return path

    kind_dir = Path(root) / "data" / "corpus" / (
        "annual" if record.get("kind") == "annual" else "announcement"
    )
    candidates = sorted(p for p in kind_dir.glob(f"{record.get('code', '')}_*.pdf") if p.is_file())
    year = str(record.get("period", ""))[:4]
    narrowed = [p for p in candidates if year and year in p.name] or candidates
    if len(narrowed) == 1:
        return narrowed[0]
    raise FileNotFoundError(
        f"{record.get('doc_id')}：找不到 PDF（manifest.file={declared!r}，"
        f"按 {record.get('code')} 回退命中 {[p.name for p in narrowed]}）；"
        "先跑 scripts/fetch_corpus.py 补齐语料"
    )


def parse_document(path: str | Path, *, pages: int | None = None) -> list[ParsedPage]:
    """解析 PDF；`pages` 只取前 N 页（冒烟用，页码保持原文页码）。"""

    parsed = parse_pdf(path)
    return parsed[:pages] if pages else parsed


def preview(
    records: list[dict[str, Any]], *, pages: int | None = None, progress_out: Any = None
) -> list[DocPreview]:
    """`--dry-run` 的主体：只解析 + 分块计数，不碰任何存储、不调任何模型。

    分块数用生产分块器现算（`chunk_pages` 由 `ingest_document` 内部默认调用），
    所以这里的数字就是入库后的 chunk 数。`progress_out` 给定时逐份打印耗时——
    全语料解析是分钟级操作，没有逐份输出就只能盯着一个空屏。
    """

    rows: list[DocPreview] = []
    for record in records:
        started = time.perf_counter()
        path = resolve_pdf(record)
        parsed = parse_document(path, pages=pages)
        chunks = chunk_pages(parsed, doc_id=record["doc_id"], source=path.name)
        row = DocPreview(
            doc_id=record["doc_id"],
            pages=len(parsed),
            chunks=len(chunks),
            chars=sum(len(chunk.text) for chunk in chunks),
        )
        rows.append(row)
        if progress_out is not None:
            print(
                f"[parse] {row.doc_id}：{row.pages} 页 → {row.chunks} chunk"
                f"（{row.chars} 字符，{time.perf_counter() - started:.1f}s）",
                file=progress_out,
                flush=True,
            )
    return rows


def print_previews(rows: list[DocPreview]) -> None:
    print(f"{'doc_id':<24} {'页数':>5} {'chunk':>6} {'字符':>10} {'字符/chunk':>11}")
    for row in rows:
        print(
            f"{row.doc_id:<24} {row.pages:>5} {row.chunks:>6} {row.chars:>10} "
            f"{row.chars_per_chunk:>11.1f}"
        )
    print(
        f"{'合计':<24} {sum(r.pages for r in rows):>5} {sum(r.chunks for r in rows):>6} "
        f"{sum(r.chars for r in rows):>10}"
    )


def _pick_embedder(kind: str) -> Any:
    """`api`（默认）走硅基流动 `BAAI/bge-large-zh-v1.5`；`local` 走本地 bge-small-zh-v1.5。"""

    return build_local_embedder() if kind == "local" else build_embedder()


def build_context(
    *, embedder: str = "api", paths: DataPaths | None = None, vector_doc_prefix: bool | None = None
) -> tuple[ToolContext, Any, Bm25Index]:
    """装配生产路径上的四库（真实 `data/` 目录 + 真实 Neo4j）。

    BM25 返回的是**内存实例**：入库过程中由 `ingest_document` 增量维护，结束后统一用
    `dump_bm25` 从 SQLite 重建再落盘，避免 `--force` 重灌后内存索引与分块表不一致。

    `vector_doc_prefix`（默认读环境变量 `AKR_VECTOR_DOC_PREFIX`）必须与目标向量库的建库
    口径一致：新增 chunk 用的是**嵌入文本**，与库里老向量不是同一套文本时，检索质量会被
    静默拉低，且事后无法从索引里看出来。所以这里开了但读不到 manifest 会直接报错。
    """

    paths = paths or DataPaths()
    if vector_doc_prefix is None:
        vector_doc_prefix = os.environ.get("AKR_VECTOR_DOC_PREFIX", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    index_text_fn = None
    if vector_doc_prefix:
        from akr.ingest.doc_meta import load_doc_meta_or_empty, make_index_text_fn

        metas = load_doc_meta_or_empty(paths.root / "manifest.json")
        if not metas:
            raise ValueError(
                "AKR_VECTOR_DOC_PREFIX 开着，但 data/manifest.json 里没有文档身份："
                "新 chunk 会与库内既有向量落在两套文本口径上，故直接失败"
            )
        index_text_fn = make_index_text_fn(metas)
    conn = sqlite_store.connect(paths.db)
    sqlite_store.init_schema(conn)
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "akr-local-pass"),
        ),
    )
    driver.verify_connectivity()
    graph = GraphStore(driver)
    graph.init_schema()
    bm25 = Bm25Index()
    ctx = ToolContext(
        conn=conn,
        vectors=VectorStore(paths.chroma, _pick_embedder(embedder), index_text_fn=index_text_fn),
        bm25=bm25,
        graph=graph,
        artifacts=ArtifactStore(paths.artifacts),
    )
    return ctx, driver, bm25


def dump_bm25(conn: Any, path: str | Path) -> int:
    """从 SQLite 分块表重建 BM25 索引并落盘，返回 chunk 数。

    重建而不是复用内存索引，是让落盘结果只取决于分块表这一个事实来源
    （服务端起动时 `build_tool_context` 是对着 `data/bm25.json` 做的读取）。
    """

    chunks = [
        chunk
        for document in sqlite_store.list_documents(conn)
        for chunk in sqlite_store.list_chunks(conn, document.doc_id)
    ]
    index = Bm25Index()
    index.build(chunks)
    index.save(path)
    return len(chunks)


def ingest_all(
    records: list[dict[str, Any]],
    *,
    ctx: ToolContext,
    client: Any,
    pages: int | None = None,
    force: bool = False,
    out: Any = None,
) -> RunStats:
    """逐份入库；单份失败只记失败并继续（批量脚本不该被一份坏 PDF 全盘阻断）。"""

    # 默认值不能在签名里写 sys.stdout：那是导入期的对象，测试的 capsys 换不掉它。
    out = sys.stdout if out is None else out
    existing = {row.doc_id for row in sqlite_store.list_documents(ctx.conn) if row.chunk_count}
    stats = RunStats()
    for record in records:
        doc_id = record["doc_id"]
        if doc_id in existing and not force:
            stats.skipped.append(doc_id)
            print(f"[skip] {doc_id}：已在库中（--force 可重灌）", file=out)
            continue
        try:
            path = resolve_pdf(record)
            parsed = parse_document(path, pages=pages)
            result: IngestResult = ingest_document(
                doc_id=doc_id,
                source=path.relative_to(ROOT).as_posix(),
                pages=parsed,
                conn=ctx.conn,
                artifacts=ctx.artifacts,
                vectors=ctx.vectors,
                bm25=ctx.bm25,
                graph=ctx.graph,
                client=client,
            )
        except CallBudgetExceeded:
            raise
        except Exception as exc:  # 单份失败不阻断整批；失败清单与退出码都会报出来
            stats.failed.append(doc_id)
            print(f"[FAIL] {doc_id}：{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        stats.ingested.append(doc_id)
        stats.chunks += result.total_chunks
        stats.added += result.added
        stats.embedded += result.embedded
        stats.extracted += result.extracted
        print(
            f"[ok] {doc_id}：{len(parsed)} 页 → chunk {result.total_chunks}"
            f"（新增 {result.added} / 复用 {result.unchanged} / 重嵌 {result.embedded}），"
            f"抽取 {result.extracted}，用时 {result.elapsed_ms / 1000:.1f}s",
            file=out,
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="全语料入库（15 份 PDF → artifact / Chroma / BM25 / 图谱）")
    parser.add_argument("--manifest", default=str(MANIFEST), help="语料清单路径，默认 data/manifest.json")
    parser.add_argument(
        "--data-root",
        default=str(DATA_ROOT),
        help="四库根目录，默认 data/；冒烟时可指到临时目录，不污染生产库",
    )
    parser.add_argument("--dry-run", action="store_true", help="只解析 + 分块报数，不写库、不调模型")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="开启图谱实体抽取（每个新增 chunk 一次 DeepSeek 调用，花钱）",
    )
    parser.add_argument("--max-calls", type=int, default=None, help="抽取调用数硬闸门（需配合 --extract）")
    parser.add_argument(
        "--max-cost-cny", type=float, default=None, help="抽取花费硬闸门（需配合 --extract）"
    )
    parser.add_argument("--off-peak", action="store_true", help="按低峰价（半价）累加成本")
    parser.add_argument("--doc-id", action="append", default=[], help="只灌这些 doc_id（可重复）")
    parser.add_argument("--limit", type=int, default=None, help="最多灌 N 份（在 --doc-id 之后截断）")
    parser.add_argument("--pages", type=int, default=None, help="每份只取前 N 页（冒烟用）")
    parser.add_argument("--force", action="store_true", help="已入库的 doc_id 也重灌")
    parser.add_argument(
        "--embedder",
        choices=("api", "local"),
        default="api",
        help="api=硅基流动 BAAI/bge-large-zh-v1.5（默认）；local=本地 bge-small-zh-v1.5（兜底）",
    )
    args = parser.parse_args(argv)

    if not args.extract and (args.max_calls is not None or args.max_cost_cny is not None):
        print("[FAIL] --max-calls / --max-cost-cny 只在 --extract 下有意义：不加 --extract 就没有模型调用，闸门会静默失效", file=sys.stderr)
        return 1
    if args.max_calls is not None and args.max_calls < 1:
        print("[FAIL] --max-calls 必须 >= 1", file=sys.stderr)
        return 1

    try:
        records = select_documents(
            load_documents(args.manifest), doc_ids=tuple(args.doc_id), limit=args.limit
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[dry-run] {len(records)} 份文档（不写库、不调任何模型）")
        try:
            rows = preview(records, pages=args.pages, progress_out=sys.stdout)
        except (OSError, ValueError) as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        print_previews(rows)
        return 0

    if args.extract and args.max_calls is None and args.max_cost_cny is None:
        print(
            "[warn] --extract 已开但没设 --max-calls / --max-cost-cny：本次抽取调用数不设上限",
            file=sys.stderr,
        )

    paths = DataPaths(Path(args.data_root))
    ctx, driver, _ = build_context(embedder=args.embedder, paths=paths)
    meter = CallMeter(
        max_calls=args.max_calls, max_cost_cny=args.max_cost_cny, peak=not args.off_peak
    )
    client = MeteredClient(LLMClient(Settings.from_env()), meter) if args.extract else None
    started = time.perf_counter()
    try:
        stats = ingest_all(
            records, ctx=ctx, client=client, pages=args.pages, force=args.force
        )
        bm25_chunks = dump_bm25(ctx.conn, paths.bm25)
        documents = sqlite_store.list_documents(ctx.conn)
    except CallBudgetExceeded as exc:
        print(f"[STOP] {exc}；已入库的文档保持完整，重跑本脚本从下一份继续", file=sys.stderr)
        print(
            f"[i] 累计调用 {meter.calls} 次、≈¥{meter.cost_cny:.4f}（peak={meter.peak}）",
            file=sys.stderr,
        )
        return 1
    finally:
        ctx.conn.close()
        driver.close()

    print(
        f"[done] 入库 {len(stats.ingested)} 份 / 跳过 {len(stats.skipped)} 份 / 失败 {len(stats.failed)} 份，"
        f"处理 chunk {stats.chunks}（其中新增 {stats.added}、重嵌 {stats.embedded}、抽取 {stats.extracted}）"
    )
    print(
        f"[done] 库内共 {len(documents)} 份文档、{sum(d.chunk_count for d in documents)} chunk；"
        f"BM25 落盘 {bm25_chunks} 条 → {paths.bm25.as_posix()}；总用时 {time.perf_counter() - started:.1f}s"
    )
    if meter.calls:
        print(f"[cost] 抽取调用 {meter.calls} 次、≈¥{meter.cost_cny:.4f}（peak={meter.peak}）")
    if stats.failed:
        print(f"[FAIL] 未入库：{stats.failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
