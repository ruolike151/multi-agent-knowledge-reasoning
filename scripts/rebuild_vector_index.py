r"""从 SQLite 分块表重建 Chroma 向量索引（不解析 PDF、不碰图谱、不写 SQLite）。

为什么需要这个脚本：`scripts/ingest_corpus.py --force` **不能**重建向量索引。`--force`
只解除文档级的「已在库则跳过」，分块级差量是 `ingest_document` 内部按内容寻址的
`chunk_id` 算的：重解析后 chunk 内容没变 ⇒ `diff.added` 为空 ⇒ `vectors.add` 一次都不
调用（`src/akr/ingest/incremental.py`）。所以向量索引一旦损坏，现有入口无论加什么参数都
修不好，只能从分块表重嵌。2026-09-25 冒烟实测撞上这件事：`data/chroma` 的 HNSW 段文件
（`header.bin` / `data_level0.bin` / `length.bin` / `link_lists.bin`）丢失，只剩
`index_metadata.pickle`，`count()` / `get()` / `query()` 一律抛
`InternalError: Error constructing hnsw segment reader`。

向量本体不落在 `chroma.sqlite3` 里（`embeddings` 表只有
`id, segment_id, embedding_id, seq_id, created_at` 五个字段，没有向量列），所以**损坏的
向量不可恢复**；但分块文本是完好的（`data/akr.sqlite` 的 `chunks` 表 + `data/artifacts`），
因此重建只花嵌入的钱，不需要重新解析 15 份 PDF、也不会动 DeepSeek 的预算。

用法（仓库根目录）：

    # 只读体检：打印条数，并在有数据时真发一次查询（不写任何东西）
    .\.venv\Scripts\python.exe scripts\rebuild_vector_index.py --check

    # 零花费预演：只算「要嵌多少条、前缀挤掉多少字符预算」，不读 Key、不建库、不调 API
    .\.venv\Scripts\python.exe scripts\rebuild_vector_index.py --dry-run --doc-prefix

    # 表格导读行版（在身份前缀与原文之间插一段表格导读行；同样可零花费预演）
    .\.venv\Scripts\python.exe scripts\rebuild_vector_index.py --dry-run --table-prefix

带前缀的口径**必须写一个新目录**（本轮用 `data\chroma-v3`）：`data\chroma` 是既往所有检索
数字的基准，由 `guard_target` 硬拦；`data\chroma-v2` 是上一轮重嵌的产物，同样别覆盖。

    # 真重建：目标目录必须是空目录（或不存在）；已有数据时要 --force 才覆盖
    .\.venv\Scripts\python.exe scripts\rebuild_vector_index.py --chroma data\chroma

    # 本地嵌入兜底（免 API）：需 `pip install -e ".[embed]"`
    .\.venv\Scripts\python.exe scripts\rebuild_vector_index.py --embedder local

嵌入器与生产链路**必须一致**：`build_tool_context` 给 `VectorStore` 传的是
`build_embedder()`（硅基流动 `BAAI/bge-large-zh-v1.5`、文档侧不加检索前缀、L2 归一化），
`VectorStore.query` 复用同一个 `embed_fn`，所以本脚本默认也用 `build_embedder()`——换别的
编码器会让查询向量与库内向量不在同一空间，检索静默变差。

默认路径同样直接取自生产常量（`akr.mcp_server.server.DEFAULT_*`），即**相对路径**
`data/akr.sqlite` / `data/chroma`，而不是自己拼一个绝对路径。原因见下：本仓路径含中文，
而 chromadb 1.5.9 打开**含非 ASCII 字符的绝对路径**时会稳定失败（同一份索引：相对路径
3/3 成功、绝对中文路径 3/3 失败、经 junction 的绝对 ASCII 路径成功；对比实验记录在
内部偏差日志）。生产链路走的就是相对路径，本脚本按生产的形态读写，
才不会出现「脚本说建好了、实验读不出来」。

退出码：0 = 成功；1 = 参数/前置条件不满足、SQLite 里没有分块、目标索引打不开或验证不过。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import load_doc_meta_or_empty, make_index_text_fn
from akr.ingest.table_context import make_table_index_text_fn
from akr.mcp_server.server import DEFAULT_DB_PATH, DEFAULT_MANIFEST_PATH
from akr.storage.embeddings import MAX_INPUT_CHARS, UsageMeter, dump_usage
from akr.storage.vector_store import EmbedFn, IndexTextFn, VectorStore

# 生产常量是相对路径，这里照抄而不重算成绝对路径（见模块 docstring）。脚本按文档要求
# 在仓库根目录下运行，两者指向同一份数据。
DEFAULT_DB = Path(DEFAULT_DB_PATH)
# 这个脚本重建的是**块级**索引（从 SQLite 的分块表重嵌原文），所以默认目标钉死在
# `data/chroma` 字面量上，**不跟** `server.DEFAULT_CHROMA_PATH`：线上 2026-09-26 已切到
# unit 级的 `data/chroma-v3`，跟着常量走会让「按默认跑一次」把块级向量写进 unit 级的库。
# v3 由 `scripts/build_unit_chunks.py` 从 dump 重嵌，不是这个脚本的活。
DEFAULT_CHROMA = Path("data/chroma")
DEFAULT_MANIFEST = Path(DEFAULT_MANIFEST_PATH)

# 一批 256 条：`VectorStore.add` 对每批只调一次 `upsert`，而嵌入器内部再按
# `API_BATCH_SIZE=16` 切分请求。批太小则 upsert 次数多、进度刷屏；批太大则在
# 一次失败里丢掉太多进度。256 是「一次 upsert 约 16 个嵌入请求」的量级。
DEFAULT_BATCH_SIZE = 256
PROBE_CHARS = 200


# 生产向量库目录（`data/chroma`）：本轮所有历史数字都建立在这份库上，它是**只读基准**。
# 带前缀的索引文本是另一套口径，必须写进一个新目录（本轮用 `data/chroma-v3`）——覆盖生产库
# 会让既往结论无法复算，所以这里用硬闸门拦住，而不是靠命令行自觉。
PRODUCTION_CHROMA = DEFAULT_CHROMA


def guard_target(chroma_path: str | Path, index_text_fn: IndexTextFn | None) -> None:
    """带前缀的索引文本不许写进生产库目录；不带前缀的普通重建（历史口径）不受限。

    闸门放在 `rebuild` 的**第一行**：这样「用错目录」在任何一次写入、甚至任何一次 `count()`
    之前就失败。只写在 CLI 里挡不住直接调函数的调用方，所以它是函数而不是 `main` 里的分支。
    """

    if index_text_fn is None:
        return
    if Path(chroma_path).resolve() == Path(PRODUCTION_CHROMA).resolve():
        raise ValueError(
            f"{chroma_path} 是生产向量库，不许把带前缀的索引文本写进去（既往所有检索数字"
            "都以这份库为基准）。换一个空目录重建：--chroma data\\chroma-v3"
        )


class IndexUnreadable(RuntimeError):
    """目标 Chroma 目录存在但读不出来（典型：HNSW 段文件丢失）。"""


def load_chunks(db_path: str | Path) -> list[Chunk]:
    """从 `chunks` 表读出全部分块，按 `(doc_id, ordinal)` 排序保证重建顺序稳定。

    `PRAGMA query_only` 让「只读」由 SQLite 自己保证，而不是靠调用方自觉——本脚本
    全程不该写 SQLite，分块表是它唯一的事实来源。
    """

    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到分块库：{path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = ON")
        rows = conn.execute(
            "SELECT chunk_id, doc_id, source, ordinal, text, content_hash, "
            "section_path, page_start, page_end, char_start, char_end "
            "FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
    finally:
        conn.close()
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


def open_store(
    chroma_path: str | Path,
    embed_fn: EmbedFn,
    *,
    index_text_fn: IndexTextFn | None = None,
) -> tuple[VectorStore, int]:
    """打开目标索引并读一次条数：坏索引在这一步就会暴露，而不是等重建到一半。

    `chromadb.PersistentClient` 的构造是惰性的，`get_or_create_collection` 也不碰段文件，
    所以必须真读一次 `count()` 才知道能不能用。

    `index_text_fn` 决定这份库的**嵌入文本口径**（默认原文；传了就是原文 + 身份前缀）。
    读一次条数不需要它，但这里一并交给 `VectorStore`，避免「开库」与「写库」两处口径分叉。
    """

    try:
        store = VectorStore(chroma_path, embed_fn, index_text_fn=index_text_fn)
        return store, store.count()
    except Exception as exc:
        raise IndexUnreadable(
            f"{chroma_path} 存在但读不出向量库（{type(exc).__name__}: {exc}）。"
            "若是 HNSW 段损坏，请把该目录改名留档（别删，见偏差日志）后指向一个空目录重建。"
        ) from exc


def probe_query(store: VectorStore, chunks: Sequence[Chunk]) -> list[str]:
    """真发一次查询，返回命中的 chunk_id 列表。

    用库内中间一条分块的前 200 字符当探针：它必然在库里，所以「命中非空」是索引可用的
    最低证据；只报条数不看命中，就没法区分「索引建好」与「索引能读但向量是坏的」。
    """

    probe = chunks[len(chunks) // 2]
    hits = store.query(probe.text[:PROBE_CHARS], top_k=5)
    return [hit.chunk_id for hit in hits]


def plan_embedding(
    chunks: Sequence[Chunk],
    *,
    chroma_path: str | Path,
    index_text_fn: IndexTextFn | None,
    manifest_path: str | Path | None,
    batch_size: int,
    existing: int | None = None,
    force: bool = False,
    out: object = None,
) -> dict[str, object]:
    """打印「这次要嵌什么、花多少条」的预演行，返回原文 / 索引文本的字符数。

    抽成独立函数是为了让 `--dry-run` 与真重建**共用同一份口径**：预演说「会挤掉
    N 条的 500 字符预算」，真重建就必须打出同一个 N。两处各写一遍必然漂移。

    `index_text_fn is None` 时只报原文行；开了前缀才报第二部分——这样「本次到底
    带不带前缀」在输出里是可判定的，而不是靠调用方记得。
    """

    out = sys.stdout if out is None else out
    raw_chars = sum(len(chunk.text) for chunk in chunks)
    # `existing=None` = 这次不查向量库（`--dry-run` 不打开 Chroma）。写死 0 会让人以为
    # 「目标是空库」，而预演根本没读过那份库。
    existing_text = f"原有 {existing} 条" if existing is not None else "原有条数不查（预演不打开向量库）"
    print(
        f"[plan] {len(chunks)} chunk（{raw_chars} 字符）→ {chroma_path}；"
        f"{existing_text}，批大小 {batch_size}"
        + ("，--force 覆盖" if force else ""),
        file=out,
        flush=True,
    )
    index_chars: int | None = None
    clipped: int | None = None
    if index_text_fn is not None:
        index_chars = sum(len(index_text_fn(chunk)) for chunk in chunks)
        clipped = sum(1 for chunk in chunks if len(index_text_fn(chunk)) > MAX_INPUT_CHARS)
        # 原文本来超限多少条也要报：不然「3753 条超限」会被读成「前缀毁了 3753 条」，
        # 实际是原文阶段就已经超了 3588 条，前缀只是又多挤掉一百多条。
        raw_clipped = sum(1 for chunk in chunks if len(chunk.text) > MAX_INPUT_CHARS)
        print(
            f"[plan] 身份前缀生效（manifest={manifest_path}）：索引文本 {index_chars} 字符"
            f"（原文 +{index_chars - raw_chars}），其中 {clipped}/{len(chunks)} 条超过"
            f" {MAX_INPUT_CHARS} 字符（原文已超 {raw_clipped} 条，前缀又挤掉 "
            f"{clipped - raw_clipped} 条）——前缀会挤占同一份预算，这些块的正文会被多截一段",
            file=out,
            flush=True,
        )
    return {"chunks": len(chunks), "raw_chars": raw_chars, "index_chars": index_chars,
            "clipped": clipped}


def dry_run(
    *,
    db_path: str | Path = DEFAULT_DB,
    chroma_path: str | Path = DEFAULT_CHROMA,
    index_text_fn: IndexTextFn | None = None,
    manifest_path: str | Path | None = DEFAULT_MANIFEST,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    out: object = None,
) -> dict[str, object]:
    """零花费预演：只读 SQLite 分块表 + manifest，算字符预算，**不碰嵌入器**。

    存在的理由是「换了前缀就得重嵌一次全量」这件事本身要花钱，而「这次前缀会挤掉多少
    字符预算」不该靠真跑才知道。它不读 Key、不开 Chroma，所以在没有 API key 的机器上
    也能复算已经写进文档的那组字符数（`data/chroma` 只是被打印、不被访问）。
    """

    out = sys.stdout if out is None else out
    chunks = load_chunks(db_path)
    if not chunks:
        raise ValueError(f"{db_path} 的 chunks 表是空的：先跑 scripts\\ingest_corpus.py 入库")
    stats = plan_embedding(
        chunks,
        chroma_path=chroma_path,
        index_text_fn=index_text_fn,
        manifest_path=manifest_path,
        batch_size=batch_size,
        force=force,
        out=out,
    )
    print("[dry-run] 未读 Key、未开向量库、未发任何请求（字符预算可复算）", file=out, flush=True)
    return stats


def rebuild(
    *,
    db_path: str | Path = DEFAULT_DB,
    chroma_path: str | Path = DEFAULT_CHROMA,
    embed_fn: EmbedFn,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    index_text_fn: IndexTextFn | None = None,
    manifest_path: str | Path | None = DEFAULT_MANIFEST,
    out: object = None,
) -> dict[str, object]:
    """把 `chunks` 表全量重嵌进 `chroma_path`，返回统计字典。

    不删除任何东西：`--force` 也只是对已有集合 `upsert`（按 chunk_id 幂等），所以
    「重建」不会把目标目录里别的东西一起端掉。

    `index_text_fn=None`（默认）= 嵌原文，与本仓历史上所有索引同口径；
    传 `akr.ingest.doc_meta.make_index_text_fn(metas)` = 嵌「身份前缀 + 原文」。
    **两者不可混用**：查询向量侧不变，但库内向量必须与建库时的口径一致，否则同一份索引
    会被两套文本解释，检索质量无从归因。
    """

    guard_target(chroma_path, index_text_fn)
    out = sys.stdout if out is None else out
    chunks = load_chunks(db_path)
    if not chunks:
        raise ValueError(f"{db_path} 的 chunks 表是空的：先跑 scripts\\ingest_corpus.py 入库")

    store, existing = open_store(chroma_path, embed_fn, index_text_fn=index_text_fn)
    if existing and not force:
        raise ValueError(
            f"{chroma_path} 里已有 {existing} 条向量（本次要写 {len(chunks)} 条）。"
            "确认要覆盖请加 --force（会重新嵌入全部，花钱）；重建通常应指向一个空目录。"
        )

    plan = plan_embedding(
        chunks,
        chroma_path=chroma_path,
        index_text_fn=index_text_fn,
        manifest_path=manifest_path,
        batch_size=batch_size,
        existing=existing,
        force=force,
        out=out,
    )
    raw_chars = plan["raw_chars"]
    started = time.perf_counter()
    written = 0
    for start in range(0, len(chunks), batch_size):
        window = list(chunks[start : start + batch_size])
        written += store.add(window)
        print(
            f"[embed] {written}/{len(chunks)}（{time.perf_counter() - started:.1f}s）",
            file=out,
            flush=True,
        )
    elapsed = time.perf_counter() - started

    total = store.count()
    hits = probe_query(store, chunks)
    if total != len(chunks):
        raise ValueError(f"重建后条数不符：库内 {total} 条，期望 {len(chunks)} 条")
    if not hits:
        raise ValueError("重建后探针查询没有命中：索引能读但检索返回空，不能拿去跑实验")
    print(
        f"[verify] 条数 {total}/{len(chunks)}；探针（{chunks[len(chunks) // 2].chunk_id}）"
        f"top-5 命中 {hits}",
        file=out,
        flush=True,
    )
    print(
        f"[done] 重嵌 {written} 条、用时 {elapsed:.1f}s"
        f"（{written / elapsed:.1f} 条/s）→ {chroma_path}",
        file=out,
        flush=True,
    )
    return {
        "chunks": len(chunks),
        "written": written,
        "elapsed_s": elapsed,
        "hits": hits,
        "chroma_path": str(chroma_path),
        "raw_chars": raw_chars,
        "doc_prefix": index_text_fn is not None,
    }


def check(*, db_path: str | Path = DEFAULT_DB, chroma_path: str | Path = DEFAULT_CHROMA,
          embed_fn: EmbedFn, out: object = None) -> dict[str, object]:
    """只读体检：报条数，并在有数据时真发一次查询。不写任何存储。"""

    out = sys.stdout if out is None else out
    chunks = load_chunks(db_path)
    store, total = open_store(chroma_path, embed_fn)
    hits = probe_query(store, chunks) if total and chunks else []
    print(
        f"[check] SQLite 分块 {len(chunks)} 条；{chroma_path} 向量 {total} 条；"
        f"探针 top-5 命中 {hits}",
        file=out,
        flush=True,
    )
    return {"chunks": len(chunks), "vectors": total, "hits": hits}


def build_embedder(
    kind: str,
    *,
    usage: UsageMeter | None = None,
    usage_path: str | Path | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """`api`（默认）走硅基流动；`local` 走本地 bge-small-zh-v1.5（需 `.[embed]`）。"""

    from akr.storage.embeddings import build_embedder as api_embedder, build_local_embedder

    if kind == "local":
        # 本地模型不花钱，`usage` 对它没有意义（没有服务端 usage 可记）。
        return build_local_embedder()
    return api_embedder(usage=usage, usage_path=usage_path)


def resolve_index_text_fn(
    doc_prefix: bool, manifest_path: str | Path | None, *, table: bool = False
) -> IndexTextFn | None:
    """把 `--doc-prefix` / `--table-prefix` 解成一个 `Chunk → 索引文本` 的函数（都不开就返 None）。

    两个开关的差别只有一处：`--doc-prefix` 的索引文本是「身份前缀 + 原文」（`data/chroma-v2`
    那份库的口径），`--table-prefix` 是「身份前缀 + 表格导读行 + 原文」（`akr.ingest.table_context`
    的规则，与 BM25 侧是**同一份实现**）。后者已经含身份前缀，所以它单独用就够；两个都开时
    以 `table` 为准——既不叠加成两份前缀，也不静默退回弱口径。`table` 是**关键字**参数，
    `(False, path)` / `(True, path)` 这样的两位置参数调用保持原样可用。

    开了却读不到 manifest **必须报错**：静默退化成「嵌原文」会白花一次全量嵌入的钱，
    还会让索引口径与命令行声明不符——这种账事后对不上，代价比直接失败大得多。
    """

    if not doc_prefix and not table:
        return None
    switch = "--table-prefix" if table else "--doc-prefix"
    metas = load_doc_meta_or_empty(manifest_path)
    if not metas:
        raise ValueError(
            f"{switch} 需要文档身份，但 {manifest_path} 不存在或没有 documents 数组；"
            "先跑 scripts\\fetch_corpus.py 生成 manifest"
        )
    if table:
        return make_table_index_text_fn(metas)
    return make_index_text_fn(metas)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从 SQLite 分块表重建 Chroma 向量索引（不解析 PDF、不碰图谱）"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="分块库，默认 data/akr.sqlite")
    parser.add_argument(
        "--chroma",
        default=str(DEFAULT_CHROMA),
        help="目标向量库，默认 data/chroma；别传含中文的绝对路径（见模块 docstring）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每次 upsert 的条数"
    )
    parser.add_argument("--force", action="store_true", help="目标已有向量时也覆盖（重新嵌入，花钱）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="只读体检，不写任何存储")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="零花费预演：只读 SQLite + manifest 算字符预算，不读 Key、不碰向量库",
    )
    parser.add_argument(
        "--doc-prefix",
        action="store_true",
        help="嵌入文本前置「公司（代码）报告期 标题 章节」身份前缀（默认关；见模块 docstring）",
    )
    parser.add_argument(
        "--table-prefix",
        action="store_true",
        help="在身份前缀与原文之间插入表格块导读行（默认关；比 --doc-prefix 更强，已含身份前缀）",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="文档身份来源，默认 data/manifest.json（只有 --doc-prefix / --table-prefix 时才读）",
    )
    parser.add_argument(
        "--usage-json",
        default=None,
        help="把服务端返回的嵌入用 token 数落盘成 JSON（默认不记；不猜 token，只抄 usage）",
    )
    parser.add_argument(
        "--embedder",
        choices=("api", "local"),
        default="api",
        help="api=硅基流动 BAAI/bge-large-zh-v1.5（与生产链路一致，默认）；local=本地兜底",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        print("[FAIL] --batch-size 必须 >= 1", file=sys.stderr)
        return 1

    if args.doc_prefix or args.table_prefix:
        # 索引文本口径在日志里必须**可判定**：`[plan]` 那行只说「身份前缀生效」，而这对
        # `--table-prefix` 也是真的（它是超集），读者分不出这一轮到底用了哪一份规则。
        label = "身份前缀 + 表格导读行" if args.table_prefix else "身份前缀"
        print(f"[knobs] 索引文本口径：{label}（manifest={args.manifest}）", flush=True)

    if args.dry_run:
        # 预演刻意排在嵌入器之前：它在没有 API key 的机器上也必须能跑，
        # 否则「复算文档里的字符数」又会变成一件要先花钱才能做的事。
        try:
            index_text_fn = resolve_index_text_fn(
                args.doc_prefix, args.manifest, table=args.table_prefix
            )
            dry_run(
                db_path=args.db,
                chroma_path=args.chroma,
                index_text_fn=index_text_fn,
                manifest_path=args.manifest,
                batch_size=args.batch_size,
                force=args.force,
            )
        except (OSError, ValueError) as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        return 0

    meter = UsageMeter() if args.usage_json else None
    try:
        embed_fn = build_embedder(args.embedder, usage=meter, usage_path=args.usage_json)
    except RuntimeError as exc:  # 缺 Key / 缺本地依赖
        print(f"[FAIL] 嵌入器不可用：{exc}", file=sys.stderr)
        return 1

    try:
        if args.check:
            check(db_path=args.db, chroma_path=args.chroma, embed_fn=embed_fn)
            return 0
        index_text_fn = resolve_index_text_fn(
            args.doc_prefix, args.manifest, table=args.table_prefix
        )
        rebuild(
            db_path=args.db,
            chroma_path=args.chroma,
            embed_fn=embed_fn,
            batch_size=args.batch_size,
            force=args.force,
            index_text_fn=index_text_fn,
            manifest_path=args.manifest,
        )
    except (OSError, ValueError, IndexUnreadable) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if meter is not None:
        dump_usage(meter, args.usage_json)
        print(
            f"[usage] 嵌入请求 {meter.requests} 次、输入 {meter.inputs} 条、"
            f"服务端计 {meter.tokens} token（缺 usage 的响应 {meter.unreported} 次）→ "
            f"{args.usage_json}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
