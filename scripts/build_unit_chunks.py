r"""建 `data/chroma-v3`：**400 字表格行级 unit** + 导读行重嵌（Task 7 的第二剂量）。

为什么是 unit 而不只是把父块重嵌一遍：`akr.storage.embeddings.ApiEncoder.encode` 是硬截断
`text[:500]`，而语料 6176 个父块里 3588 个（58.1%）正文超 500 字符——3588 条的尾巴**从未
进过嵌入**，被截掉的正好是答案所在的那一行（`…归属于上市公司股东的净利润
-6,419,556,843.85` 落在第 500 字之后）。只加前缀（`data/chroma-v2`）治不了这个病：前缀
还会**多挤掉**一段预算。零 token 的 BM25 受控探针（内部探针脚本）给出了
剂量方向：同样的问句在 800 字块上 top-8 命中 6/25，换成 400 字 unit 后 9/25，单行 unit
12/25，而「父块级有没有」在细粒度下恒为 25/25。

切块规则本身在 `akr.ingest.unit_chunks`（行边界切、索引文本压进嵌入窗口、继承父块导读行）；
本脚本只做取数与落盘三件事：

1. `data/manifest.json` → `parse_pdf` → `unit_chunks`（不读 SQLite、不动 `data/chunks.jsonl`：
   重解析与生产分块器逐字复现已由 `tmp/v3_parse_probe.py` 对过账，`docs_with_diff=0/15`）；
2. 嵌入 unit 的**索引文本**（身份前缀 → 导读行 → unit 原文），写进 `{chroma_path}`；
3. 落 `data/unit-map.json`（unit id → 父块 id）。这份映射是评测侧折叠的唯一依据：
   `VectorStore(id_map=...)` 把 unit 名次折回父块，`chunk_id` 一律是父块 id，所以 gold
   不必重造、RRF 也不会把同一个父块算成两支候选。

**父块 id 逐字等于 `chunk_pages` 的产出**（`parent_blocks` 是唯一实现），所以 unit 命中
折回的 id 直接就是 `data/retrieval-ab.json` 里那份 gold。

带索引文本的库必须写新目录：`data/chroma` 是所有历史数字的基准、`data/chroma-v2` 是上一轮
重嵌的产物，两者都由 `guard_target` 硬拦。

用法（仓库根目录；解释器 `.\.venv\Scripts\python.exe`）：

    # 零花费预演：重解析 + 切 unit + 报规模/字符预算，不读 Key、不开 Chroma
    # （顺带把 unit 落盘，真建库时用 --from-dump 复用，省掉第二次全量重解析）
    .\.venv\Scripts\python.exe scripts\build_unit_chunks.py --dry-run --dump tmp\units-v3.json

    # 真建库（读 dump，不重解析）
    .\.venv\Scripts\python.exe tmp\dotenv_run.py scripts\build_unit_chunks.py ^
        --from-dump tmp\units-v3.json --usage-json tmp\chroma-v3-usage.json

退出码：0 = 成功；1 = 前置/参数不满足、目标目录不合法、条数或探针校验不过。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import load_doc_meta
from akr.ingest.parser import parse_pdf
from akr.ingest.unit_chunks import (
    MIN_UNIT_CHARS,
    PROSE_UNIT_CHARS,
    TABLE_UNIT_CHARS,
    UnitChunk,
    unit_chunks,
)
from akr.mcp_server.server import DEFAULT_MANIFEST_PATH
from akr.storage.embeddings import MAX_INPUT_CHARS, UsageMeter, dump_usage
from akr.storage.vector_store import EmbedFn, VectorStore

# 生产常量是相对路径，这里照抄而不重算成绝对路径（chromadb 1.5.9 打开含非 ASCII 字符的
# 绝对路径会稳定失败，详见 `scripts/rebuild_vector_index.py` 的模块 docstring）。
DEFAULT_MANIFEST = Path(DEFAULT_MANIFEST_PATH)
DEFAULT_CHROMA_V3 = Path("data/chroma-v3")
DEFAULT_UNIT_MAP = Path("data/unit-map.json")

# 一批 256 条：与 `rebuild_vector_index.py` 同量级（一次 upsert ≈ 16 个嵌入请求）。
DEFAULT_BATCH_SIZE = 256
PROBE_CHARS = 200

# 不许覆盖的两份库：v1 是历史基准，v2 是上一轮的产物。写进去会让既往结论无法复算。
FORBIDDEN_TARGETS = (Path("data/chroma"), Path("data/chroma-v2"))


def guard_target(chroma_path: str | Path) -> None:
    """拦住会覆盖既有基准的目标目录；闸门放在写库路径的第一行。"""

    resolved = Path(chroma_path).resolve()
    for forbidden in FORBIDDEN_TARGETS:
        if resolved == forbidden.resolve():
            raise ValueError(
                f"{chroma_path} 是既有的向量库（历史数字的基准），不许覆盖。"
                "换成空目录：--chroma data\\chroma-v3"
            )


def load_documents(manifest_path: str | Path) -> list[dict[str, Any]]:
    """读 manifest 的 `documents` 数组：`doc_id` 与 `file` 缺一不可，缺了就报错。"""

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = payload.get("documents") if isinstance(payload, dict) else None
    if not records:
        raise ValueError(f"{manifest_path} 里没有 documents 数组：先跑 scripts\\fetch_corpus.py")
    for record in records:
        if not record.get("doc_id") or not record.get("file"):
            raise ValueError(f"{manifest_path} 里有缺 doc_id / file 的记录：{record!r}")
    return list(records)


def build_units(
    manifest_path: str | Path,
    *,
    table_chars: int = TABLE_UNIT_CHARS,
    prose_chars: int = PROSE_UNIT_CHARS,
    min_chars: int = MIN_UNIT_CHARS,
    out: object = None,
) -> list[UnitChunk]:
    """逐份重解析 PDF → unit 切块；打印每份文档的规模与耗时。

    取数走 PDF 重解析而不是读 `data/chunks.jsonl`：unit 要落在父块覆盖的 `spans` 上
    （「按行边界切」这件事只有解析层知道行在哪），而重解析与生产分块器逐字等价已对过账。
    """

    out = sys.stdout if out is None else out
    metas = load_doc_meta(manifest_path)
    documents = load_documents(manifest_path)
    units: list[UnitChunk] = []
    started = time.perf_counter()
    for record in documents:
        doc_id = str(record["doc_id"])
        path = Path(record["file"])
        doc_started = time.perf_counter()
        pages = parse_pdf(path)
        built = unit_chunks(
            pages,
            doc_id=doc_id,
            source=str(path),
            meta=metas.get(doc_id),
            table_chars=table_chars,
            prose_chars=prose_chars,
            min_chars=min_chars,
        )
        units.extend(built)
        print(
            f"[parse] {doc_id:<22} unit={len(built):>5} "
            f"chars={sum(len(item.index_text) for item in built):>8} "
            f"({time.perf_counter() - doc_started:.1f}s)",
            file=out,
            flush=True,
        )
    print(f"[parse] 合计 {len(units)} 个 unit，用时 {time.perf_counter() - started:.1f}s", file=out)
    return units


def plan(units: list[UnitChunk], *, chroma_path: str | Path, out: object = None) -> dict[str, Any]:
    """预演行：要嵌多少条、多少字符、几条超窗口——真跑与预演共用这一份口径。"""

    out = sys.stdout if out is None else out
    index_chars = sum(len(item.index_text) for item in units)
    raw_chars = sum(len(item.chunk.text) for item in units)
    clipped = sum(1 for item in units if len(item.index_text) > MAX_INPUT_CHARS)
    parents = len({item.parent_id for item in units})
    print(
        f"[plan] {len(units)} 个 unit（折回 {parents} 个父块）→ {chroma_path}；"
        f"索引文本 {index_chars} 字符（unit 原文 {raw_chars}，前缀+导读行 "
        f"{index_chars - raw_chars}）；超过 {MAX_INPUT_CHARS} 字符的 unit {clipped} 条"
        f"（{clipped / len(units):.1%}）",
        file=out,
        flush=True,
    )
    if clipped:
        print(
            f"[plan] 超窗口的那些是「导读行把预算吃光、触到 {MIN_UNIT_CHARS} 字下限」的 unit："
            "尾巴会被嵌入截断，这是刻意取舍（切得更小也没有检索价值）",
            file=out,
            flush=True,
        )
    return {
        "units": len(units),
        "parents": parents,
        "index_chars": index_chars,
        "raw_chars": raw_chars,
        "clipped": clipped,
    }


def dump_units(units: list[UnitChunk], path: str | Path) -> None:
    """把 unit 落盘（含 `index_text`）：真建库时 `--from-dump` 复用，省掉一次全量重解析。

    `index_text` 一起落盘是刻意的：它是装配期派生量，重算要么再解析一次 PDF、要么把
    `table_lead` 的规则在第二个地方实现一遍——两条路都会引入第二份事实来源。
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [
                {
                    "chunk": asdict(item.chunk),
                    "parent_id": item.parent_id,
                    "index_text": item.index_text,
                }
                for item in units
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_units(path: str | Path) -> list[UnitChunk]:
    """读回 `dump_units` 的产物；结构不对立刻报错（宁可失败，也不要静默少几条）。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} 里没有 unit 数组：先用 --dry-run --dump 生成")
    return [
        UnitChunk(
            chunk=Chunk(**item["chunk"]),
            parent_id=str(item["parent_id"]),
            index_text=str(item["index_text"]),
        )
        for item in payload
    ]


def load_unit_map(path: str | Path) -> dict[str, str]:
    """读 unit → 父块 id 的映射（`rebuild` 落盘的那份）。

    评测侧（`scripts/retrieval_pool_depth.py`）拿它当 `VectorStore(id_map=...)`。缺文件或结构
    不对**必须报错**：静默退化成「不折叠」会让 unit 级的命中 id 落在 gold 空间之外，
    整列数字会以「召回率为 0」的形状骗过所有检查。

    实现搬到 `akr.storage.vector_store.load_id_map`（装配层也要读同一份文件），这里只留别名，
    保证「映射只有一个解析口径」。
    """

    from akr.storage.vector_store import load_id_map

    return load_id_map(path)


def rebuild(
    units: list[UnitChunk],
    *,
    chroma_path: str | Path,
    embed_fn: EmbedFn,
    unit_map_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    out: object = None,
) -> dict[str, Any]:
    """把 unit 全量嵌进 `chroma_path`，并落 unit → 父块的映射。

    `index_text_fn` 用 dump 里的 `index_text` 查表（同一个 `chunk_id` 只对应一条文本），
    写进 Chroma 的 `documents` 仍是 unit 原文——命中回给模型的片段不该带前缀或导读行。

    `id_map` 也一并交给 `VectorStore`（它只影响 `query`、不影响 `add`），这样收尾那次探针
    跑的就是**评测侧的真实取数路径**：命中折回父块 id，而不是只证明「索引里存了 unit id」。
    """

    guard_target(chroma_path)
    out = sys.stdout if out is None else out
    index_texts = {item.chunk.chunk_id: item.index_text for item in units}
    id_map = {item.chunk.chunk_id: item.parent_id for item in units}
    # 打开目标索引并在同一步读一次条数：坏索引在这里就暴露，而不是等嵌到一半。
    store = VectorStore(
        chroma_path,
        embed_fn,
        index_text_fn=lambda chunk: index_texts[chunk.chunk_id],
        id_map=id_map,
    )
    existing = store.count()
    if existing and not force:
        raise ValueError(
            f"{chroma_path} 里已有 {existing} 条向量（本次要写 {len(units)} 条）。"
            "确认要覆盖请加 --force（会重新嵌入全部，花钱）；首次建库应指向一个空目录。"
        )
    plan(units, chroma_path=chroma_path, out=out)

    started = time.perf_counter()
    written = 0
    for start in range(0, len(units), batch_size):
        window = units[start : start + batch_size]
        written += store.add([item.chunk for item in window])
        print(
            f"[embed] {written}/{len(units)}（{time.perf_counter() - started:.1f}s）",
            file=out,
            flush=True,
        )
    elapsed = time.perf_counter() - started

    total = store.count()
    if total != len(units):
        raise ValueError(f"建库后条数不符：库内 {total} 条，期望 {len(units)} 条")
    probe = units[len(units) // 2]
    probe_hits = store.query(probe.chunk.text[:PROBE_CHARS], top_k=5)
    parent_ids = {item.parent_id for item in units}
    if not probe_hits:
        raise ValueError("建库后探针查询没有命中：索引能读但检索返回空，不能拿去跑实验")
    folded = sum(1 for hit in probe_hits if hit.chunk_id in parent_ids)
    print(
        f"[verify] 条数 {total}/{len(units)}；探针（{probe.chunk.chunk_id}）top-5 命中 "
        f"{[hit.chunk_id for hit in probe_hits]}（折叠后 {folded}/{len(probe_hits)} 条落在"
        "父块 id 空间，与 gold 同口径）",
        file=out,
        flush=True,
    )
    unit_map_path = Path(unit_map_path)
    unit_map_path.parent.mkdir(parents=True, exist_ok=True)
    unit_map_path.write_text(
        json.dumps(id_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] 重嵌 {written} 条、用时 {elapsed:.1f}s（{written / elapsed:.1f} 条/s）→ "
        f"{chroma_path}；映射 {len(units)} 条 → {unit_map_path}",
        file=out,
        flush=True,
    )
    return {
        "units": len(units),
        "written": written,
        "elapsed_s": elapsed,
        "chroma_path": str(chroma_path),
        "unit_map": str(unit_map_path),
    }


def build_embedder(kind: str, *, usage: UsageMeter | None = None, usage_path: str | Path | None = None):
    """`api`（默认）走硅基流动、`local` 走本地 bge-small-zh-v1.5（需 `.[embed]`）。

    必须与生产链路同模型：`build_tool_context` 给 `VectorStore` 传的是 `build_embedder()`，
    换编码器会让查询向量与库内向量不在同一空间，检索静默变差。
    """

    from akr.storage.embeddings import build_embedder as api_embedder, build_local_embedder

    if kind == "local":
        return build_local_embedder()
    return api_embedder(usage=usage, usage_path=usage_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="建 data/chroma-v3：400 字表格行级 unit + 导读行重嵌"
    )
    parser.add_argument("--chroma", default=str(DEFAULT_CHROMA_V3), help="目标向量库（别传含中文的绝对路径）")
    parser.add_argument("--unit-map", default=str(DEFAULT_UNIT_MAP), help="unit → 父块 id 的映射落盘路径")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="文档身份与 PDF 路径来源")
    parser.add_argument("--table-chars", type=int, default=TABLE_UNIT_CHARS, help="表格块 unit 目标字数")
    parser.add_argument("--prose-chars", type=int, default=PROSE_UNIT_CHARS, help="正文块 unit 目标字数")
    parser.add_argument("--min-chars", type=int, default=MIN_UNIT_CHARS, help="unit 字数下限")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每次 upsert 的条数")
    parser.add_argument("--dump", default=None, help="把切好的 unit 落盘成 JSON（可复用，免二次解析）")
    parser.add_argument("--from-dump", default=None, help="从 --dump 的产物读 unit，跳过重解析")
    parser.add_argument("--force", action="store_true", help="目标已有向量时也覆盖（重新嵌入，花钱）")
    parser.add_argument(
        "--dry-run", action="store_true", help="零花费预演：只重解析 + 切块，不读 Key、不开向量库"
    )
    parser.add_argument("--usage-json", default=None, help="把服务端返回的嵌入用 token 数落盘（不猜 token）")
    parser.add_argument("--embedder", choices=("api", "local"), default="api", help="api=硅基流动（与生产一致）")
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        print("[FAIL] --batch-size 必须 >= 1", file=sys.stderr)
        return 1

    try:
        guard_target(args.chroma)
        if args.from_dump:
            units = load_units(args.from_dump)
            print(f"[load] 从 {args.from_dump} 读回 {len(units)} 个 unit（跳过重解析）")
        else:
            units = build_units(
                args.manifest,
                table_chars=args.table_chars,
                prose_chars=args.prose_chars,
                min_chars=args.min_chars,
            )
        if not units:
            raise ValueError("切出来 0 个 unit：manifest 或解析结果不对")
        if args.dump:
            dump_units(units, args.dump)
            print(f"[write] {len(units)} 个 unit → {args.dump}")
        if args.dry_run:
            plan(units, chroma_path=args.chroma)
            print("[dry-run] 未读 Key、未开向量库、未发任何请求（规模与字符预算可复算）")
            return 0
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    meter = UsageMeter() if args.usage_json else None
    try:
        embed_fn = build_embedder(args.embedder, usage=meter, usage_path=args.usage_json)
    except RuntimeError as exc:  # 缺 Key / 缺本地依赖
        print(f"[FAIL] 嵌入器不可用：{exc}", file=sys.stderr)
        return 1

    try:
        rebuild(
            units,
            chroma_path=args.chroma,
            embed_fn=embed_fn,
            unit_map_path=args.unit_map,
            batch_size=args.batch_size,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
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
