r"""只读检索层 A/B：量化「gold chunk 到底进不进得了检索池」。

为什么要这个脚本（偏差日志第二十九节）：主实验跑完后做只读归因发现，`ours` 的 25 条
可答题里 gold chunk 只有 **5 条**进过最终的检索池（`baseline_a` 是 17 条），而主表 gap
的主因就在这里，不在 120 秒墙钟。要定「检索层先修哪几条」就必须先知道：**每条候选修法
各自能把多少条题从 MISS 拉进 top-8**——拍脑袋选修法等于拿真钱去试。

本脚本**不调用任何 LLM**，只做两件事：

1. 向量侧：用真实嵌入模型给「查询文本」编码，拿 Chroma 里的真实向量排序（有嵌入花费，
   量级是每次查询一个短文本，比一次实验便宜四到五个数量级）；
2. BM25 侧：在内存里重建索引，测「压掉 CJK 换行/空格」与「查询侧加 bigram」两条修法。

对照口径（同一批 25 条可答题，同一份 6176 chunk 语料）：

* `goals@8` / `goals@50`：用**编排器真实写下的 retrieve 子任务目标**当查询，8 路与
  50 路召回深度。`goals@8` 就是主实验的复现基线（应能对上 `gold_hit=False` 的那批题）。
* `question@8` / `question@50`：**原始问句**当查询。这是候选修法④（规划器别改写问句）。
* `question@8+norm` / `question@50+norm`：候选修法①（分词前压掉 CJK 换行/空格）。
* `question@50+norm+bigram`：候选修法①+②（查询侧补字符 bigram，救未登录词）。

2026-09-25 新增的两条（都是零模型调用的检索层改动，见内部记录 A-6）：

* `+meta`：BM25 的**索引文本**前置文档身份（公司 + 证券代码 + 报告期 + 标题 + 章节），
  由 `akr.ingest.doc_meta` 从 `data/manifest.json` 派生。`Chunk.text` 不动，
  所以**不需要重新嵌入**——向量侧那一列就是线上现状，这不是代理口径。
* `+meta+narrow`：在此之上，从问句里抽证券代码 / 年份 / 公司名，把第一段检索的
  `doc_ids` 收窄到 1~3 份文档（`akr.storage.query_hints`，纯规则、零 LLM）。

2026-09-25 晚新增：`+vmeta` 后缀的几列走 `--chroma-v2`——**另一份重嵌过的向量库**，
库内每条的嵌入文本前置了同一套文档身份前缀（`scripts/rebuild_vector_index.py --doc-prefix`）。
它和 `+meta` 不是一回事：`+meta` 只改 BM25 的打分文本、零嵌入花费；`+vmeta` 是真花了
嵌入钱重嵌的对照，用来验证「让向量侧也知道身份」到底涨还是退。查询侧不变，所以两列的
差异只可能来自库内向量。

同一批问句的**第一段融合池深度**（`rrf_fuse` 拿几路候选去融）刻意不在本表里：那张表要的是
「工具层在池深 D 下真正返回了什么」，重放口径答不了，所以另开
`scripts/retrieval_pool_depth.py`（直接调 `search_documents`）。

判据有三个：

* **命中率**：`gold 排名`（取该题所有 gold chunk 里最好的名次），`None` 表示该深度内 MISS；
* **标准 Recall@K**（`recalls`）：`命中 gold 数 / gold 总数`，与主表 recall@8 同口径，
  可以直接和 `docs/experiments/2026-09-23-results.md` 的 4.0% / 22.0% 对齐。
* **MRR@K**（`mrr@8` / `mrr@50`）：该题最好名次取倒数后对 25 题求均值，MISS 记 0。
  它和命中率一样只是 `ranks` 的函数——所以**不重新检索也能重算**，见 `--from-json`。

用法（仓库根目录）：

    .\.venv\Scripts\python.exe scripts\retrieval_ab.py
    .\.venv\Scripts\python.exe scripts\retrieval_ab.py --json data\retrieval-ab.json
    # 加了汇总列（如 MRR）后只想刷新汇总、不想再付一次嵌入钱：
    .\.venv\Scripts\python.exe scripts\retrieval_ab.py --from-json data\retrieval-ab.json

退出码：0 = 成功；1 = 前置条件不满足（缺文件、缺表、任务集与记录对不上）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# 脚本间互引：`load_unit_map` 是 unit → 父块映射的**唯一**读取口径
# （写入侧在 `scripts/build_unit_chunks.py`）；两处各写一遍会漂成两份事实。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_unit_chunks import load_unit_map  # noqa: E402
from akr.ingest.chunker import Chunk
from akr.ingest.doc_meta import load_doc_meta
from akr.ingest.table_context import make_table_index_text_fn, table_lead
from akr.storage.bm25_index import Bm25Index
from akr.storage.embeddings import build_embedder
from akr.storage.fusion import rrf_fuse
from akr.storage.query_hints import extract_hints, narrow_doc_ids
from akr.storage.vector_store import VectorStore

DEFAULT_TASKSET = Path("eval/taskset/core.jsonl")
DEFAULT_CHUNKS = Path("data/chunks.jsonl")
DEFAULT_DB = Path("data/akr.sqlite")
DEFAULT_CHROMA = Path("data/chroma")
DEFAULT_JOIN = Path("data/main-trace-join.json")
DEFAULT_MANIFEST = Path("data/manifest.json")
# v3 是 **unit 级**索引（`scripts/build_unit_chunks.py`）：查出来的是 unit id，必须靠这份
# 映射折回父块，返回的 `chunk_id` 才落在 gold（父块 id）空间里。
DEFAULT_UNIT_MAP = Path("data/unit-map.json")

# CJK 统一表意文字：判断「换行两侧是不是中文」，决定压成空格还是直接接上。
_CJK = r"\u4e00-\u9fff"
# 纯空格：只有两侧都是汉字才算「被排版撑开的词」，才接上（否则 `代码 002129` 会粘成一个 token）。
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])[\s\u3000]+(?=[{_CJK}])")
# 含换行的空白：PDF 的行尾换行是排版产物，只要有一侧是汉字就接上（`TCL\n中环` → `TCL中环`）。
_LINE_BREAK = re.compile(rf"(?<=[{_CJK}])[\s\u3000]*\n[\s\u3000]*(?=[^\s])|(?<=[^\s])[\s\u3000]*\n[\s\u3000]*(?=[{_CJK}])")
_CJK_RUN = re.compile(rf"[{_CJK}]+")


class AbError(RuntimeError):
    """前置条件不满足（缺文件、缺表、任务集对不上）。"""


def normalize_cjk(text: str) -> str:
    """把「中文之间的换行/全角空格」直接接上。

    PDF 抽取会在行尾切词（实测原文是「硅\\n片整体市占率」），而 `tokenize_zh` 丢掉空白后
    得到的是 `硅` + `片` 两个单字 token——既不是词，也碰不上查询里的 `硅片`。压掉中文之间
    的空白才能让「硅片」重新成为一个 token。含换行的空白只要一侧是汉字就接上（`TCL\\n中环`），
    纯空格则要求两侧都是汉字（`代码 002129` 里的空格有语义，不能动）。
    """

    return _CJK_GAP.sub("", _LINE_BREAK.sub("", text))


def bigram_tokens(text: str) -> list[str]:
    """查询侧的字符 bigram 兜底：jieba 分不出「出货量」时，`出货`/`货量` 仍能命中。"""

    tokens: list[str] = []
    for match in _CJK_RUN.finditer(text):
        run = match.group()
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise AbError(f"缺文件：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# `data/chunks.jsonl` 的导出器只留检索需要的字段（溯源字段 char_start / char_end /
# content_hash 不在其中）。BM25 只吃 `text`，所以缺的字段补占位值，不假装它们是真的。
_CHUNK_DEFAULTS: dict[str, Any] = {"content_hash": "", "char_start": 0, "char_end": 0}
_CHUNK_FIELDS = {field.name for field in dataclasses.fields(Chunk)}


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    for row in load_jsonl(path):
        merged = {**_CHUNK_DEFAULTS, **row}
        chunks.append(Chunk(**{key: merged[key] for key in _CHUNK_FIELDS}))
    return chunks


def load_ours_goals(db_path: Path, join_path: Path) -> dict[str, list[str]]:
    """取 `ours` 每条主实验运行里真实下发的 retrieve 子任务目标（按 task_id 归位）。

    评测侧的 `run_id` 与 trace 侧的 hex 不同名，映射只在 `main-trace-join.json` 里有；
    所以这里必须读那份配对结果，不能拿 `run_id` 直接去查账本。
    """

    if not join_path.exists():
        raise AbError(f"缺文件：{join_path}（先跑 scripts/join_run_trace.py --json）")
    join = json.loads(join_path.read_text(encoding="utf-8"))
    rows = [row for row in join["rows"] if row.get("system") == "ours" and row.get("trace_run_id")]
    if not rows:
        raise AbError(f"{join_path} 里没有 system=ours 的配对行")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        payloads = {
            row["run_id"]: json.loads(row["payload"])
            for row in conn.execute("SELECT run_id, payload FROM ledgers")
        }
    finally:
        conn.close()

    goals: dict[str, list[str]] = {}
    for row in rows:
        payload = payloads.get(row["trace_run_id"])
        if payload is None:
            raise AbError(f"账本里没有 run_id={row['trace_run_id']}（配对结果与库不是同一份）")
        goals[row["task_id"]] = [
            str(subtask["goal"])
            for subtask in payload.get("subtasks", [])
            if subtask.get("kind") == "retrieve"
        ]
    return goals


def _chunk_id(item: Any) -> str:
    """取 chunk id：命中对象（读 `.chunk_id`）与**纯 id 字符串**都接受。

    扩窗列返回的是 `list[str]`（`expand_window -> list[str]`），要和别的列共用同一套
    `rank_of` / `recall_at`，这两把尺子就必须同时认两种形状——否则只能在外层再包一层假命中
    对象，凭空多出一处「形状转换」，而转换一旦写错，两列的口径就悄悄分家了。
    """

    return item if isinstance(item, str) else str(item.chunk_id)


def rank_of(gold_ids: list[str], hits: list[Any]) -> int | None:
    """gold 命中列表里的最好名次（1 起）；全部 MISS 返回 None。

    `hits` 既可以是命中对象列表（其余各列），也可以是纯 chunk id 列表（扩窗列）。
    """

    wanted = set(gold_ids)
    for rank, hit in enumerate(hits, start=1):
        if _chunk_id(hit) in wanted:
            return rank
    return None


def doc_rank_of(gold_docs: set[str], hits: list[Any]) -> int | None:
    """**文档级**名次：第一个属于 gold 所在文档的命中排在第几。"""

    for rank, hit in enumerate(hits, start=1):
        if hit.doc_id in gold_docs:
            return rank
    return None


def recall_at(gold_ids: list[str], hits: list[Any], k: int) -> float:
    """标准 Recall@k：命中的 gold 数 / gold 总数。

    这是与主表（`docs/experiments/2026-09-23-results.md` 第 1 节）同一口径的指标，
    和上面的「命中率 + MRR」不是一回事：一道题有两三条 gold 时，命中一条就记 1 次命中，
    但在标准 recall 里只拿 2/3 分。两种都报，是为了不被问「你这个 40% 到底指什么」。
    `hits` 与 `rank_of` 一样，同时接受命中对象与纯 chunk id。
    """

    wanted = set(gold_ids)
    if not wanted:
        return 0.0
    found = sum(1 for hit in hits[:k] if _chunk_id(hit) in wanted)
    return found / len(wanted)


def window_hit(
    gold_ids: list[str],
    doc_of: dict[str, str],
    hits: list[Any],
    ordinals: dict[str, int],
    width: int,
) -> tuple[bool, int | None]:
    """「先命中文档、再取邻近块」的修法能不能救回 gold。

    锚点必须是**gold 所在文档自己的命中**（ordinal 是文档内序号，跨文档比毫无意义），
    再看该文档的 gold chunk 是否落在锚点 ±width 内。返回 (是否覆盖, 锚点名次)。
    """

    wanted_by_doc: dict[str, list[int]] = {}
    for gold_id in gold_ids:
        if gold_id in ordinals:
            wanted_by_doc.setdefault(doc_of[gold_id], []).append(ordinals[gold_id])

    for rank, hit in enumerate(hits, start=1):
        if hit.doc_id not in wanted_by_doc or hit.chunk_id not in ordinals:
            continue
        anchor = ordinals[hit.chunk_id]
        if any(abs(neighbour - anchor) <= width for neighbour in wanted_by_doc[hit.doc_id]):
            return True, rank
    return False, None


def _window_offsets(width: int) -> tuple[int, ...]:
    """邻块偏移顺序：锚点自己（0）先，然后按距离由近到远，同距离先取 ordinal **更小**那条。

    为什么 0 必须排最前：扩窗后的列表要截回 `top_k`，锚点本身是「检索路已经认可过的证据」，
    先占位才不会被自己扩出来的邻块挤掉。同距离先取下一条（`-1` 先于 `+1`），是因为年报表格块的
    「指标名 + 期间」导读行排在数值行**之前**——救表格题时上一条比下一条更值钱。
    """

    offsets = [0]
    for distance in range(1, width + 1):
        offsets.extend((-distance, distance))
    return tuple(offsets)


def expand_window(
    hits: list[Any],
    *,
    anchors: int,
    width: int,
    top_k: int,
    chunk_at: dict[str, Chunk],
) -> list[str]:
    """前 `anchors` 条命中 →「它自己 + 同文档内 ±width 的邻块」→ 截回 `top_k` 条 id。

    离线只读模拟：`chunk_at` 是 `{chunk_id: Chunk}` 的语料快照，邻块完全在内存里取、一次库都
    不查。（线上工具层那份要查 SQLite，见 Task 5；两边靠 `scripts/retrieval_tool_smoke.py` 的
    同名参数逐题对账。）

    `anchors <= 0` 或 `top_k <= 0` 返回空列表——调用方说「不扩窗」时不该产出任何东西。
    锚点不在 `chunk_at` 里（跨库混用、或该块不在语料快照中）就跳过它，而不是报错：缺一条锚点
    不该让整次测量失败。
    """

    if anchors <= 0 or top_k <= 0:
        return []
    by_doc: dict[str, dict[int, str]] = {}
    for chunk in chunk_at.values():
        by_doc.setdefault(chunk.doc_id, {})[chunk.ordinal] = chunk.chunk_id
    picked: list[str] = []
    seen: set[str] = set()
    for hit in hits[:anchors]:
        anchor = chunk_at.get(_chunk_id(hit))
        if anchor is None:
            continue
        siblings = by_doc.get(anchor.doc_id, {})
        for offset in _window_offsets(width):
            neighbour = siblings.get(anchor.ordinal + offset)
            if neighbour is not None and neighbour not in seen:
                seen.add(neighbour)
                picked.append(neighbour)
        if len(picked) >= top_k:
            break
    return picked[:top_k]


def fused(vector: list[Any], bm25: list[Any], depth: int) -> list[Any]:
    """按工具层的真实口径融合：两路各取 depth，RRF 再截 depth。"""

    return rrf_fuse([vector[:depth], bm25[:depth]], top_k=depth)


def merge_ranked(lists: list[list[Any]]) -> list[Any]:
    """把「多次检索各自的排名」并成一条序列，按**最好名次**排序。

    一次 `ours` 运行可能派多个 retrieve 子任务，检索池是它们的并集；只有并起来才能对上
    实测的「池子里有没有 gold」。`baseline_a` 的 12 步 ReAct 也是同理（它天生并了十几次）。
    """

    best_rank: dict[str, int] = {}
    order: dict[str, Any] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            order.setdefault(hit.chunk_id, hit)
            best_rank[hit.chunk_id] = min(best_rank.get(hit.chunk_id, rank), rank)
    return sorted(order.values(), key=lambda hit: best_rank[hit.chunk_id])


def mrr_at(rows: list[dict[str, Any]], config: str, k: int) -> float:
    """MRR@k：该题最好 gold 名次取倒数、对全部题求均值（MISS 记 0）。

    `ranks[config]` 已经是「该配置自己的深度内、所有 gold 里最好的名次」，所以这里只做截断，
    不碰检索。**汇总因此是名次的纯函数**——加了新汇总列不必重付嵌入钱（见 `--from-json`）。
    深度 8 的配置里名次天生 ≤8，所以它的 `mrr@50` 就等于 `mrr@8`，不是两套口径。
    """

    total = sum(
        1.0 / row["ranks"][config]
        for row in rows
        if row["ranks"][config] and row["ranks"][config] <= k
    )
    return total / len(rows)


def summarize(
    rows: list[dict[str, Any]], depth_of: dict[str, int]
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    """把逐题明细折成三张汇总表：命中率/recall/MRR、文档级召回、邻块窗口。

    只读 `rows` 与 `depth_of`——这就是 `--from-json` 能等价复算的原因。
    """

    summary = {
        config: {
            "hit@8": sum(1 for row in rows if (row["ranks"][config] or 999) <= 8),
            "hit@50": sum(1 for row in rows if (row["ranks"][config] or 9999) <= 50),
            # 标准 recall = 25 道题的「命中 gold 数 / gold 总数」的均值，与主表同口径。
            # 各配置的截断深度不同（8 或 50），`recalls` 就是它自己那一档的 recall。
            "recall_k": depth_of[config],
            "recall": round(sum(row["recalls"][config] for row in rows) / len(rows), 4),
            "mrr@8": round(mrr_at(rows, config, 8), 4),
            "mrr@50": round(mrr_at(rows, config, 50), 4),
            "total": len(rows),
        }
        for config in rows[0]["ranks"]
    }
    doc_summary = {
        config: sum(1 for row in rows if row["doc_ranks"][config] is not None)
        for config in rows[0]["doc_ranks"]
    }
    window_summary = {
        name: sum(1 for row in rows if row["window"][name]["covered"])
        for name in rows[0]["window"]
    }
    return summary, doc_summary, window_summary


def print_summaries(
    summary: dict[str, Any],
    doc_summary: dict[str, int],
    window_summary: dict[str, int],
    depth: int,
) -> None:
    """三张汇总表的唯一打印口径（真跑与 `--from-json` 共用，避免两处漂移）。"""

    first = next(iter(summary.values()))
    print(
        f"\n[summary] {first['total']} 条可答题：命中率（gold 进池的**题数**）"
        "+ 标准 recall（命中的 gold **条数**占比）+ MRR"
    )
    for config, stat in summary.items():
        print(
            f"  {config:<34} top-8 {stat['hit@8']:>2}/{stat['total']}   top-{depth} {stat['hit@50']:>2}/{stat['total']}"
            f"   recall@{stat['recall_k']} {stat['recall']:>6.1%}"
            f"   MRR@8 {stat['mrr@8']:.4f}   MRR@{depth} {stat['mrr@50']:.4f}"
        )

    print("\n[summary] 文档级召回（gold 所在文档进池的条数）")
    for config, count in doc_summary.items():
        print(f"  {config:<34} {count:>2}/{first['total']}")

    print("\n[summary] 「命中文档 + 邻块窗口」能覆盖 gold 的条数（查询=原问句，深 50）")
    for name, count in window_summary.items():
        print(f"  {name:<34} {count:>2}/{first['total']}")


def resummarize(source: Path, destination: Path | None) -> int:
    """用落盘 JSON 里的逐题名次重算汇总，零检索、零模型调用。

    存在的理由：给汇总加一列（比如这次的 MRR）不该逼着重跑一遍嵌入查询。名次是原始事实、
    汇总只是它的函数，所以从 `rows` 复算不违反「数字只能由评测记录渲染」这条护栏——
    它不会凭空产生名次，只会把已有名次重新折算。
    """

    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["rows"]
    depth_of = {config: stat["recall_k"] for config, stat in payload["summary"].items()}
    summary, doc_summary, window_summary = summarize(rows, depth_of)
    payload["summary"] = summary
    payload["doc_summary"] = doc_summary
    payload["window_summary"] = window_summary
    # 把「每条配置自己的深度」也落盘，复算路径就不必再从旧汇总里猜。
    payload["config_depth"] = depth_of
    print_summaries(summary, doc_summary, window_summary, payload["depth"])
    target = destination or source
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {target}（复用既有名次，未重新检索）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读检索层 A/B（零 LLM 花费）")
    parser.add_argument("--taskset", type=Path, default=DEFAULT_TASKSET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument(
        "--chroma-v2",
        type=Path,
        default=None,
        help="对照向量库（向量侧带文档身份前缀重嵌的那份）；给了才加 +vmeta 那几列",
    )
    parser.add_argument(
        "--chroma-v3",
        type=Path,
        default=None,
        help="unit 级向量库（400 字表格行级切块 + 导读行重嵌）；给了才加 +v3 那几列",
    )
    parser.add_argument(
        "--unit-map",
        type=Path,
        default=DEFAULT_UNIT_MAP,
        help="v3 的 unit → 父块 id 映射；只有给了 --chroma-v3 时才读",
    )
    parser.add_argument("--join", type=Path, default=DEFAULT_JOIN)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=None, help="把逐题明细落成 JSON")
    parser.add_argument("--depth", type=int, default=50, help="深召回深度（默认 50）")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="不重新检索：读已有 A/B JSON，用落盘的逐题名次重算汇总（含 MRR）并覆盖写回",
    )
    args = parser.parse_args(argv)

    # 复算通路：加了汇总列之后只想刷新汇总时用它，避免为一行 MRR 再付一次嵌入钱。
    if args.from_json:
        return resummarize(args.from_json, args.json)

    started = time.perf_counter()
    tasks = load_jsonl(args.taskset)
    chunks = load_chunks(args.chunks)
    goals = load_ours_goals(args.db, args.join)
    metas = load_doc_meta(args.manifest)
    print(
        f"[load] 任务 {len(tasks)} 条、语料 {len(chunks)} chunk、"
        f"retrieve 目标 {len(goals)} 条、文档身份 {len(metas)} 份"
    )

    answerable = [task for task in tasks if task.get("answerable") and task.get("gold_chunk_ids")]
    absent = [task["task_id"] for task in answerable if task["task_id"] not in goals]
    if absent:
        raise AbError(f"这些题在配对结果里没有对应运行：{absent}")
    # 有一次运行（core-sh2-003）的 plan 是**空的**——一个子任务都没派，直接给拒答文案。
    # 那次的「真实查询」不存在，只能用原问句代替，并把这件事标出来而不是悄悄跳过。
    empty_plan = [task["task_id"] for task in answerable if not goals[task["task_id"]]]
    if empty_plan:
        print(f"[warn] 这些运行没有 retrieve 子任务（plan 为空），查询退化为原问句：{empty_plan}")
    print(f"[scope] 可答题 {len(answerable)} 条（拒答题不参与 gold 排名）")

    raw = Bm25Index()
    raw.build(chunks)
    normalized_texts = {chunk.chunk_id: normalize_cjk(chunk.text) for chunk in chunks}
    flipped = sum(1 for chunk in chunks if normalized_texts[chunk.chunk_id] != chunk.text)
    norm_chunks = [
        dataclasses.replace(chunk, text=normalized_texts[chunk.chunk_id]) for chunk in chunks
    ]
    norm = Bm25Index()
    norm.build(norm_chunks)
    # 带身份前缀的那一份：索引文本 = 「公司（代码）报告期 标题 章节」+ 原文，
    # 但 `search()` 返回的 `RetrievalHit.text` 仍是原文（前缀只进打分矩阵）。
    meta = Bm25Index(doc_metas=metas)
    meta.build(chunks)
    sample = next(chunk for chunk in chunks if chunk.doc_id in metas)
    print(f"[bm25] 归一化改动 {flipped}/{len(chunks)} chunk 的文本；三份索引已在内存建好")
    print(f"[metaprefix] 样例：{meta.index_text_of(sample).splitlines()[0]!r}")

    # 表格导读行那一份：与 `+meta` 共用同一套输入，只在索引文本里多插一段导读行。
    # 规则来自 `akr.ingest.table_context`（Task 7 的向量侧重嵌复用同一份，口径只能有一份）。
    table = Bm25Index(doc_metas=metas, index_text_fn=make_table_index_text_fn(metas))
    table.build(chunks)
    lead_sample = next((chunk for chunk in chunks if table_lead(chunk.text)), None)
    lead_text = table_lead(lead_sample.text) if lead_sample is not None else "(全语料没有表格行)"
    print(f"[tablelead] 样例：{lead_text!r}")

    vectors = VectorStore(args.chroma, build_embedder())
    print(f"[vector] Chroma 计数 {vectors.count()}")
    # 「向量侧也带身份前缀」是**另一份库**（前缀只进嵌入文本、进不了查询向量），
    # 所以对照必须在同一次运行里同时持有两份向量库：两列共用同一套查询、同一份 BM25、
    # 同一个收窄结果，差异只可能来自「库内向量是怎么嵌出来的」。
    vectors_v2 = None
    if args.chroma_v2 is not None:
        vectors_v2 = VectorStore(args.chroma_v2, build_embedder())
        print(f"[vector] Chroma v2（文档身份前缀）计数 {vectors_v2.count()}")
    # v3 是 unit 级索引：`id_map` 把 unit 名次折回父块，`chunk_id` 因此仍落在 gold 空间里
    # （缺映射必须报错——静默不折叠会让这一列以「召回率归零」的形状骗过所有检查）。
    vectors_v3 = None
    if args.chroma_v3 is not None:
        if not args.unit_map.exists():
            raise AbError(
                f"缺 unit → 父块 映射：{args.unit_map}（先跑 scripts\\build_unit_chunks.py）"
            )
        unit_map = load_unit_map(args.unit_map)
        vectors_v3 = VectorStore(args.chroma_v3, build_embedder(), id_map=unit_map)
        print(
            f"[vector] Chroma v3（unit 级 + 导读行）计数 {vectors_v3.count()}，"
            f"折叠映射 {len(unit_map)} 条"
        )

    ordinal_of = {chunk.chunk_id: chunk.ordinal for chunk in chunks}
    doc_of = {chunk.chunk_id: chunk.doc_id for chunk in chunks}
    chunk_at = {chunk.chunk_id: chunk for chunk in chunks}
    embed_cache: dict[str, list[Any]] = {}

    def vector_hits(text: str) -> list[Any]:
        if text not in embed_cache:
            embed_cache[text] = vectors.query(text, top_k=args.depth)
        return embed_cache[text]

    stage2_cache: dict[tuple[str, tuple[str, ...]], list[Any]] = {}

    def vector_hits_in(text: str, doc_ids: tuple[str, ...]) -> list[Any]:
        key = (text, doc_ids)
        if key not in stage2_cache:
            stage2_cache[key] = vectors.query(text, top_k=args.depth, doc_ids=doc_ids)
        return stage2_cache[key]

    embed_cache_v2: dict[str, list[Any]] = {}
    stage2_cache_v2: dict[tuple[str, tuple[str, ...]], list[Any]] = {}

    def vector_hits_v2(text: str) -> list[Any]:
        assert vectors_v2 is not None
        if text not in embed_cache_v2:
            embed_cache_v2[text] = vectors_v2.query(text, top_k=args.depth)
        return embed_cache_v2[text]

    def vector_hits_in_v2(text: str, doc_ids: tuple[str, ...]) -> list[Any]:
        assert vectors_v2 is not None
        key = (text, doc_ids)
        if key not in stage2_cache_v2:
            stage2_cache_v2[key] = vectors_v2.query(text, top_k=args.depth, doc_ids=doc_ids)
        return stage2_cache_v2[key]

    embed_cache_v3: dict[str, list[Any]] = {}
    stage2_cache_v3: dict[tuple[str, tuple[str, ...]], list[Any]] = {}

    def vector_hits_v3(text: str) -> list[Any]:
        assert vectors_v3 is not None
        if text not in embed_cache_v3:
            embed_cache_v3[text] = vectors_v3.query(text, top_k=args.depth)
        return embed_cache_v3[text]

    def vector_hits_in_v3(text: str, doc_ids: tuple[str, ...]) -> list[Any]:
        assert vectors_v3 is not None
        key = (text, doc_ids)
        if key not in stage2_cache_v3:
            stage2_cache_v3[key] = vectors_v3.query(text, top_k=args.depth, doc_ids=doc_ids)
        return stage2_cache_v3[key]

    def top_docs(hits: list[Any], count: int) -> tuple[str, ...]:
        """融合结果里前 `count` 个**不同文档**（保持名次顺序）。"""

        ordered: list[str] = []
        for hit in hits:
            if hit.doc_id not in ordered:
                ordered.append(hit.doc_id)
            if len(ordered) >= count:
                break
        return tuple(ordered)

    def two_stage(
        text: str,
        ranked: list[Any],
        docs: int,
        top_k: int,
        *,
        bm25_index: Any = None,
        vector: Any = None,
    ) -> list[Any]:
        """两段式：先用无过滤检索定文档，再把这些文档当过滤条件重搜一次。

        这是「命中文档之后怎么在文档内定位」的第二种答案——不靠邻块窗口，靠
        带 `doc_ids` 过滤的二次召回（工具层已支持该参数，无需新模型调用）。

        `bm25_index` 让「第一段带身份前缀」的变体也能走这条路径——线上 BM25 只有一份
        索引，它的索引文本要么带前缀要么不带，不存在两套。

        `vector` 同理，用来把「向量侧也带身份前缀」的那份库接进同一条流水线
        （默认仍是旧库的 `vector_hits_in`）。
        """

        vector_fn = vector or vector_hits_in
        picked = top_docs(ranked, docs)
        if not picked:
            return []
        return rrf_fuse(
            [
                vector_fn(text, picked)[:top_k],
                list((bm25_index or raw).search(text, top_k=top_k, doc_ids=picked))[:top_k],
            ],
            top_k=top_k,
        )

    rows: list[dict[str, Any]] = []
    depth_of: dict[str, int] = {}
    for task in answerable:
        gold = list(task["gold_chunk_ids"])
        question = str(task["question"])
        goal_texts = goals[task["task_id"]] or [question]
        query_goal = goal_texts[0]

        v_goal = merge_ranked([vector_hits(text) for text in goal_texts])
        b_goal = merge_ranked([raw.search(text, top_k=args.depth) for text in goal_texts])
        v_question = vector_hits(question)
        b_question = raw.search(question, top_k=args.depth)
        b_question_norm = norm.search(question, top_k=args.depth)
        b_question_norm_bigram = norm.search(
            question + " " + " ".join(bigram_tokens(question)), top_k=args.depth
        )
        # 修法⑤：BM25 索引文本带文档身份（向量侧**不动**——前缀只进 BM25 的打分矩阵，
        # 所以这一列就是「只改代码、不重新嵌入」的线上可达状态）。
        b_question_meta = meta.search(question, top_k=args.depth)
        # 修法⑦：在⑤之上，把「表格导读行」也插进 BM25 的索引文本（零嵌入花费；
        # 向量侧要拿到同一效果必须重嵌，那是 Task 7，默认不执行）。
        b_question_meta_table = table.search(question, top_k=args.depth)
        # 修法⑥：在⑤之上，用问句抽出的代码/年份/公司名把候选文档收窄。
        hints = extract_hints(question, metas)
        narrowed = narrow_doc_ids(hints, metas)
        v_question_narrow = vector_hits_in(question, narrowed) if narrowed else []
        b_question_meta_narrow = (
            list(meta.search(question, top_k=args.depth, doc_ids=narrowed)) if narrowed else []
        )
        b_question_meta_narrow_table = (
            list(table.search(question, top_k=args.depth, doc_ids=narrowed)) if narrowed else []
        )
        # 同一套查询、同一套收窄，只换「库内向量是不是带身份前缀建的」——v2 那几列的全部
        # 差异来源都在这里，所以两边必须共用 `narrowed` 与 `b_question_meta_narrow`。
        v_question_narrow_v2 = (
            vector_hits_in_v2(question, narrowed) if vectors_v2 is not None and narrowed else []
        )
        fused_question_meta_narrow_v2 = (
            fused(v_question_narrow_v2, b_question_meta_narrow, args.depth)
            if vectors_v2 is not None
            else []
        )
        # v3 与 v2 的唯一差别是「库内向量怎么来的」（unit 级切块 + 导读行），查询、收窄、
        # BM25 一律共用上面那几个变量；命中已被 `id_map` 折回父块，两边口径可比。
        v_question_narrow_v3 = (
            vector_hits_in_v3(question, narrowed) if vectors_v3 is not None and narrowed else []
        )
        fused_question_meta_narrow_v3 = (
            fused(v_question_narrow_v3, b_question_meta_narrow, args.depth)
            if vectors_v3 is not None
            else []
        )
        gold_docs = {doc_of[gid] for gid in gold if gid in doc_of}
        fused_question = fused(v_question, b_question, args.depth)
        fused_question_meta = fused(v_question, b_question_meta, args.depth)
        fused_question_meta_narrow = fused(
            v_question_narrow, b_question_meta_narrow, args.depth
        )
        fused_question_meta_narrow_table = fused(
            v_question_narrow, b_question_meta_narrow_table, args.depth
        )
        fused_goal = fused(v_goal, b_goal, args.depth)
        # 多查询融合：把「原问句」与「规划子目标」两套查询各自的向量/BM25 四路并起来再 RRF。
        # 这是候选池的**并集**，用来回答「修法④要不要替换掉 goal，还是两者并存」。
        fused_union_8 = rrf_fuse(
            [v_goal[:8], b_goal[:8], v_question[:8], b_question[:8]], top_k=8
        )
        fused_union_depth = rrf_fuse(
            [
                v_goal[: args.depth],
                b_goal[: args.depth],
                v_question[: args.depth],
                b_question[: args.depth],
            ],
            top_k=args.depth,
        )
        windows = {
            f"window±{width}": window_hit(gold, doc_of, fused_question, ordinal_of, width)
            for width in (2, 5, 10)
        }
        # 真实实现里只会在融合结果的前几个锚点周围展开（prompt 成本护栏），
        # 所以除了「全部命中都能当锚点」以外，另测 top-3 锚点这一档。
        # 关键问题：锚点从**浅池（top-8）**取还是从**深池（top-50）**取。
        # 浅池只有 8 个锚点、且都很远；深池里 gold 自己或它的邻居常常就在其中。
        for label, ranked in (
            ("union@8", fused_union_8),
            ("q@50", fused_question),
            ("g@50", fused_goal),
        ):
            for width in (2, 5):
                for anchors in (3, 5):
                    windows[f"{label}±{width}/{anchors}anchors"] = window_hit(
                        gold, doc_of, ranked[:anchors], ordinal_of, width
                    )
            windows[f"{label}±2/all"] = window_hit(gold, doc_of, ranked, ordinal_of, 2)
        # 线上只会调一次工具，返回的必须是「一段+二段合并后的前 8 条」——否则
        # RecordingGateway 记下的顺序与模型真正看到的证据不一致（指标就会失真）。
        stage2_hits = two_stage(question, fused_question, 3, 8)
        merged_m3 = merge_ranked([fused_question[:8], stage2_hits])[:8]
        windows["merged@8/M3±2/all"] = window_hit(gold, doc_of, merged_m3, ordinal_of, 2)
        # 线上生产配置这一轮的命中（`stage2@8/M3+meta+narrow`）：扩窗列的锚点直接取它，
        # 免得 `by_config` 与 `windows` 两处各算一遍（同一份事实只留一个变量）。
        prod_hits = two_stage(question, fused_question_meta_narrow, 3, 8, bm25_index=meta)

        # 每条配置 = (命中列表, 该列表自身的截断深度)。名次与标准 recall 都从同一个列表
        # 派生，避免「算名次用一份命中、算 recall 用另一份」这种两处口径漂移。
        by_config: dict[str, tuple[list[Any], int]] = {
            "goals@8": (fused(v_goal, b_goal, 8), 8),
            "goals@50": (fused(v_goal, b_goal, args.depth), args.depth),
            "question@8": (fused(v_question, b_question, 8), 8),
            "question@50": (fused(v_question, b_question, args.depth), args.depth),
            "question@8+norm": (fused(v_question, b_question_norm, 8), 8),
            "question@50+norm": (fused(v_question, b_question_norm, args.depth), args.depth),
            "question@50+norm+bigram": (
                fused(v_question, b_question_norm_bigram, args.depth),
                args.depth,
            ),
            "vector@50": (v_question, args.depth),
            "bm25@50": (b_question, args.depth),
            "bm25@50+norm": (b_question_norm, args.depth),
            "union@8": (fused_union_8, 8),
            "union@50": (fused_union_depth, args.depth),
            "stage2@8/M1": (two_stage(question, fused_question, 1, 8), 8),
            "stage2@8/M3": (two_stage(question, fused_question, 3, 8), 8),
            "stage2@8/M5": (two_stage(question, fused_question, 5, 8), 8),
            "stage2@50/M3": (two_stage(question, fused_question, 3, args.depth), args.depth),
            "stage2@8/M3/goal": (two_stage(query_goal, fused_goal, 3, 8), 8),
            "merged@8/M3": (merged_m3, 8),
            # ---- 2026-09-25 新增：文档身份前缀（+meta）与查询收窄（+meta+narrow）----
            "bm25@8+meta": (b_question_meta[:8], 8),
            "bm25@50+meta": (b_question_meta, args.depth),
            "question@8+meta": (fused(v_question, b_question_meta, 8), 8),
            "question@50+meta": (fused_question_meta, args.depth),
            "question@8+meta+narrow": (fused_question_meta_narrow[:8], 8),
            "question@50+meta+narrow": (fused_question_meta_narrow, args.depth),
            # ---- 2026-09-25 新增：表格导读行（+table）----
            # 三列与 `+meta` 三列一一对应，唯一差别是索引文本里多了一段导读行：问句、收窄、
            # 池深、两段式参数全部相同，所以差异只可能来自「表格行有没有进打分文本」。
            "bm25@50+meta+table": (b_question_meta_table, args.depth),
            "question@50+meta+narrow+table": (fused_question_meta_narrow_table, args.depth),
            "stage2@8/M3+meta+narrow+table": (
                two_stage(question, fused_question_meta_narrow_table, 3, 8, bm25_index=table),
                8,
            ),
            "stage2@8/M3+meta": (
                two_stage(question, fused_question_meta, 3, 8, bm25_index=meta),
                8,
            ),
            "stage2@8/M3+meta+narrow": (prod_hits, 8),
            # `merged@*` 走的是 `merge_ranked`（按**最好名次**把两段并成一条序列），这是离线
            # 「候选池并集」的形状，**工具层不实现它**（线上是「二段结果在前 + 一段额外命中
            # 按序补尾」，工具层 `TWO_STAGE_EXTRA_HITS=4`）。2026-09-25 晚复核过：逐题对得上
            # 线上工具层的是 `stage2@8/M3+meta+narrow{+vmeta}` 那两列，`merged@*` 两边都不是
            # ——所以 `merged@8/M3+meta+narrow+vmeta` 的 15/25 **不能当线上成绩引用**
            # （线上同库同问句实测 12/25）。留着它是为了记「还有这种并集形状」这个负结果。
            "merged@8/M3+meta": (
                merge_ranked(
                    [fused_question_meta[:8], two_stage(question, fused_question_meta, 3, 8, bm25_index=meta)]
                )[:8],
                8,
            ),
            "merged@8/M3+meta+narrow": (
                merge_ranked([fused_question_meta_narrow[:8], prod_hits])[:8],
                8,
            ),
        }
        # ---- 2026-09-25 晚：向量侧也带身份前缀（需要 --chroma-v2，那份库重嵌过）----
        # 只有给了 v2 库才加列：没给就不该在 JSON 里留下一条「看起来测过」的空列。
        if vectors_v2 is not None:
            by_config["vector@50+vmeta"] = (vector_hits_v2(question), args.depth)
            by_config["question@50+meta+narrow+vmeta"] = (
                fused_question_meta_narrow_v2,
                args.depth,
            )
            by_config["stage2@8/M3+meta+narrow+vmeta"] = (
                two_stage(
                    question,
                    fused_question_meta_narrow_v2,
                    3,
                    8,
                    bm25_index=meta,
                    vector=vector_hits_in_v2,
                ),
                8,
            )
        # ---- 2026-09-25 深夜：unit 级重嵌（需要 --chroma-v3，那份库是 400 字行级 unit）----
        # `vector@50+v3` 就是用户要盯的那个数：**纯向量侧**的深召回，与 `vector@50`（9/25）
        # 逐题同比。`+v3` 与 `+vmeta` 的唯一差别是库内向量，列名后缀必须一眼可分辨。
        if vectors_v3 is not None:
            by_config["vector@50+v3"] = (vector_hits_v3(question), args.depth)
            by_config["question@50+meta+narrow+v3"] = (
                fused_question_meta_narrow_v3,
                args.depth,
            )
            by_config["stage2@8/M3+meta+narrow+v3"] = (
                two_stage(
                    question,
                    fused_question_meta_narrow_v3,
                    3,
                    8,
                    bm25_index=meta,
                    vector=vector_hits_in_v3,
                ),
                8,
            )
            by_config["merged@8/M3+meta+narrow+vmeta"] = (
                merge_ranked(
                    [
                        fused_question_meta_narrow_v2[:8],
                        two_stage(
                            question,
                            fused_question_meta_narrow_v2,
                            3,
                            8,
                            bm25_index=meta,
                            vector=vector_hits_in_v2,
                        ),
                    ]
                )[:8],
                8,
            )
        # ---- 2026-09-25 晚：邻块扩窗的「真扩、再截回 8」效果（Task 4）----
        # 与上面 `window±*` / `q@50±*` 那几列**不是一回事**：那几列只问「gold 在不在锚点
        # ±width 之内」（上界），这一列问「真的把邻块展开、再截回 8 之后，gold 还在不在返回
        # 列表里」。宽窗口上两者必然分叉——扩出来的邻块会把排在后面的锚点挤出前 8，而线上
        # 不可能不截断（README 全局约束「扩完必须截回固定条数」）。锚点分两档：
        #   * `q50`  = 深池收窄融合（`question@50+meta+narrow`，池内 20/25 含 gold）；
        #   * `prod` = 线上返回给模型的那 8 条（`stage2@8/M3+meta+narrow`，12/25）。
        for label, ranked in (("q50", fused_question_meta_narrow), ("prod", prod_hits)):
            for width in (2, 5):
                for anchors in (3, 5):
                    name = f"expand@8/{label}±{width}/{anchors}anchors"
                    ids = expand_window(
                        ranked, anchors=anchors, width=width, top_k=8, chunk_at=chunk_at
                    )
                    by_config[name] = (ids, 8)
                    # `covered` 与这一列的 `hit@8` 恒等（同一个 id 列表算出来的）；两处都写，
                    # 是为了让「锚点窗口」与「真扩窗」在同张表里并排比。`anchor_rank` 对扩窗列
                    # 没有意义（锚点名次由并排的 `q@50±*` 列回答），按形状要求固定写 None。
                    windows[name] = (any(gid in set(ids) for gid in gold), None)
        depth_of.update({name: k for name, (_, k) in by_config.items()})
        row: dict[str, Any] = {
            "task_id": task["task_id"],
            "type": task["type"],
            "gold_chunk_ids": gold,
            "goal": query_goal,
            "question": question,
            "hints": {
                "codes": list(hints.codes),
                "years": list(hints.years),
                "companies": list(hints.companies),
                "narrowed": list(narrowed),
            },
            "ranks": {name: rank_of(gold, hits) for name, (hits, _) in by_config.items()},
            "recalls": {
                name: round(recall_at(gold, hits, k), 4) for name, (hits, k) in by_config.items()
            },
            "doc_ranks": {
                "question@8": doc_rank_of(gold_docs, fused_question[:8]),
                "question@50": doc_rank_of(gold_docs, fused_question),
                "goals@50": doc_rank_of(gold_docs, fused_goal),
            },
            "window": {
                name: {"covered": covered, "anchor_rank": rank}
                for name, (covered, rank) in windows.items()
            },
        }
        rows.append(row)
        print(
            f"[{task['task_id']:>14}] goals@8={row['ranks']['goals@8']} "
            f"q@8={row['ranks']['question@8']} q@50={row['ranks']['question@50']} "
            f"q@50+norm={row['ranks']['question@50+norm']} "
            f"vec@50={row['ranks']['vector@50']} bm25@50={row['ranks']['bm25@50']} "
            f"meta+n={row['ranks']['stage2@8/M3+meta+narrow']}"
            f"(docs={len(narrowed)})"
        )

    summary, doc_summary, window_summary = summarize(rows, depth_of)
    print_summaries(summary, doc_summary, window_summary, args.depth)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "depth": args.depth,
                    "answerable": len(rows),
                    "summary": summary,
                    "doc_summary": doc_summary,
                    "window_summary": window_summary,
                    "config_depth": depth_of,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[write] {args.json}")

    caches = f"{len(embed_cache)} 条" + (
        f" + v2 库 {len(embed_cache_v2)} 条" if vectors_v2 is not None else ""
    ) + (f" + v3 库 {len(embed_cache_v3)} 条" if vectors_v3 is not None else "")
    print(f"[done] 耗时 {time.perf_counter() - started:.1f}s；嵌入查询去重后 {caches}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
