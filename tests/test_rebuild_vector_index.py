"""`scripts/rebuild_vector_index.py` 的单测。

与 `tests/test_ingest_corpus.py` 一致：脚本不在 `src/` 下，用 importlib 按路径加载。
这里刻意不触发真实链路（不调硅基流动、不读生产 `data/`），真链路由脚本自己跑。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from akr.ingest.chunker import Chunk
from akr.storage import sqlite_store

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rebuild_vector_index.py"


def _load():
    spec = importlib.util.spec_from_file_location("rebuild_vector_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # 先注册进 sys.modules：`from __future__ import annotations` 下 dataclass 解析字符串
    # 注解时要回头查 sys.modules[cls.__module__]。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rebuild_vector_index = _load()


def fake_embed(texts: list[str]) -> list[list[float]]:
    """确定性假嵌入：按关键词出现次数构造 3 维向量（与 tests/storage/test_vector_store.py 同款）。"""

    return [[float(t.count("营收")), float(t.count("员工")), 1.0] for t in texts]


def make_chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c{ordinal:04d}",
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        ordinal=ordinal,
        text=text,
        content_hash=f"hash{ordinal:04d}",
        section_path="第一章",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


def seed_db(tmp_path: Path, chunks: list[Chunk]) -> Path:
    """把给定的分块灌进一个临时 SQLite——重建脚本只认这张表。"""

    db = tmp_path / "akr.sqlite"
    conn = sqlite_store.connect(db)
    sqlite_store.init_schema(conn)
    for doc_id in dict.fromkeys(chunk.doc_id for chunk in chunks):
        sqlite_store.upsert_document(
            conn,
            doc_id=doc_id,
            source=f"{doc_id}.pdf",
            doc_type="annual_report",
            page_count=1,
            updated_at="2026-09-25T00:00:00+00:00",
        )
    for doc_id in dict.fromkeys(chunk.doc_id for chunk in chunks):
        sqlite_store.replace_chunks(
            conn, doc_id, [chunk for chunk in chunks if chunk.doc_id == doc_id]
        )
    conn.close()
    return db


def test_load_chunks_is_read_only_and_ordered(tmp_path):
    """读取顺序必须稳定（`doc_id, ordinal`），否则重建的分批边界每次都不一样。"""

    db = seed_db(
        tmp_path,
        [
            make_chunk("doc-b", 0, "营收 3"),
            make_chunk("doc-a", 1, "营收 2"),
            make_chunk("doc-a", 0, "营收 1"),
        ],
    )
    chunks = rebuild_vector_index.load_chunks(db)
    assert [chunk.chunk_id for chunk in chunks] == [
        "doc-a#c0000",
        "doc-a#c0001",
        "doc-b#c0000",
    ]

    # 只读由 SQLite 保证，而不是靠调用方自觉
    conn = sqlite_store.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
        before = conn.execute("SELECT updated_at FROM documents").fetchall()
    finally:
        conn.close()
    assert len(before) == 2


def test_load_chunks_reports_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到分块库"):
        rebuild_vector_index.load_chunks(tmp_path / "nope.sqlite")


def test_rebuild_writes_every_chunk_and_probe_hits(tmp_path, capsys):
    db = seed_db(
        tmp_path,
        [
            make_chunk("doc1", 0, "公司营收 100 亿元"),
            make_chunk("doc1", 1, "员工人数 5000 人"),
            make_chunk("doc1", 2, "公司营收 200 亿元"),
        ],
    )
    chroma = tmp_path / "chroma"

    stats = rebuild_vector_index.rebuild(
        db_path=db, chroma_path=chroma, embed_fn=fake_embed, batch_size=2
    )

    assert stats["chunks"] == 3
    assert stats["written"] == 3
    # 探针取的是中间那条（ordinal 1），必须至少有一条命中——否则「能读但查不到」会漏过去
    assert stats["hits"]
    assert "doc1#c0001" in stats["hits"]
    assert "[verify] 条数 3/3" in capsys.readouterr().out


def test_rebuild_refuses_to_overwrite_existing_index(tmp_path):
    """已有向量时默认拒绝：重建通常指向空目录，`--force` 才是「覆盖」的显式授权。"""

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    chroma = tmp_path / "chroma"
    rebuild_vector_index.rebuild(db_path=db, chroma_path=chroma, embed_fn=fake_embed)

    with pytest.raises(ValueError, match="已有 1 条向量"):
        rebuild_vector_index.rebuild(db_path=db, chroma_path=chroma, embed_fn=fake_embed)

    stats = rebuild_vector_index.rebuild(
        db_path=db, chroma_path=chroma, embed_fn=fake_embed, force=True
    )
    assert stats["written"] == 1


def test_rebuild_rejects_empty_chunk_table(tmp_path):
    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    conn = sqlite_store.connect(db)
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="chunks 表是空的"):
        rebuild_vector_index.rebuild(
            db_path=db, chroma_path=tmp_path / "chroma", embed_fn=fake_embed
        )


def test_unreadable_index_becomes_actionable_error(tmp_path, monkeypatch):
    """坏索引（HNSW 段丢失）必须变成带处置建议的 `IndexUnreadable`，不是裸 InternalError。"""

    class BrokenStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def count(self):
            raise RuntimeError("Error constructing hnsw segment reader")

    monkeypatch.setattr(rebuild_vector_index, "VectorStore", BrokenStore)
    with pytest.raises(rebuild_vector_index.IndexUnreadable, match="HNSW 段损坏"):
        rebuild_vector_index.open_store(tmp_path / "chroma", fake_embed)


def test_check_reports_counts_without_writing(tmp_path, capsys):
    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    chroma = tmp_path / "chroma"
    rebuild_vector_index.rebuild(db_path=db, chroma_path=chroma, embed_fn=fake_embed)

    result = rebuild_vector_index.check(db_path=db, chroma_path=chroma, embed_fn=fake_embed)

    assert result == {"chunks": 1, "vectors": 1, "hits": ["doc1#c0000"]}
    assert "[check] SQLite 分块 1 条" in capsys.readouterr().out


def test_rebuild_with_index_text_fn_embeds_the_prefixed_text(tmp_path, capsys):
    """`--doc-prefix` 的真身：库内向量由「前缀 + 原文」嵌出来，条数与探针仍要过。"""

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    seen: list[list[str]] = []

    def recording_embed(texts: list[str]) -> list[list[float]]:
        seen.append(list(texts))
        return fake_embed(texts)

    stats = rebuild_vector_index.rebuild(
        db_path=db,
        chroma_path=tmp_path / "chroma",
        embed_fn=recording_embed,
        index_text_fn=lambda chunk: f"甲公司（002129） 2024年年度\n{chunk.text}",
    )

    assert seen[0] == ["甲公司（002129） 2024年年度\n公司营收 100 亿元"]
    assert stats["doc_prefix"] is True
    assert stats["raw_chars"] == len("公司营收 100 亿元")
    out = capsys.readouterr().out
    assert "身份前缀生效" in out
    assert "0/1 条超过 500 字符" in out


def test_resolve_index_text_fn_requires_a_manifest(tmp_path):
    """开了 `--doc-prefix` 却没有 manifest：必须报错，不能静默退化成嵌原文白花钱。"""

    assert rebuild_vector_index.resolve_index_text_fn(False, tmp_path / "x.json") is None
    with pytest.raises(ValueError, match="需要文档身份"):
        rebuild_vector_index.resolve_index_text_fn(True, tmp_path / "不存在.json")


def test_resolve_index_text_fn_builds_the_prefix_renderer(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "doc1",
                        "company": "甲公司",
                        "code": "002129",
                        "period": "2024年年度",
                        "announcement_title": "2024年年度报告",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    render = rebuild_vector_index.resolve_index_text_fn(True, manifest)
    assert render is not None
    assert render(make_chunk("doc1", 0, "正文")).startswith("甲公司（002129） 2024年年度")
    # 身份表里没有的文档退化成原文，不会拼出一个空前缀
    unknown = rebuild_vector_index.resolve_index_text_fn(True, manifest)
    assert unknown(make_chunk("doc9", 0, "正文")) == "正文"


def test_dry_run_reports_budget_without_an_embedder(tmp_path, capsys):
    """预演必须**不接受 `embed_fn`**：它就是「在没有嵌入器（甚至没有 Key）时也能算花多少」。

    若哪天有人给它加回嵌入器参数，这条测试会因 TypeError 直接失败——这是故意的。
    """

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    stats = rebuild_vector_index.dry_run(
        db_path=db,
        chroma_path=tmp_path / "chroma",
        index_text_fn=lambda chunk: f"甲公司（002129） 2024年年度\n{chunk.text}",
    )

    assert stats["chunks"] == 1
    assert stats["raw_chars"] == len("公司营收 100 亿元")
    assert stats["index_chars"] == len("甲公司（002129） 2024年年度\n公司营收 100 亿元")
    assert stats["clipped"] == 0
    out = capsys.readouterr().out
    assert "身份前缀生效" in out
    assert "[dry-run] 未读 Key" in out
    # 预演不建库：目标目录一个字节都不该被写出来
    assert not (tmp_path / "chroma").exists()


def test_dry_run_cli_needs_no_api_key(tmp_path):
    """脚本直跑 `--dry-run`：**在没有 `.env` 的目录里**也必须 exit 0。

    预演不构造嵌入器，所以真实 Key 的缺失不该拦它；否则「复算文档里的字符数」又要先花钱。
    子进程的 cwd 刻意设在 tmp_path：`EmbeddingSettings.from_env` 读的是相对的 `.env`，
    仓库根目录下那份在预演里不该被读到（读到了就说明预演偷偷构造了嵌入器）。
    """

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "doc1",
                        "company": "甲公司",
                        "code": "002129",
                        "period": "2024年年度",
                        "announcement_title": "2024年年度报告",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("SILICONFLOW")}
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--doc-prefix",
            "--db",
            str(db),
            "--chroma",
            str(tmp_path / "chroma"),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "身份前缀生效" in done.stdout
    assert not (tmp_path / "chroma").exists()


def test_cli_help_runs_as_a_script():
    """脚本直跑一次：守 import 期不炸（`akr` 走可编辑安装），也守中文帮助能输出。"""

    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert done.returncode == 0, done.stderr
    assert "重建" in done.stdout


def table_manifest(tmp_path: Path, doc_id: str = "doc1") -> Path:
    """写一份最小 manifest（与 `test_resolve_index_text_fn_builds_the_prefix_renderer` 同形）。"""

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": doc_id,
                        "company": "甲公司",
                        "code": "002129",
                        "period": "2024年年度",
                        "announcement_title": "2024年年度报告",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_resolve_index_text_fn_table_prefix_requires_a_manifest(tmp_path):
    """`--table-prefix` 的索引文本自带身份前缀，所以它同样要 manifest，且报错要点名这个开关。"""

    assert rebuild_vector_index.resolve_index_text_fn(False, tmp_path / "x.json") is None
    with pytest.raises(ValueError, match="--table-prefix"):
        rebuild_vector_index.resolve_index_text_fn(False, tmp_path / "不存在.json", table=True)


def test_resolve_index_text_fn_builds_the_table_renderer(tmp_path):
    """`--table-prefix` = 身份前缀 → 导读行 → 原文；没有表格行的块逐字等于 `--doc-prefix` 那一档。"""

    manifest = table_manifest(tmp_path)
    render = rebuild_vector_index.resolve_index_text_fn(False, manifest, table=True)
    prefix_render = rebuild_vector_index.resolve_index_text_fn(True, manifest)
    assert render is not None and prefix_render is not None

    body = "营业收入（元） 1 2 3\n1 2 3 4"
    rendered = render(make_chunk("doc1", 0, body))
    assert rendered.startswith("甲公司（002129） 2024年年度")
    assert "营业收入（元） 1 2 3 | 1 2 3 4" in rendered
    assert rendered.endswith(body)
    # 正文块（没有表格行）必须逐字等于只带身份前缀那一档：这是「同一份规则、不叠加」的证据
    assert render(make_chunk("doc1", 0, "正文")) == prefix_render(make_chunk("doc1", 0, "正文"))


def test_guard_target_refuses_a_prefixed_index_on_the_production_path(tmp_path):
    """带前缀的口径不许写生产库：那会让既往的 12/25 与池深曲线失去基准。"""

    with pytest.raises(ValueError, match="生产向量库"):
        rebuild_vector_index.guard_target(
            rebuild_vector_index.PRODUCTION_CHROMA, lambda chunk: chunk.text
        )
    # 不带前缀的普通重建（历史口径）、以及写到别处的带前缀重建，都不受限
    assert rebuild_vector_index.guard_target(rebuild_vector_index.PRODUCTION_CHROMA, None) is None
    assert rebuild_vector_index.guard_target(tmp_path / "chroma", lambda chunk: chunk.text) is None


def test_rebuild_refuses_a_prefixed_index_on_the_production_path(tmp_path, monkeypatch):
    """闸门接在 `rebuild` 的入口上：绕过 CLI 直接调函数也拦得住，且一个字节都不写。"""

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    target = tmp_path / "chrome"
    monkeypatch.setattr(rebuild_vector_index, "PRODUCTION_CHROMA", target)
    with pytest.raises(ValueError, match="生产向量库"):
        rebuild_vector_index.rebuild(
            db_path=db,
            chroma_path=target,
            embed_fn=fake_embed,
            index_text_fn=lambda chunk: f"甲公司（002129） 2024年年度\n{chunk.text}",
        )
    assert not target.exists()


def test_dry_run_cli_with_table_prefix_needs_no_api_key(tmp_path):
    """`--dry-run --table-prefix` 同样不读 Key：换索引文本口径前必须先能免费算出要花多少。"""

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "营业收入（元） 1 2 3\n1 2 3 4")])
    manifest = table_manifest(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SILICONFLOW")}
    done = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--table-prefix",
            "--db",
            str(db),
            "--chroma",
            str(tmp_path / "chroma"),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "[knobs] 索引文本口径：身份前缀 + 表格导读行" in out
    assert "身份前缀生效" in out
    # 导读行把索引文本顶长了：那一段差额必须在预演里报出来，否则这份库与 `--doc-prefix` 那份
    # 从日志上分不出差别
    delta = int(out.split("（原文 +", 1)[1].split("）", 1)[0])
    assert delta > 0
    # 预演不建库：目标目录一个字节都不该被写出来
    assert not (tmp_path / "chroma").exists()


def test_table_prefix_wins_over_doc_prefix(tmp_path, capsys):
    """两个开关同时给：以更全的那份为准（表格口径已含身份前缀），不报错、也不静默退回弱口径。"""

    db = seed_db(tmp_path, [make_chunk("doc1", 0, "公司营收 100 亿元")])
    manifest = table_manifest(tmp_path)
    code = rebuild_vector_index.main(
        [
            "--dry-run",
            "--doc-prefix",
            "--table-prefix",
            "--db",
            str(db),
            "--chroma",
            str(tmp_path / "chroma"),
            "--manifest",
            str(manifest),
        ]
    )
    assert code == 0
    assert "[knobs] 索引文本口径：身份前缀 + 表格导读行" in capsys.readouterr().out
