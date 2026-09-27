import pytest

from akr.agents.errors import ToolResultError
from akr.agents.graph_walker import (
    AMBIGUITY_MARGIN,
    EVIDENCE_TOP_K,
    GRAPH_SYSTEM,
    QUERY_SYSTEM,
    run_graph_walk,
)
from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.orchestration.ledger import SubTask

GOOD_QUERY = (
    '{"mention": "甲公司", "relation_types": ["SUBSIDIARY_OF"],'
    ' "hypothesis": "甲公司通过子公司与丙公司存在供应关系"}'
)
GOOD_SUMMARY = (
    '{"findings": ["甲公司的子公司乙公司向丙公司供应原材料"],'
    ' "chunk_ids": ["doc2#chunk-01", "doc3#chunk-05"],'
    ' "confidence": 0.7, "open_questions": []}'
)
HALLUCINATED_SUMMARY = (
    '{"findings": ["甲公司的子公司乙公司向丙公司供应原材料"],'
    ' "chunk_ids": ["doc2#chunk-01", "doc9#chunk-99"],'
    ' "confidence": 0.7, "open_questions": []}'
)

CANDIDATES = [
    {
        "entity_id": "company:甲公司",
        "name": "甲公司",
        "type": "company",
        "aliases": [],
        "score": 0.93,
    },
    {
        "entity_id": "company:甲控股",
        "name": "甲控股",
        "type": "company",
        "aliases": [],
        "score": 0.52,
    },
]

PATHS = [
    {
        "nodes": [
            {"entity_id": "company:甲公司", "name": "甲公司", "type": "company", "aliases": []},
            {"entity_id": "company:乙公司", "name": "乙公司", "type": "company", "aliases": []},
        ],
        "relations": [
            {
                "relation": "SUBSIDIARY_OF",
                "source_id": "company:乙公司",
                "target_id": "company:甲公司",
            }
        ],
        "text": "乙公司 -[SUBSIDIARY_OF]-> 甲公司",
    },
    {
        "nodes": [
            {"entity_id": "company:乙公司", "name": "乙公司", "type": "company", "aliases": []},
            {"entity_id": "company:丙公司", "name": "丙公司", "type": "company", "aliases": []},
        ],
        "relations": [
            {
                "relation": "SUPPLIES",
                "source_id": "company:乙公司",
                "target_id": "company:丙公司",
            }
        ],
        "text": "乙公司 -[SUPPLIES]-> 丙公司",
    },
]

HIT_BY_PATH = {
    PATHS[0]["text"]: [
        {
            "chunk_id": "doc2#chunk-01",
            "doc_id": "doc2",
            "source": "甲公司2024年报.pdf",
            "text": "本公司持有乙公司 100% 股权，乙公司纳入合并报表范围",
            "score": 0.81,
            "retriever": "both",
        }
    ],
    PATHS[1]["text"]: [
        {
            "chunk_id": "doc3#chunk-05",
            "doc_id": "doc3",
            "source": "丙公司2024年报.pdf",
            "text": "报告期内公司主要原材料供应商包括乙公司",
            "score": 0.77,
            "retriever": "bm25",
        }
    ],
}


def make_subtask(kind="graph", goal="甲公司通过子公司与丙公司的供应关系是什么？"):
    return SubTask(task_id="t1", goal=goal, kind=kind)


class FakeGraphGateway:
    """假的 ToolGateway：只实现 GraphWalker 该用的三个工具。

    `calls` 记录调用顺序，用来断言「先链接实体 → 再走图 → 再按路径检索原文」
    这条流水线不可颠倒：反过来的话 citations 就只能来自图谱输出，而图谱没有
    `chunk_id`，引用支撑率将无从计算。
    """

    def __init__(
        self,
        candidates=None,
        paths=None,
        hits_by_query=None,
        resolve_error=None,
        walk_error=None,
        search_error=None,
    ):
        self.candidates = list(CANDIDATES if candidates is None else candidates)
        self.paths = list(PATHS if paths is None else paths)
        self.hits_by_query = dict(HIT_BY_PATH if hits_by_query is None else hits_by_query)
        self.resolve_error = resolve_error
        self.walk_error = walk_error
        self.search_error = search_error
        self.calls = []

    def resolve_entity(self, mention, context=""):
        # 签名必须与 `ToolGateway` 协议（src/akr/agents/tool_gateway.py）逐字一致：
        # 多一个生产网关不认的 `limit`，这个假网关就会把真 bug 挡在测试之外。
        self.calls.append(("resolve_entity", mention))
        if self.resolve_error is not None:
            return {"error": self.resolve_error}
        return {"candidates": [dict(c) for c in self.candidates]}

    def walk_graph(self, entity, hops=2, relation_types=None):
        self.calls.append(("walk_graph", entity, hops, relation_types))
        if self.walk_error is not None:
            return {"error": self.walk_error}
        return {"paths": [dict(p) for p in self.paths]}

    def search_documents(self, query, top_k=8, doc_filters=None):
        self.calls.append(("search_documents", query, top_k))
        if self.search_error is not None:
            return {"error": self.search_error}
        hits = self.hits_by_query.get(query, [])
        return {
            "hits": [dict(hit) for hit in hits],
            "fusion": "rrf",
            "retriever_stats": {"vector": len(hits), "bm25": len(hits)},
        }

    def fetch_chunk(self, *args, **kwargs):
        raise AssertionError("GraphWalker 不应调用 fetch_chunk")

    def list_documents(self, *args, **kwargs):
        raise AssertionError("GraphWalker 不应调用 list_documents")

    def close(self):
        return None


class ProtocolShapedGateway:
    """签名与 `ToolGateway` 协议逐字一致的网关：多传一个关键字参数就 TypeError。

    与 `FakeGraphGateway` 的区别只在签名。为什么值得单独造一个：假网关一旦比真网关
    「宽容」，被测代码就会在单测里一路绿灯、在生产里当场崩——2026-09-25 的
    `resolve_entity(limit=5)` 就是这么漏过去的。
    """

    def __init__(self):
        self.calls = []
        self._inner = FakeGraphGateway()

    def resolve_entity(self, mention, context=""):
        self.calls.append(("resolve_entity", {"mention": mention, "context": context}))
        return self._inner.resolve_entity(mention, context=context)

    def walk_graph(self, entity, hops=2, relation_types=None):
        self.calls.append(
            ("walk_graph", {"entity": entity, "hops": hops, "relation_types": relation_types})
        )
        return self._inner.walk_graph(entity, hops=hops, relation_types=relation_types)

    def search_documents(self, query, top_k=8, doc_filters=None):
        self.calls.append(
            ("search_documents", {"query": query, "top_k": top_k, "doc_filters": doc_filters})
        )
        return self._inner.search_documents(query, top_k=top_k, doc_filters=doc_filters)

    def fetch_chunk(self, chunk_id):
        raise AssertionError("GraphWalker 不应调用 fetch_chunk")

    def list_documents(self):
        raise AssertionError("GraphWalker 不应调用 list_documents")

    def close(self):
        return None


class FakeClient:
    """假的 LLMClient：按调用顺序返回预置文本（第 1 次查询解析，第 2 次归纳）。"""

    def __init__(self, query_text=GOOD_QUERY, summary_text=GOOD_SUMMARY, usage=None):
        self.texts = [query_text, summary_text]
        self.usage = (
            usage
            if usage is not None
            else Usage(hit_tokens=0, miss_tokens=1_000, output_tokens=200)
        )
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append({"system": system, "messages": messages, "mode": mode})
        index = min(len(self.calls) - 1, len(self.texts) - 1)
        return LLMResult(
            text=self.texts[index],
            reasoning="",
            usage=self.usage,
            mode=mode,
            tool_calls=[],
        )


def test_kind_must_be_graph():
    with pytest.raises(ValueError, match="kind='graph'"):
        run_graph_walk(
            make_subtask(kind="retrieve"), gateway=FakeGraphGateway(), client=FakeClient()
        )


def test_parses_query_with_model_then_walks_with_whitelisted_relations():
    gateway = FakeGraphGateway()
    client = FakeClient(query_text='{"mention": "甲公司", "hypothesis": "供应关系",'
                                    ' "relation_types": ["SUBSIDIARY_OF", "INVENTS"]}')
    report = run_graph_walk(make_subtask(), gateway=gateway, client=client)
    assert client.calls[0]["system"] == QUERY_SYSTEM
    assert client.calls[0]["mode"] == "non-thinking"
    walk = [call for call in gateway.calls if call[0] == "walk_graph"][0]
    assert walk[1] == "company:甲公司"  # mention 命中候选，而不是把整句 goal 丢给图
    assert walk[3] == ["SUBSIDIARY_OF"]
    assert any("INVENTS" in question for question in report.open_questions)


def test_pure_graph_mode_needs_no_model():
    gateway = FakeGraphGateway()
    report = run_graph_walk(make_subtask(), gateway=gateway, client=None)
    assert gateway.calls[0][1] == "甲公司通过子公司与丙公司的供应关系是什么？"
    assert gateway.calls[1][3] == []
    assert report.findings == []
    assert any("未启用摘要模型" in question for question in report.open_questions)


def test_unresolved_entity_reports_honestly_without_summary_call():
    gateway = FakeGraphGateway(candidates=[])
    client = FakeClient()
    report = run_graph_walk(make_subtask(), gateway=gateway, client=client)
    assert report.findings == []
    assert report.citations == []
    assert report.confidence == 0.0
    assert len(client.calls) == 1  # 只解析了查询，没有材料就不调归纳模型
    assert client.calls[0]["system"] == QUERY_SYSTEM
    assert not [call for call in gateway.calls if call[0] == "walk_graph"]
    assert any("未找到" in question for question in report.open_questions)


def test_empty_paths_reports_honestly_without_summary_call():
    gateway = FakeGraphGateway(paths=[])
    client = FakeClient()
    report = run_graph_walk(make_subtask(), gateway=gateway, client=client)
    assert report.findings == []
    assert len(client.calls) == 1
    assert not [call for call in gateway.calls if call[0] == "search_documents"]
    assert any("路径" in question for question in report.open_questions)


def test_paths_without_evidence_report_honestly_without_summary_call():
    gateway = FakeGraphGateway(hits_by_query={})  # 图谱有路径，但原文一条都没召回
    client = FakeClient()
    report = run_graph_walk(make_subtask(), gateway=gateway, client=client)
    searches = [call for call in gateway.calls if call[0] == "search_documents"]
    assert len(searches) == len(PATHS)
    assert report.findings == []
    assert report.citations == []
    assert len(client.calls) == 1
    assert any("原文证据" in question for question in report.open_questions)


def test_citations_come_from_search_hits_and_queries_are_path_texts():
    gateway = FakeGraphGateway()
    report = run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient())
    searched = [call[1] for call in gateway.calls if call[0] == "search_documents"]
    assert searched == [path["text"] for path in PATHS]
    assert [c.chunk_id for c in report.citations] == ["doc2#chunk-01", "doc3#chunk-05"]
    assert [c.doc_id for c in report.citations] == ["doc2", "doc3"]
    assert all(c.source for c in report.citations)  # 引用必须能被 Verifier 回溯到文件


def test_summary_call_carries_only_goal_and_evidence():
    client = FakeClient()
    run_graph_walk(make_subtask(), gateway=FakeGraphGateway(), client=client)
    summary_call = client.calls[1]
    assert summary_call["system"] == GRAPH_SYSTEM
    assert summary_call["mode"] == "non-thinking"
    assert len(summary_call["messages"]) == 1  # 没有历史、没有别的 worker 的上下文
    content = summary_call["messages"][0]["content"]
    assert "甲公司通过子公司与丙公司的供应关系是什么？" in content
    assert "doc2#chunk-01" in content and "doc3#chunk-05" in content
    assert "SUBSIDIARY_OF" in content  # 路径本身也要给模型，否则它只能靠猜


def test_ambiguous_candidates_recorded_in_open_questions():
    close = [
        dict(CANDIDATES[0], score=0.93),
        dict(CANDIDATES[1], score=0.93 - AMBIGUITY_MARGIN / 2),
    ]
    gateway = FakeGraphGateway(candidates=close)
    report = run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient())
    assert gateway.calls[1][1] == "company:甲公司"  # 仍按最高分继续，不中断
    assert any("歧义" in question for question in report.open_questions)


def test_clear_winner_keeps_open_questions_free_of_ambiguity_note():
    report = run_graph_walk(make_subtask(), gateway=FakeGraphGateway(), client=FakeClient())
    assert not any("歧义" in question for question in report.open_questions)


def test_hallucinated_chunk_ids_are_dropped_and_recorded():
    client = FakeClient(summary_text=HALLUCINATED_SUMMARY)
    report = run_graph_walk(make_subtask(), gateway=FakeGraphGateway(), client=client)
    assert [c.chunk_id for c in report.citations] == ["doc2#chunk-01"]
    assert any("幻觉引用" in question for question in report.open_questions)
    assert report.findings == ["甲公司的子公司乙公司向丙公司供应原材料"]


def test_cost_accumulates_both_model_calls_and_counts_steps():
    usage = Usage(hit_tokens=0, miss_tokens=1_000, output_tokens=200)
    client = FakeClient(usage=usage)
    report = run_graph_walk(make_subtask(), gateway=FakeGraphGateway(), client=client)
    assert report.cost.tokens == 2 * usage.total_tokens
    assert report.cost.cost_cny == pytest.approx(2 * usage.cost_cny(peak=True))
    assert report.cost.steps == 6  # 2 次模型调用 + 1 次实体链接 + 1 次图遍历 + 2 次检索
    assert report.cost.latency_ms >= 0


def test_evidence_top_k_and_hops_are_forwarded():
    gateway = FakeGraphGateway()
    run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient(), hops=3)
    assert gateway.calls[1][2] == 3
    top_ks = [call[2] for call in gateway.calls if call[0] == "search_documents"]
    assert top_ks == [EVIDENCE_TOP_K, EVIDENCE_TOP_K]


def test_graph_walk_runs_against_a_protocol_shaped_gateway():
    """守住「GraphWalker 只按 ToolGateway 协议调工具」这条线。

    2026-09-25 主实验实测：`_resolve_entity` 用 `gateway.resolve_entity(mention, limit=5)`
    调用，而协议与两个真实网关（`_GatewayBase` / `RecordingGateway`）的签名都只有
    `(mention, context="")`。这条线在单测里被上面那个「多一个 limit 参数」的假网关
    盖住了，一进生产就在第 0 步抛
    `TypeError: RecordingGateway.resolve_entity() got an unexpected keyword argument 'limit'`。
    本用例的网关签名与协议逐字一致（多传一个关键字参数就会 TypeError），因此它认的
    调用恰好就是生产网关认的调用。
    """
    gateway = ProtocolShapedGateway()
    report = run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient())

    assert report.findings == ["甲公司的子公司乙公司向丙公司供应原材料"]
    assert [call[0] for call in gateway.calls] == [
        "resolve_entity",
        "walk_graph",
        "search_documents",
        "search_documents",
    ]
    assert gateway.calls[0][1] == {"mention": "甲公司", "context": ""}


def test_resolve_entity_error_payload_raises_tool_result_error():
    gateway = FakeGraphGateway(resolve_error="neo4j 连接失败")
    with pytest.raises(ToolResultError, match="resolve_entity"):
        run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient())


def test_walk_graph_error_payload_raises_tool_result_error():
    gateway = FakeGraphGateway(walk_error="查询超时")
    with pytest.raises(ToolResultError, match="walk_graph"):
        run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient())


def test_path_missing_text_raises_tool_result_error():
    gateway = FakeGraphGateway(paths=[{"nodes": [], "relations": []}])
    with pytest.raises(ToolResultError, match="text"):
        run_graph_walk(make_subtask(), gateway=gateway, client=FakeClient())
