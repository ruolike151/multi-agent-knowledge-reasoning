import pytest

from akr.agents import retriever
from akr.agents.errors import ToolResultError
from akr.agents.retriever import SNIPPET_CHARS, SUMMARY_SYSTEM, run_retrieve
from akr.model.client import LLMResult
from akr.model.cost import Usage
from akr.model.structured import StructuredOutputError
from akr.orchestration.ledger import SubTask

GOOD_SUMMARY = (
    '{"findings": ["甲公司 2024 年营业收入为 12.3 亿元"],'
    ' "chunk_ids": ["doc1#chunk-03"], "confidence": 0.8, "open_questions": []}'
)

HITS = [
    {
        "chunk_id": "doc1#chunk-03",
        "doc_id": "doc1",
        "source": "甲公司2024年报.pdf",
        "text": "2024 年公司实现营业收入 12.3 亿元，同比增长 8%",
        "score": 0.91,
        "retriever": "both",
    },
    {
        "chunk_id": "doc1#chunk-07",
        "doc_id": "doc1",
        "source": "甲公司2024年报.pdf",
        "text": "营业收入按产品拆分：A 产品 7.1 亿元，B 产品 5.2 亿元",
        "score": 0.72,
        "retriever": "vector",
    },
]


def make_subtask(kind="retrieve", goal="甲公司的营业收入是多少？"):
    return SubTask(task_id="t1", goal=goal, kind=kind)


class FakeGateway:
    """只实现 Retriever 该用的那个工具，其余方法被调用即失败。

    「只调用 search_documents」是设计约束：混合检索（向量 + BM25 + RRF）已经在
    工具层融合完成，worker 不该自己分别调两路再合并。
    """

    def __init__(self, hits=None, error=None):
        self.hits = list(HITS if hits is None else hits)
        self.error = error
        self.query = None
        self.top_k = None
        self.two_stage_docs = None
        self.expand_neighbors = None

    def search_documents(
        self, query, top_k=8, doc_filters=None, two_stage_docs=0, expand_neighbors=0
    ):
        self.query = query
        self.top_k = top_k
        self.two_stage_docs = two_stage_docs
        self.expand_neighbors = expand_neighbors
        if self.error is not None:
            return {"error": self.error}
        return {
            "hits": [dict(hit) for hit in self.hits],
            "fusion": "rrf",
            "retriever_stats": {"vector": len(self.hits), "bm25": len(self.hits)},
        }

    def walk_graph(self, *args, **kwargs):
        raise AssertionError("Retriever 不应调用 walk_graph")

    def resolve_entity(self, *args, **kwargs):
        raise AssertionError("Retriever 不应调用 resolve_entity")

    def fetch_chunk(self, *args, **kwargs):
        raise AssertionError("Retriever 不应调用 fetch_chunk")

    def list_documents(self, *args, **kwargs):
        raise AssertionError("Retriever 不应调用 list_documents")

    def close(self):
        return None


class FakeClient:
    """假的 LLMClient：记录调用参数，返回预置文本，绝不发真实请求。"""

    def __init__(self, text=GOOD_SUMMARY, usage=None):
        self.text = text
        self.usage = usage if usage is not None else Usage(
            hit_tokens=0, miss_tokens=1_000, output_tokens=200
        )
        self.calls = []

    def chat(self, *, system, messages, mode, tools=None, json_object=False):
        self.calls.append(
            {"system": system, "messages": messages, "mode": mode, "json_object": json_object}
        )
        return LLMResult(text=self.text, reasoning="", usage=self.usage, mode=mode, tool_calls=[])


def test_kind_must_be_retrieve():
    with pytest.raises(ValueError, match="kind='retrieve'"):
        run_retrieve(make_subtask(kind="graph"), gateway=FakeGateway(), client=FakeClient())


def test_calls_search_documents_with_goal_and_top_k():
    gateway = FakeGateway()
    run_retrieve(make_subtask(), gateway=gateway, client=FakeClient(), top_k=3)
    assert gateway.query == "甲公司的营业收入是多少？"
    assert gateway.top_k == 3


def test_query_defaults_to_subtask_goal_and_two_stage_stays_off():
    """默认（不传 `query` / `two_stage_docs`）时调用形状与旧版逐字相同。"""
    gateway = FakeGateway()
    run_retrieve(make_subtask(goal="查甲公司营业收入"), gateway=gateway, client=None)
    assert gateway.query == "查甲公司营业收入"
    assert gateway.two_stage_docs == 0


def test_query_and_two_stage_are_forwarded_to_the_tool():
    """编排层给的原问句与两段式开关必须原样传到工具层——这是修法生效的唯一入口。"""
    gateway = FakeGateway()
    run_retrieve(
        make_subtask(goal="查甲公司营业收入"),
        gateway=gateway,
        client=None,
        query="甲公司 2024 年的营业收入是多少？",
        two_stage_docs=3,
    )
    assert gateway.query == "甲公司 2024 年的营业收入是多少？"
    assert gateway.two_stage_docs == 3


def test_worker_context_contains_only_goal_and_hits():
    client = FakeClient()
    run_retrieve(make_subtask(), gateway=FakeGateway(), client=client)
    call = client.calls[0]
    assert call["system"] == SUMMARY_SYSTEM
    assert len(call["messages"]) == 1  # 没有历史、没有其他 worker 的上下文
    assert call["messages"][0]["role"] == "user"
    content = call["messages"][0]["content"]
    assert "甲公司的营业收入是多少？" in content
    assert "doc1#chunk-03" in content


def test_system_prompt_is_static_across_different_subtasks():
    client = FakeClient()
    run_retrieve(make_subtask(), gateway=FakeGateway(), client=client)
    run_retrieve(
        make_subtask(goal="乙公司的研发投入是多少？"), gateway=FakeGateway(), client=client
    )
    assert {call["system"] for call in client.calls} == {SUMMARY_SYSTEM}
    # 动态内容一律不进 system prompt，否则前缀缓存不可能命中（spec §7.5）
    assert "甲公司" not in SUMMARY_SYSTEM
    assert "doc1#chunk-03" not in SUMMARY_SYSTEM


def test_structured_output_is_requested_as_json_object():
    client = FakeClient()
    run_retrieve(make_subtask(), gateway=FakeGateway(), client=client)
    assert client.calls[0]["json_object"] is True


def test_mode_is_forwarded_to_the_model():
    client = FakeClient()
    run_retrieve(make_subtask(), gateway=FakeGateway(), client=client, mode="thinking")
    assert client.calls[0]["mode"] == "thinking"


def test_fabricated_chunk_ids_are_dropped_and_reported():
    text = (
        '{"findings": ["甲公司 2024 年营业收入为 12.3 亿元", "甲公司 2023 年营业收入为 10 亿元"],'
        ' "chunk_ids": ["doc1#chunk-03", "doc9#chunk-99"],'
        ' "confidence": 0.7, "open_questions": []}'
    )
    report = run_retrieve(make_subtask(), gateway=FakeGateway(), client=FakeClient(text))
    assert [citation.chunk_id for citation in report.citations] == ["doc1#chunk-03"]
    assert any("不在本次检索结果中" in question for question in report.open_questions)
    # 断言本身保留（它可能仍然正确），被剔除的是无法溯源的引用
    assert len(report.findings) == 2


def test_citations_carry_doc_id_and_source_from_hits():
    report = run_retrieve(make_subtask(), gateway=FakeGateway(), client=FakeClient())
    citation = report.citations[0]
    assert citation.doc_id == "doc1"
    assert citation.source == "甲公司2024年报.pdf"
    assert report.confidence == pytest.approx(0.8)


def test_tokens_and_cost_are_measured_from_usage():
    usage = Usage(hit_tokens=100, miss_tokens=900, output_tokens=1_000)
    report = run_retrieve(make_subtask(), gateway=FakeGateway(), client=FakeClient(usage=usage))
    assert report.cost.tokens == usage.total_tokens == 2_000
    assert report.cost.cost_cny == pytest.approx(usage.cost_cny(peak=True))
    assert report.cost.cost_cny > 0
    assert report.cost.steps == 2  # 一次工具调用 + 一次模型调用


def test_latency_ms_is_measured_not_guessed(monkeypatch):
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr(retriever.time, "perf_counter", lambda: next(ticks))
    report = run_retrieve(make_subtask(), gateway=FakeGateway(), client=FakeClient())
    assert report.cost.latency_ms == 250


def test_pure_retrieval_mode_without_client_uses_snippets():
    report = run_retrieve(make_subtask(), gateway=FakeGateway(), client=None)
    assert len(report.findings) == 2
    assert all(len(finding) <= SNIPPET_CHARS + 1 for finding in report.findings)
    assert [citation.chunk_id for citation in report.citations] == [
        "doc1#chunk-03",
        "doc1#chunk-07",
    ]
    assert report.confidence == 0.0
    assert any("未启用摘要模型" in question for question in report.open_questions)


def test_no_hits_returns_honest_open_question_and_skips_model():
    client = FakeClient()
    report = run_retrieve(make_subtask(), gateway=FakeGateway(hits=[]), client=client)
    assert report.findings == []
    assert report.citations == []
    assert client.calls == []  # 没有材料可归纳，这一次调用不花
    assert any("无命中" in question for question in report.open_questions)


def test_unparseable_model_output_raises_instead_of_empty_report():
    with pytest.raises(StructuredOutputError):
        run_retrieve(make_subtask(), gateway=FakeGateway(), client=FakeClient(text="今天天气不错"))


def test_tool_error_payload_raises_tool_result_error():
    with pytest.raises(ToolResultError) as excinfo:
        run_retrieve(
            make_subtask(), gateway=FakeGateway(error="chroma 连接失败"), client=FakeClient()
        )
    assert excinfo.value.tool == "search_documents"
    assert "chroma 连接失败" in str(excinfo.value)


def test_hit_missing_required_field_raises_tool_result_error():
    broken = [{"chunk_id": "doc1#chunk-03"}]  # 缺 doc_id / source
    with pytest.raises(ToolResultError, match="doc_id"):
        run_retrieve(make_subtask(), gateway=FakeGateway(hits=broken), client=FakeClient())
