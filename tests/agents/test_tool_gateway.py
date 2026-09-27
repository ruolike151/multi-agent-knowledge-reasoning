import json
from types import SimpleNamespace

import pytest

from akr.agents.tool_gateway import InProcessGateway, McpStdioGateway, ToolCallError
from akr.mcp_server.tools import ToolContext
from akr.storage import sqlite_store


class FakeSession:
    """假的 MCP ClientSession：只需满足 gateway 用到的那几个方法。"""

    def __init__(self, *, tools=("search_documents", "list_documents"), payload=None, is_error=False):
        self._tools = list(tools)
        self._payload = {"documents": []} if payload is None else payload
        self._is_error = is_error
        self.calls = []
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def list_tools(self):
        return SimpleNamespace(tools=[SimpleNamespace(name=n) for n in self._tools])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        content = [SimpleNamespace(type="text", text=json.dumps(self._payload))]
        return SimpleNamespace(content=content, isError=self._is_error)


class FakeRealStyleSession(FakeSession):
    """mcp 2.x 的 CallToolResult 用 ``is_error``（不是 ``isError``）。"""

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self._payload))],
            is_error=self._is_error,
        )


def make_factory(session):
    return lambda: session


def make_in_process_ctx(tmp_path):
    conn = sqlite_store.connect(tmp_path / "akr.sqlite")
    sqlite_store.init_schema(conn)
    return ToolContext(conn=conn, vectors=None, bm25=None, graph=None, artifacts=None)


def test_in_process_gateway_delegates_to_dispatch(tmp_path):
    gateway = InProcessGateway(make_in_process_ctx(tmp_path))
    assert gateway.list_documents() == {"documents": []}
    assert set(gateway.tools) == {
        "search_documents",
        "walk_graph",
        "resolve_entity",
        "fetch_chunk",
        "list_documents",
    }


def test_in_process_gateway_close_does_not_close_shared_context(tmp_path):
    InProcessGateway(make_in_process_ctx(tmp_path)).close()  # 不抛异常即可（ctx 由调用方持有）


def test_mcp_gateway_discovers_tools_instead_of_hardcoding():
    session = FakeSession(tools=["search_documents", "walk_graph", "custom_tool"])
    gateway = McpStdioGateway(session_factory=make_factory(session))
    try:
        assert set(gateway.tools) == {"search_documents", "walk_graph", "custom_tool"}
    finally:
        gateway.close()


def test_mcp_gateway_parses_json_payload_and_forwards_arguments():
    session = FakeSession(payload={"hits": []})
    gateway = McpStdioGateway(session_factory=make_factory(session))
    try:
        result = gateway.search_documents("甲公司", top_k=3)
        assert result == {"hits": []}
        assert session.calls == [
            (
                "search_documents",
                {
                    "query": "甲公司",
                    "top_k": 3,
                    "doc_filters": None,
                    # 两段式检索的开关走同一个入参通道（默认 0 = 关闭，工具层默认行为不变）
                    "two_stage_docs": 0,
                    # 邻块扩窗的宽度同理（默认 0 = 关闭）
                    "expand_neighbors": 0,
                },
            )
        ]
    finally:
        gateway.close()


def test_mcp_gateway_raises_on_error_result():
    session = FakeSession(payload={"error": "boom"}, is_error=True)
    gateway = McpStdioGateway(session_factory=make_factory(session))
    try:
        with pytest.raises(ToolCallError, match="boom"):
            gateway.list_documents()
    finally:
        gateway.close()


def test_mcp_gateway_raises_on_error_key_even_without_is_error_flag():
    session = FakeSession(payload={"error": "工具参数不合法"})
    gateway = McpStdioGateway(session_factory=make_factory(session))
    try:
        with pytest.raises(ToolCallError, match="工具参数不合法"):
            gateway.list_documents()
    finally:
        gateway.close()


def test_mcp_gateway_recognises_mcp2_is_error_field():
    """mcp 2.x 把 isError 改名为 is_error，网关必须两种都认。"""

    session = FakeRealStyleSession(payload={"documents": []}, is_error=True)
    gateway = McpStdioGateway(session_factory=make_factory(session))
    try:
        with pytest.raises(ToolCallError, match="失败"):
            gateway.list_documents()
    finally:
        gateway.close()


def test_mcp_gateway_close_exits_session():
    session = FakeSession()
    gateway = McpStdioGateway(session_factory=make_factory(session))
    gateway.list_documents()
    gateway.close()
    assert session.exited is True


def test_mcp_gateway_rejects_calls_after_close():
    session = FakeSession()
    gateway = McpStdioGateway(session_factory=make_factory(session))
    gateway.close()
    with pytest.raises(ToolCallError, match="已关闭"):
        gateway.list_documents()
