"""真实子进程路径：以 stdio 拉起 ``python -m akr.mcp_server.server`` 并完成 MCP 握手。

归到集成测试是因为 server 启动时会连 Neo4j（Task 3 的 Docker 服务），
单测只覆盖 ``session_factory`` 注入的假会话。
"""

import os
import sys

import pytest

from akr.agents.tool_gateway import McpStdioGateway, ToolCallError

pytestmark = pytest.mark.integration


def server_env(tmp_path):
    """子进程环境：MCP stdio 不继承 NEO4J_* / AKR_*，必须显式传。

    两个路径显式置空 = 关闭向量与 BM25 依赖，避免启动时加载嵌入模型（本测试只验证协议链路）。
    """

    return {
        "NEO4J_URI": os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        "NEO4J_USER": os.environ.get("NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", "akr-local-pass"),
        "AKR_DB_PATH": str(tmp_path / "akr.sqlite"),
        "AKR_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
        "AKR_CHROMA_PATH": "",
        "AKR_BM25_PATH": "",
    }


def test_stdio_gateway_discovers_tools_and_calls_them(tmp_path):
    gateway = McpStdioGateway(command=sys.executable, env=server_env(tmp_path))
    try:
        assert set(gateway.tools) == {
            "search_documents",
            "walk_graph",
            "resolve_entity",
            "fetch_chunk",
            "list_documents",
        }
        # 空库：真实往返（子进程 → SQLite）应当返回空文档列表
        assert gateway.list_documents() == {"documents": []}
    finally:
        gateway.close()


def test_stdio_gateway_surfaces_tool_errors_from_subprocess(tmp_path):
    gateway = McpStdioGateway(command=sys.executable, env=server_env(tmp_path))
    try:
        with pytest.raises(ToolCallError, match="d1#cmissing"):
            gateway.fetch_chunk("d1#cmissing")
    finally:
        gateway.close()
