"""编排层与工具层之间的唯一通道。

两种实现，对编排层完全等价：

- ``InProcessGateway``：同进程直调 ``dispatch``，单测快、无子进程，用于内联 demo。
- ``McpStdioGateway``：把 MCP server 作为子进程启动，通过 stdio 通信（真正的进程隔离）。

编排层（Task 27–31）只 import 本模块：工具集变化时编排层零改动，
替换工具实现也不需要改 agent 代码。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Protocol

from akr.mcp_server.tools import TOOL_FUNCTIONS, ToolContext, dispatch

__all__ = ["ToolCallError", "ToolGateway", "InProcessGateway", "McpStdioGateway"]

DEFAULT_TIMEOUT_SECONDS = 60.0


class ToolCallError(RuntimeError):
    """工具调用失败：含 MCP 侧返回的 ``{"error": ...}`` 载荷与非法响应。"""


class ToolGateway(Protocol):
    """编排层看到的工具表面。五个方法对应 spec §6.1 的五个工具。"""

    tools: Sequence[str]

    def search_documents(
        self,
        query: str,
        top_k: int = 8,
        doc_filters: dict[str, Any] | None = None,
        two_stage_docs: int = 0,
        expand_neighbors: int = 0,
    ) -> dict[str, Any]: ...

    def walk_graph(
        self,
        entity: str,
        hops: int = 2,
        relation_types: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...

    def resolve_entity(self, mention: str, context: str = "") -> dict[str, Any]: ...

    def fetch_chunk(self, chunk_id: str) -> dict[str, Any]: ...

    def list_documents(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _is_error(result: Any) -> bool:
    """mcp 1.x 的字段是 ``isError``，2.x 改名为 ``is_error``：两种都要认。

    只认旧名会让真实的 MCP 层错误被静默当成成功（拿到空结果），因此这里对两个名字都查。
    """

    if getattr(result, "isError", False):
        return True
    return bool(getattr(result, "is_error", False))


class _GatewayBase:
    """把 5 个协议方法翻译成 ``(工具名, arguments)``，具体执行交给 ``_invoke``。"""

    def _invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def search_documents(
        self,
        query: str,
        top_k: int = 8,
        doc_filters: dict[str, Any] | None = None,
        two_stage_docs: int = 0,
        expand_neighbors: int = 0,
    ) -> dict[str, Any]:
        return self._invoke(
            "search_documents",
            {
                "query": query,
                "top_k": top_k,
                "doc_filters": doc_filters,
                "two_stage_docs": two_stage_docs,
                # 邻块扩窗的宽度（0 = 关闭，工具层默认行为不变）。Task 2/3 的
                # `pool_depth` / `rerank` **这两个旋钮刻意不进这条链**：它们留在工具层，
                # 谁能配、配多少是「待拍板项」（Task 8 的清单），本任务只打通扩窗这一条。
                # 注意别把「不传旋钮」读成「跑的是纯 RRF」：2026-09-26 起工具层**默认档**
                # 就是 `rerank="cross"` + 锁 4，所以不传 `rerank` 的调用方（评测的
                # RecordingGateway / 线上 orchestrator）走的都是「带一次重排」那条路。
                "expand_neighbors": expand_neighbors,
            },
        )

    def walk_graph(
        self,
        entity: str,
        hops: int = 2,
        relation_types: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return self._invoke(
            "walk_graph",
            {
                "entity": entity,
                "hops": hops,
                "relation_types": list(relation_types) if relation_types is not None else None,
            },
        )

    def resolve_entity(self, mention: str, context: str = "") -> dict[str, Any]:
        return self._invoke("resolve_entity", {"mention": mention, "context": context})

    def fetch_chunk(self, chunk_id: str) -> dict[str, Any]:
        return self._invoke("fetch_chunk", {"chunk_id": chunk_id})

    def list_documents(self) -> dict[str, Any]:
        return self._invoke("list_documents", {})


class InProcessGateway(_GatewayBase):
    """同进程直调 ``dispatch``。工具集从 ``TOOL_FUNCTIONS`` 取，仍是动态来源。"""

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self.tools: tuple[str, ...] = tuple(sorted(TOOL_FUNCTIONS))

    def _invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return dispatch(self._ctx, name, arguments)

    def close(self) -> None:
        """不关闭共享 ctx：连接由构造它的调用方持有并负责释放。"""


@asynccontextmanager
async def stdio_session(
    command: str,
    args: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[Any]:
    """启动 MCP server 子进程并完成握手（真实路径，集成测试与生产用）。

    ``env`` 必须显式传：MCP stdio 只继承少数"安全"环境变量（PATH 等），
    ``NEO4J_URI`` / ``AKR_*`` 不在其中，否则子进程拿不到配置，启动即失败。
    """

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=command,
        args=list(args),
        env=dict(env) if env else None,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


class McpStdioGateway(_GatewayBase):
    """MCP stdio 子进程网关：同步门面 + 后台事件循环线程。

    工具集动态发现：启动后调用一次 ``list_tools()``，把工具名缓存进 ``.tools``；
    协议方法本身只走固定五个工具，因此工具集变化（新增/删除）不会让编排层失效。

    线程模型（不是"每个调用一个 task"，这点必须这么写）：MCP 的 stdio 传输用 anyio
    cancel scope / TaskGroup，进入与退出必须发生在**同一个 task** 里。
    ``run_coroutine_threadsafe`` 每次调用都新建 task，那样会在退出时报
    "Attempted to exit cancel scope in a different task than it was entered in"。
    因此后台线程只跑一个长驻协程：它打开会话、循环处理请求队列、收到结束哨兵后
    在同一个 task 内正常关闭会话。同步方法只负责投递 ``(工具名, 参数, future)`` 并等结果。
    """

    def __init__(
        self,
        *,
        command: str = "python",
        args: Sequence[str] = ("-m", "akr.mcp_server.server"),
        env: Mapping[str, str] | None = None,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._closed = False
        self._failure: BaseException | None = None
        self._requests: asyncio.Queue[Any] = asyncio.Queue()
        self._opened: concurrent.futures.Future[tuple[str, ...]] = concurrent.futures.Future()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="akr-mcp-loop", daemon=True)
        self._session_factory = session_factory or partial(
            stdio_session, command, tuple(args), env
        )
        self.tools: tuple[str, ...] = ()
        self._thread.start()
        if not self._ready.wait(timeout=DEFAULT_TIMEOUT_SECONDS):
            raise ToolCallError("MCP 事件循环线程启动超时")
        self.tools = self._await(self._opened, what="MCP 会话初始化")

    # --- 事件循环线程管理 ---

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_until_complete(self._serve_forever())
        finally:
            self._loop.close()

    def _await(self, future: Any, *, what: str) -> Any:
        """等一个 ``concurrent.futures.Future``；超时转成 ToolCallError，异常原样抛出。"""

        try:
            return future.result(timeout=DEFAULT_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise ToolCallError(f"{what}超时（>{DEFAULT_TIMEOUT_SECONDS:.0f}s）") from None

    # --- 会话任务（后台线程里唯一的长驻协程） ---

    async def _serve_forever(self) -> None:
        session_cm = None
        try:
            session_cm = self._session_factory()
            session = await session_cm.__aenter__()
            listed = await session.list_tools()
            self._opened.set_result(tuple(tool.name for tool in listed.tools))
            while True:
                request = await self._requests.get()
                if request is None:  # 结束哨兵：正常退出，__aexit__ 仍在同一个 task 内
                    return
                name, arguments, future = request
                try:
                    future.set_result(await call_session(session, name, arguments))
                except BaseException as exc:  # noqa: BLE001 - 单次调用失败只影响这一次请求
                    future.set_exception(exc)
        except BaseException as exc:  # noqa: BLE001 - 必须传给等待者，不能让它死等超时
            self._failure = exc
            self._drain_pending(exc)
            if not self._opened.done():
                self._opened.set_exception(exc)
        finally:
            if session_cm is not None:
                await session_cm.__aexit__(None, None, None)

    def _drain_pending(self, exc: BaseException) -> None:
        while not self._requests.empty():
            request = self._requests.get_nowait()
            if request is not None and not request[2].done():
                request[2].set_exception(exc)

    # --- 同步门面 ---

    def _invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise ToolCallError("gateway 已关闭，无法继续调用工具")
        if self._failure is not None:
            raise ToolCallError(f"MCP 会话已中断：{self._failure}") from self._failure
        future: concurrent.futures.Future[dict[str, Any]] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._requests.put_nowait, (name, arguments, future))
        return self._await(future, what=f"工具 {name} ")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._requests.put_nowait, None)
            except RuntimeError:  # 循环已停：已经没有会话需要收尾
                pass
        self._thread.join(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._ready.clear()


async def call_session(session: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """一次 MCP 工具调用：解析文本载荷，把 MCP 层或工具层的错误转成 ``ToolCallError``。"""

    result = await session.call_tool(name, arguments)
    text = "".join(getattr(item, "text", "") for item in result.content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallError(f"工具 {name} 返回非 JSON 文本：{text[:200]}") from exc
    if _is_error(result) or (isinstance(payload, dict) and "error" in payload):
        message = payload.get("error") if isinstance(payload, dict) else text
        raise ToolCallError(f"工具 {name} 失败：{message}")
    return payload
