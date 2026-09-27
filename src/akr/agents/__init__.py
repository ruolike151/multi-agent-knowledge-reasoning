"""编排层：状态机、Supervisor、Worker、Runner。

本包的模块只通过 ``akr.agents.tool_gateway`` 访问工具层，
不直接依赖 MCP 工具的实现模块（进程隔离与可替换性）。
"""
