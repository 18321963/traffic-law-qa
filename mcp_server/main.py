from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from api_contracts.client import DEFAULT_TIMEOUT, RagClient
from mcp_server.tools import TOOLS, call_tool
from rag_contracts.domain.errors import QaError

__all__ = ["main", "build_server", "use_utf8_stdio"]

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
SERVER_NAME = "traffic-law-rag"

USAGE = """\
把 RAG 服务挂成 MCP（stdio）给宿主用。本进程只做 HTTP 转发：不载模型、不连 Milvus。

    python -m mcp_server                          # 连默认地址
    python -m mcp_server --base-url http://host:8000
    python -m mcp_server --list-tools             # 只打工具表（离线，不连服务）

地址也可以走环境变量 RAG_BASE_URL；命令行参数优先。服务得先起着，
工具返回的都是端点原样透传的 dict，字段口径见 api_contracts/openapi.json。
"""


def build_server(client: RagClient) -> Any:
    from mcp import types
    from mcp.server.lowlevel import Server

    async def list_tools(_ctx: Any, _params: Any) -> Any:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=spec.name, description=spec.description, input_schema=spec.schema)
                for spec in TOOLS
            ]
        )

    async def handle_tool(_ctx: Any, params: Any) -> Any:
        try:
            payload = call_tool(client, params.name, params.arguments)
        except QaError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))], is_error=True
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
        )

    return Server(
        SERVER_NAME,
        instructions=(
            "交通法规知识库检索：查法条用 rag_search / rag_get_article，法规清单用 rag_laws，"
            "用户上传的材料用 rag_search_materials，要整篇带引用的答案用 rag_ask。"
        ),
        on_list_tools=list_tools,
        on_call_tool=handle_tool,
    )


async def _serve(server: Any) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def use_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="python -m mcp_server",
        description=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAG_BASE_URL") or DEFAULT_BASE_URL,
        help=f"RAG 服务地址，默认 {DEFAULT_BASE_URL}（或环境变量 RAG_BASE_URL）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("RAG_TIMEOUT") or DEFAULT_TIMEOUT),
        help=f"单次请求超时秒数，默认 {DEFAULT_TIMEOUT:g}",
    )
    parser.add_argument("--list-tools", action="store_true", help="只打印工具表后退出（不连服务）")
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps([{"name": spec.name, "schema": spec.schema} for spec in TOOLS], ensure_ascii=False, indent=2))
        return 0

    client = RagClient(args.base_url, timeout=args.timeout)
    try:
        asyncio.run(_serve(build_server(client)))
    except KeyboardInterrupt:
        return 130
    finally:
        client.close()
    return 0
