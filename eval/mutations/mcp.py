from __future__ import annotations

__all__ = ["MUTATIONS"]

TESTS = ["tests/test_mcp_tools.py"]

MUTATIONS = [
    (
        "M1 工具参数与端点脱钩（debug 改名 trace）",
        "mcp_server/tools.py",
        '                "debug": {\n',
        '                "trace": {\n',
        TESTS,
    ),
    (
        "M2 工具面改了个名字",
        "mcp_server/tools.py",
        '        name="rag_search",\n',
        '        name="search_law",\n',
        TESTS,
    ),
    (
        "M3 处理器不再原样透传（把载荷复制一遍）",
        "mcp_server/tools.py",
        'def _laws(client: RagClient, args: dict[str, Any]) -> dict:\n'
        '    return client.request("GET", "/laws")\n',
        'def _laws(client: RagClient, args: dict[str, Any]) -> dict:\n'
        '    return dict(client.request("GET", "/laws"))\n',
        TESTS,
    ),
    (
        "M4 上界两边各写一份（top_k 上限 20 → 21）",
        "mcp_server/tools.py",
        '                    "maximum": 20,\n                    "description": "返回条数，默认 6。",\n',
        '                    "maximum": 21,\n                    "description": "返回条数，默认 6。",\n',
        TESTS,
    ),
    (
        "M5 工具表顶层 import MCP SDK",
        "mcp_server/tools.py",
        "from api_contracts.client import RagClient\n",
        "from api_contracts.client import RagClient\nfrom mcp import types\n",
        TESTS,
    ),
    (
        "M6 适配层拉服务端实现（连 Milvus）",
        "mcp_server/tools.py",
        "from rag_contracts.domain.errors import QaError\n",
        "from rag_service.adapters.milvus import MilvusStore\nfrom rag_contracts.domain.errors import QaError\n",
        TESTS,
    ),
    (
        "M7 import 期拉起重依赖",
        "mcp_server/tools.py",
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass\nimport torch\n",
        TESTS,
    ),
]
