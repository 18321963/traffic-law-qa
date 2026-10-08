from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server.tools import TOOLS, call_tool
from rag_contracts.domain.errors import QaError

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "api_contracts" / "openapi.json"

HEAVY = ("torch", "pymilvus", "transformers", "langgraph", "openai")

OPERATIONS = {
    "rag_search": ("/qa", "post"),
    "rag_ask": ("/qa", "post"),
    "rag_get_article": ("/articles/lookup", "post"),
    "rag_laws": ("/laws", "get"),
    "rag_search_materials": ("/materials/search", "post"),
}

NAMES = [
    "rag_search",
    "rag_ask",
    "rag_get_article",
    "rag_laws",
    "rag_search_materials",
]


class _Recorder:
    def __init__(self) -> None:
        self.payload = {"marker": "端点原样", "nested": {"a": [1, 2]}}
        self.calls: list[tuple] = []

    def qa(self, question: str, **kwargs) -> dict:
        self.calls.append(("qa", question, kwargs))
        return self.payload

    def article(self, **kwargs) -> dict:
        self.calls.append(("article", kwargs))
        return self.payload

    def request(self, method: str, path: str, **kwargs) -> dict:
        self.calls.append(("request", method, path))
        return self.payload

    def materials(self, query: str, doc_ids=(), **kwargs) -> dict:
        self.calls.append(("materials", query, tuple(doc_ids), kwargs))
        return self.payload


CASES = {
    "rag_search": (
        {"question": "醉驾怎么罚", "top_k": 3},
        ("qa", "醉驾怎么罚", {"mode": "search", "top_k": 3, "law_filter": (), "debug": False}),
    ),
    "rag_ask": (
        {"question": "醉驾怎么罚", "top_k": 3},
        ("qa", "醉驾怎么罚", {"mode": "ask", "top_k": 3, "law_filter": ()}),
    ),
    "rag_get_article": (
        {"article_no": "90", "law_name": "中华人民共和国道路交通安全法"},
        ("article", {"article_no": "90", "law_name": "中华人民共和国道路交通安全法", "text": None}),
    ),
    "rag_laws": ({}, ("request", "GET", "/laws")),
    "rag_search_materials": (
        {"query": "头盔", "doc_ids": ["d1"], "top_k": 3},
        ("materials", "头盔", ("d1",), {"top_k": 3}),
    ),
}


def _run_child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _request_schema(spec: dict, path: str, method: str) -> dict:
    body = spec["paths"][path][method].get("requestBody")
    if not body:
        return {}
    schema = body["content"]["application/json"]["schema"]
    ref = schema.get("$ref")
    if ref:
        return spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    return schema


def _bounds(prop: dict) -> dict:
    options = prop.get("anyOf") or prop.get("allOf") or [prop]
    return {
        key: option[key]
        for option in options
        for key in ("minimum", "maximum")
        if key in option
    }


def test_the_tool_face_is_exactly_the_five_of_the_service() -> None:
    names = [spec.name for spec in TOOLS]
    assert names == NAMES, (
        "工具面变了。这五个对应服务端点：rag_search/rag_ask → POST /qa（mode=search/ask）、"
        "rag_get_article → POST /articles/lookup、rag_laws → GET /laws、"
        "rag_search_materials → POST /materials/search。\n  " + repr(names)
    )
    assert "web_search" not in names, (
        "网搜（博查）是 agent 侧的外部检索，不属 RAG：它随下一轮搬去 agent_service，"
        "这层只做「问知识库」这一件事。"
    )
    assert OPERATIONS.keys() == set(names), "对齐表与工具表对不上了，下面的检查会静默少测几个工具"


@pytest.mark.parametrize("spec", TOOLS, ids=[spec.name for spec in TOOLS])
def test_every_tool_parameter_exists_on_the_matching_endpoint(spec) -> None:
    path, method = OPERATIONS[spec.name]
    request = _request_schema(_spec(), path, method)
    allowed = set(request.get("properties") or {})
    declared = set(spec.schema["properties"])
    extra = sorted(declared - allowed)
    assert not extra, (
        f"{spec.name} 声明了 {method.upper()} {path} 上没有的参数：{extra} —— "
        "这一层只做透传，端点上没有的键发过去就是 422 或者被静默丢掉"
    )

    required = set(spec.schema.get("required") or ())
    assert required <= declared, f"{spec.name} 的 required 里有没声明的参数：{sorted(required - declared)}"

    for name, prop in (spec.schema["properties"] or {}).items():
        bound = _bounds(request["properties"].get(name) or {})
        for key, value in _bounds(prop).items():
            assert key in bound, (
                f"{spec.name}.{name} 写了 {key}，端点那边没写 —— 端点会把超出范围的值收下"
            )
            assert value == bound[key], (
                f"{spec.name}.{name} 的 {key} = {value}，端点上是 {bound[key]} —— "
                "两边各写一份，改了端点这边不会跟着动"
            )


@pytest.mark.parametrize("name", sorted(CASES))
def test_handlers_pass_the_endpoint_payload_straight_through(name: str) -> None:
    args, expected = CASES[name]
    client = _Recorder()
    assert call_tool(client, name, args) is client.payload, (
        f"{name} 把端点的载荷重新加工过了。这一层刻意不做结构建模 —— "
        "建模就有了第二份字段定义，端点改了它不跟着改"
    )
    assert client.calls == [expected], f"{name} 发给端点的请求不是预期的：{client.calls}"


def test_an_unknown_tool_or_a_missing_argument_says_so() -> None:
    client = _Recorder()
    with pytest.raises(QaError) as unknown:
        call_tool(client, "web_search", {"query": "深圳 电动车 新规"})
    assert "web_search" in str(unknown.value) and "rag_search" in str(unknown.value)

    with pytest.raises(QaError) as missing:
        call_tool(client, "rag_search", {})
    assert "question" in str(missing.value)
    assert client.calls == [], "参数都没齐就把请求发出去了"


def test_the_tool_table_never_needs_the_mcp_sdk() -> None:
    proc = _run_child("import sys, mcp_server.tools; print('mcp' in sys.modules)")
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "False", (
        "mcp_server/tools.py 把 MCP SDK 拉进来了：工具表就不再能离线单测，"
        "SDK 换版本也会连坐到这里（SDK 只该在 main.py 里、真起服务时才 import）"
    )

    probe = _run_child("import sys, mcp.server.lowlevel; print('mcp' in sys.modules)")
    assert probe.returncode == 0, probe.stderr[-500:]
    assert probe.stdout.strip() == "True", "探针认不出 mcp，上面那条恒绿"


def test_the_adaptor_only_reaches_into_the_domain_types() -> None:
    bad: list[str] = []
    for path in sorted((ROOT / "mcp_server").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                if parts[0] == "agent_service":
                    bad.append(f"{path.name}:{node.lineno} agent_service")
                    continue
                if parts[0] != "rag_service":
                    continue
                heads = {parts[1]} if len(parts) > 1 else {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.Import):
                heads = set()
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] == "agent_service":
                        bad.append(f"{path.name}:{node.lineno} agent_service")
                    elif parts[0] == "rag_service":
                        heads.add(parts[1] if len(parts) > 1 else "")
            else:
                continue
            if heads - {"domain"}:
                bad.append(f"{path.name}:{node.lineno} {','.join(sorted(heads - {'domain'}))}")
    assert not bad, (
        "适配层伸进了服务端的模块。它只该经 api_contracts.client 走 HTTP，"
        "跨进程能拿到的只有 domain 里的纯类型与错误；\n  剩下四个子包（adapters/api/query/indexing）"
        "一旦 import 进来，这个进程就和服务进程绑在一起了；\n  "
        "agent_service 连 domain 都没有例外 —— 它整个是另一个进程：\n  " + "\n  ".join(bad)
    )


def test_the_adaptor_pulls_no_heavy_dependency_at_import_time() -> None:
    code = (
        "import sys, mcp_server.tools, mcp_server.main, rag_service;"
        f"heavy = {HEAVY!r};"
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules} & set(heavy))))"
    )
    proc = _run_child(code)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", (
        f"MCP 适配层把重依赖拉起来了：{proc.stdout.strip()} —— "
        "载模型/连 Milvus 是服务进程的事，这边只发 HTTP"
    )


def test_the_entry_point_lists_the_tools_offline() -> None:
    proc = subprocess.run(
        [sys.executable, "-P", "-m", "mcp_server", "--list-tools"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    listed = json.loads(proc.stdout)
    assert [item["name"] for item in listed] == NAMES
    assert all(item["schema"]["type"] == "object" for item in listed)


def test_the_stdio_server_round_trips_over_the_mcp_wire() -> None:
    pytest.importorskip("mcp", reason="mcp extra 没装：协议冒烟跳过")
    pytest.importorskip("anyio", reason="mcp extra 没装：协议冒烟跳过")

    import asyncio

    from anyio import create_task_group
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    from mcp_server.main import build_server

    client = _Recorder()
    server = build_server(client)

    async def round_trip() -> tuple[list, object, object, object]:
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with ClientSession(*client_streams) as session:

                async def serve() -> None:
                    await server.run(*server_streams, server.create_initialization_options())

                async with create_task_group() as tg:
                    tg.start_soon(serve)
                    await session.initialize()
                    listed = await session.list_tools()
                    ok = await session.call_tool("rag_laws", {})
                    bad = await session.call_tool("web_search", {"query": "x"})
                    tg.cancel_scope.cancel()
        return listed, ok, bad

    listed, ok, bad = asyncio.run(round_trip())
    assert [tool.name for tool in listed.tools] == NAMES
    assert ok.is_error is False
    assert json.loads(ok.content[0].text) == client.payload, "过了一趟协议，中文或结构被改掉了"
    assert bad.is_error is True, "错误该以 is_error 回给宿主，而不是把整条会话打崩"
    assert "web_search" in bad.content[0].text


def test_the_module_sources_never_import_the_sdk_outside_the_wiring() -> None:
    sdk: dict[str, list[str]] = {}
    for path in sorted((ROOT / "mcp_server").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "mcp":
                sdk.setdefault(path.name, []).append(f"{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "mcp":
                        sdk.setdefault(path.name, []).append(f"{node.lineno}:{alias.name}")
    assert sorted(sdk) == ["main.py"], f"MCP SDK 只能在 main.py 里出现，这些地方也有：{sdk}"
