from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import tomllib

import rag_contracts
import rag_service
from rag_contracts import config

PACKAGE = Path(rag_service.__file__).resolve().parent
ROOT = PACKAGE.parent
SOURCES = sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
assert len(SOURCES) > 30, f"扫描根不成立：{PACKAGE} 下只有 {len(SOURCES)} 个 .py，下面的门禁会静默放行"

CONTRACTS_PACKAGE = Path(rag_contracts.__file__).resolve().parent
CONTRACT_SOURCES = sorted(
    path for path in CONTRACTS_PACKAGE.rglob("*.py") if "__pycache__" not in path.parts
)
assert len(CONTRACT_SOURCES) > 10, (
    f"扫描根不成立：{CONTRACTS_PACKAGE} 下只有 {len(CONTRACT_SOURCES)} 个 .py，"
    "契约侧的门禁会静默放行"
)

GATE_EXEMPT = {"container.py"}

ASSEMBLY_ALLOWED = {"container.py"}

ASSEMBLY_OWNER = {
    "MilvusStore": {"adapters/milvus.py"},
    "LocalEmbedder": {"adapters/embedding.py"},
    "LocalReranker": {"adapters/reranking.py"},
    "OpenAILLM": set(),
}

HEAVY = ("langgraph", "pymilvus", "openai", "langfuse", "langchain_core", "torch", "transformers", "pypdf")

TOP_PACKAGES = sorted(
    path for path in ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()
)
assert {"rag_service", "eval"} <= {path.name for path in TOP_PACKAGES}, (
    f"顶层包扫描不成立：{ROOT} 下只找到 {[path.name for path in TOP_PACKAGES]}"
)


def _rel(path: Path) -> str:
    return "/".join(path.relative_to(PACKAGE).parts)


def _rootrel(path: Path) -> str:
    return "/".join(path.relative_to(ROOT).parts)


def _functions(tree: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _call_lines(tree: ast.AST, name: str) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            out.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            out.append(node.lineno)
    return out


def _assembly_calls(tree: ast.AST) -> list[int]:
    out = _call_lines(tree, "build_rag")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if node.func.attr == "load" and isinstance(target, ast.Name) and target.id == "LegalRAG":
                out.append(node.lineno)
    return out


def _constructs(tree: ast.AST, name: str) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            out.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            out.append(node.lineno)
    return out


def _innermost(functions: list[ast.AST], lineno: int) -> ast.AST | None:
    best = None
    for fn in functions:
        if fn.lineno <= lineno <= (fn.end_lineno or fn.lineno):
            if best is None or fn.lineno > best.lineno:
                best = fn
    return best


def _assemble_before_gate(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = _functions(tree)
    gates = _call_lines(tree, "ensure_ready") + _call_lines(tree, "readiness")
    bad = []
    for line in _assembly_calls(tree):
        scope = _innermost(functions, line)
        low = scope.lineno if scope is not None else 0
        if not any(low <= gate < line for gate in gates):
            bad.append(f"{path.relative_to(PACKAGE.parent)}:{line}")
    return bad


def _imports_the_agent_service(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "agent_service"
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "agent_service" for alias in node.names)
    return False


def test_the_rag_side_never_imports_the_agent_service() -> None:
    offenders: list[str] = []
    for path in list(SOURCES) + list(CONTRACT_SOURCES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [
            f"{path.relative_to(ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if _imports_the_agent_service(node)
        ]
    assert not offenders, (
        "rag 侧 import 了 agent 服务 —— 两个服务之间只有 HTTP，import 会把「谁调谁」"
        "重新变回一个进程里的事，出包就白出了：\n  " + "\n  ".join(offenders)
    )


def test_rag_is_assembled_behind_the_ready_gate() -> None:
    offenders: list[str] = []
    for path in SOURCES:
        if _rel(path) in GATE_EXEMPT:
            continue
        offenders += _assemble_before_gate(path)
    assert not offenders, (
        "这些地方装配了 RAG 却没有先过就绪门（同一函数内、且在装配之前调 readiness()）：\n  "
        + "\n  ".join(offenders)
        + "\n（确实有意不过门的，加进 GATE_EXEMPT 并写明理由）"
    )


def test_only_the_container_builds_the_swappable_parts() -> None:
    offenders: list[str] = []
    for path in SOURCES:
        rel = _rel(path)
        if rel.split("/")[0] == "eval" or rel in ASSEMBLY_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, homes in ASSEMBLY_OWNER.items():
            if rel in homes:
                continue
            offenders += [f"{rel}:{line} 自己造了 {name}" for line in _constructs(tree, name)]
    assert not offenders, (
        "换向量库 / 换嵌入 / 换重排 / 换供应商只该改装配点一处，这些地方绕过了 container：\n  "
        + "\n  ".join(offenders)
    )


def test_only_the_llm_adapter_imports_the_openai_sdk() -> None:
    homes: set[str] = set()
    for path in list(SOURCES) + list(CONTRACT_SOURCES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openai"):
                homes.add(_rootrel(path))
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("openai") for alias in node.names
            ):
                homes.add(_rootrel(path))
    assert homes == {"rag_contracts/llm.py"}, (
        "openai SDK 的家该恰好是 rag_contracts/llm.py。多出来的是绕过 LLM 端口的；"
        f"少了说明那份文件搬走了、这条门禁已经空转：{sorted(homes)}"
    )


def _run_child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-P", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_import_package_keeps_heavy_deps_out() -> None:
    code = (
        f"import sys, {PACKAGE.name};"
        f"heavy = {HEAVY!r};"
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules} & set(heavy))))"
    )
    proc = _run_child(code)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", f"import {PACKAGE.name} 拉起了：{proc.stdout.strip()}"

    probe = _run_child(f"import sys, {HEAVY[2]}; print({HEAVY[2]!r} in sys.modules)")
    assert probe.returncode == 0, probe.stderr[-500:]
    assert probe.stdout.strip() == "True", f"探针认不出 {HEAVY[2]}，上面那条门禁恒绿"


def test_package_door_does_not_pull_the_agent_service_or_the_builder() -> None:
    door = f"{PACKAGE.name}.__main__"
    watched = (
        "agent_service",
        f"{PACKAGE.name}.indexing.build",
        f"{PACKAGE.name}.api.app",
    )
    code = f"import sys, {door}; print(','.join(sorted(m for m in {watched!r} if m in sys.modules)))"
    proc = _run_child(code)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", f"门拉起了：{proc.stdout.strip()}"

    probe = _run_child(f"import sys, {watched[2]}; print({watched[2]!r} in sys.modules)")
    assert probe.returncode == 0, probe.stderr[-500:]
    assert probe.stdout.strip() == "True", f"探针认不出 {watched[2]}，上面那条门禁恒绿"


def test_the_rag_assembly_never_happens_before_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_service import container
    from rag_service.api import runtime as server_runtime
    from rag_service.indexing import readiness as ready

    class _GateHit(Exception):
        pass

    def _boom(**_kwargs):
        raise _GateHit

    assembled: list[str] = []
    monkeypatch.setattr(ready, "ensure_ready", _boom)
    monkeypatch.setattr(container, "build_rag", lambda **_kwargs: assembled.append("rag"))
    with pytest.raises(_GateHit):
        server_runtime.boot_runtime()
    assert assembled == [], "就绪门抛了，装配却还是跑过了"


def test_the_http_routes_hide_no_import_inside_a_function() -> None:
    path = PACKAGE / "api" / "routes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = _functions(tree)
    bad = [
        f"api/routes.py:{node.lineno} 在函数里 import 了 {node.module or ''}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and _innermost(functions, node.lineno) is not None
    ]
    assert not bad, (
        "api/routes.py 里出现函数内 import —— 这通常是为了绕开循环导入写的，"
        "症状是「谁把服务和路由装到一起」没有定下来：\n  " + "\n  ".join(bad)
    )


SINGLE_SOURCE_WORDING = {
    "稠密+BM25": "rag_contracts/domain/reports.py",
    "纯 BM25": "rag_contracts/domain/reports.py",
    "已启用": "rag_contracts/domain/reports.py",
    "未启用（仅 BM25）": "rag_contracts/domain/reports.py",
    "未知（Milvus 未连接）": "rag_contracts/domain/reports.py",
    "检索#": "rag_contracts/domain/retrieval.py",
    "网搜#": "rag_contracts/domain/retrieval.py",
    "材料#": "rag_contracts/domain/retrieval.py",
    "length": "rag_contracts/ports.py",
    "docker compose -f deploy/docker-compose.yml": "rag_contracts/config.py",
}


def _string_constants(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_shared_wording_is_written_in_exactly_one_module() -> None:
    homes: dict[str, set[str]] = {literal: set() for literal in SINGLE_SOURCE_WORDING}
    for path in list(SOURCES) + list(CONTRACT_SOURCES):
        rel = _rootrel(path)
        for literal in _string_constants(ast.parse(path.read_text(encoding="utf-8"))):
            if literal in homes:
                homes[literal].add(rel)
    bad = [
        f"「{literal}」写在 {sorted(homes[literal]) or ['哪儿都没有']} —— 它只在 {home} 里定义"
        for literal, home in SINGLE_SOURCE_WORDING.items()
        if homes[literal] != {home}
    ]
    assert not bad, (
        "同一句话抄成了两份（或那份原文不在了），改一处另一处不会跟着变：\n  "
        + "\n  ".join(bad)
        + "\n（引用那个模块里的常量，别把字面串再写一遍）"
    )


def test_the_prompt_teaches_exactly_the_tools_the_model_gets() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    from agent_service import prompts
    from agent_service.agents import nodes

    names = sorted(tool["function"]["name"] for tool in nodes.TOOLS)
    assert names == ["get_article", "search_law", "search_materials"], (
        "工具面变了。下发几个就要在提示词里教几个，改这里之前先想清楚为什么改：\n  " + repr(names)
    )
    missing = [name for name in names if name not in prompts.AGENT_SYSTEM_PROMPT]
    assert not missing, f"这些工具对模型可见，提示词里却一个字没提：{missing}"
    assert "web_search" not in prompts.AGENT_SYSTEM_PROMPT, (
        "提示词教模型用 web_search，但它不在 TOOLS 里 —— 模型会去调一个看不见的工具。"
        "要恢复网搜：把 tools/registry.py 里 web_search 那行的 llm_visible 改成 True，"
        "并在这里换成 4 个名字。"
    )

    rendered = prompts.AGENT_SYSTEM_PROMPT % {"max_steps": 3, "law_count": 1, "laws": "- 示例法规"}
    assert "最多 3 轮" in rendered and "- 示例法规" in rendered, "提示词是 % 模板，改文本别引入裸 %"


def test_the_prompt_quotes_the_material_receipt_verbatim() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    from agent_service import prompts
    from agent_service.tools import schemas
    from rag_service.query.materials import NO_MATERIALS_NOTE

    quoted = "「本次会话没有上传材料」"
    assert quoted in prompts.AGENT_SYSTEM_PROMPT, (
        "系统提示词里对材料回执的引文失真了：模型会照着一个不存在的回执认情况"
    )
    assert quoted in schemas.SEARCH_MATERIALS_TOOL["function"]["description"], (
        "工具描述里对材料回执的引文失真了：模型会照着一个不存在的回执认情况"
    )
    assert "本次会话没有上传材料" in NO_MATERIALS_NOTE, (
        "真回执改了词：上面两处引文没跟着改 —— 引文必须逐字引自 rag_service/query/materials.py"
    )


def test_the_reviewed_prompt_fixes_stay_fixed() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    from agent_service import prompts

    text = prompts.AGENT_SYSTEM_PROMPT
    assert "不要复述你「已命中」了什么。**要引用就引工具返回的原文**" in text, (
        "防幻觉句又被粘回一句：两个分句之间必须有句号断开，否则「已命中」会被读成允许复述的内容"
    )
    assert "续查时，查询里带上法规名片段" in text, (
        "规则 5 少了「续查时」限定，与规则 2「第一次检索一字不改用原话」字面冲突"
    )


def test_the_law_tool_description_hardcodes_no_corpus_counts() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    from agent_service.tools import schemas

    description = schemas.SEARCH_LAW_TOOL["function"]["description"]
    assert "8 部" not in description and "656" not in description, (
        "工具描述里写死法规部数/条数，语料一变就是假话 —— 数字以 GET /health 为准"
    )


def test_every_offered_tool_has_something_to_run_it() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    from agent_service.agents import nodes
    from agent_service.tools import registry

    offered = {tool["function"]["name"] for tool in nodes.TOOLS}
    missing = sorted(offered - set(registry.HANDLERS))
    assert not missing, (
        "这些工具下发给模型了，却没有实现 —— 模型一调就拿到「未知工具」，"
        "而且是运行期才炸、测试全绿：\n  " + repr(missing) + "\n"
        "（反方向多出几个是允许的：web_search 就是刻意留着的接线位——"
        "registry 里 llm_visible=False，恢复网搜时不必再写一遍实现）"
    )


def test_the_tool_registry_names_match_their_schemas() -> None:
    from agent_service.tools import registry

    names = [spec.name for spec in registry.TOOL_REGISTRY]
    assert len(names) == len(set(names)), f"工具名重复：{names}"
    for spec in registry.TOOL_REGISTRY:
        declared = spec.schema["function"]["name"]
        assert spec.name == declared, (
            f"注册表的名字与 schema 里声明的名字对不上：{spec.name} vs {declared}"
        )


def test_handlers_and_tools_are_derived_from_the_same_table() -> None:
    from agent_service.tools import registry

    declared = {spec.schema["function"]["name"] for spec in registry.TOOL_REGISTRY}
    assert set(registry.HANDLERS) == declared, (
        "HANDLERS 的键与注册表 schema 声明的名字对不上（分发面漂了）：\n  "
        f"HANDLERS：{sorted(registry.HANDLERS)}\n  schema：{sorted(declared)}"
    )
    visible = [spec.schema for spec in registry.TOOL_REGISTRY if spec.llm_visible]
    assert registry.TOOLS == visible, (
        "TOOLS 与注册表的 llm_visible 派生结果不一致（被手改了？）：\n  "
        f"TOOLS：{[tool['function']['name'] for tool in registry.TOOLS]}\n  "
        f"应有：{[tool['function']['name'] for tool in visible]}"
    )


def test_web_search_stays_the_only_tool_the_model_cannot_see() -> None:
    from agent_service.tools import registry

    invisible = [spec.name for spec in registry.TOOL_REGISTRY if not spec.llm_visible]
    assert invisible == ["web_search"], (
        "不可见工具集合变了：新增第二个要让模型看不见的工具，先在这行写下理由。"
        f"实际：{invisible}"
    )


RETIRED_PACKAGES = {
    "kb": "indexing/",
    "services": "adapters/ + indexing/ + query/",
    "qa": "query/",
    "database": "adapters/sqlite.py",
    "ready.py": "indexing/readiness.py",
    "pipeline.py": "indexing/build.py + cli/build.py",
    "main.py": "api/app.py + api/runtime.py + cli/serve.py",
    "agents/cli.py": "cli/ask.py",
    "contracts": "domain/",
    "infra": "adapters/",
    "search": "query/",
    "generation": "query/",
    "app": "indexing/ + query/",
    "domain": "rag_contracts/domain/",
    "config.py": "rag_contracts/config.py",
    "ports.py": "rag_contracts/ports.py",
    "adapters/llm.py": "rag_contracts/llm.py",
    "observability": "rag_contracts/observability/ + agent_service/trace.py",
    "observability/tracer.py": "rag_contracts/observability/tracer.py",
    "observability/langfuse.py": "rag_contracts/observability/langfuse.py",
    "observability/trace.py": "agent_service/trace.py",
    "agents": "agent_service/agents/",
    "tools": "agent_service/tools/",
    "adapters/websearch.py": "agent_service/websearch.py",
    "query/merge.py": "agent_service/merge.py",
}

MERGED_SINGLE_HOMES = {
    "DocxReader": "indexing/sources.py",
    "PdfReader": "indexing/sources.py",
    "TextReader": "indexing/sources.py",
    "reader_for": "indexing/sources.py",
    "DocumentRow": "adapters/sqlite.py",
    "DDL": "adapters/sqlite.py",
}


def _imported_heads(tree: ast.AST) -> set[str]:
    heads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if node.level >= 1:
                heads.add(parts[0])
            elif parts[:1] == [PACKAGE.name] and len(parts) > 1:
                heads.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE.name and len(parts) > 1:
                    heads.add(parts[1])
    return heads


def _defined_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out


def test_the_retired_package_names_stay_retired() -> None:
    offenders = [
        f"{PACKAGE.name}/{name} 又长出来了（它归 {home}）"
        for name, home in RETIRED_PACKAGES.items()
        if (PACKAGE / name).exists()
    ]
    heads = {name.removesuffix(".py") for name in RETIRED_PACKAGES}
    scanned = sorted(list(SOURCES) + list((ROOT / "tests").rglob("*.py")))
    for path in scanned:
        if "__pycache__" in path.parts:
            continue
        for head in sorted(_imported_heads(ast.parse(path.read_text(encoding="utf-8"))) & heads):
            offenders.append(f"{path.relative_to(ROOT)} import 了退役的 {head}")
    assert not offenders, (
        "上一轮的包名又回来了。现在的家：\n  "
        + "\n  ".join(f"{name} → {home}" for name, home in RETIRED_PACKAGES.items())
        + "\n出问题的：\n  "
        + "\n  ".join(offenders)
    )


def test_the_merged_readers_and_ledger_have_exactly_one_home() -> None:
    homes: dict[str, set[str]] = {name: set() for name in MERGED_SINGLE_HOMES}
    for path in SOURCES:
        rel = _rel(path)
        for name in _defined_names(ast.parse(path.read_text(encoding="utf-8"))):
            if name in homes:
                homes[name].add(rel)
    bad = [
        f"{name} 定义在 {sorted(homes[name]) or ['哪儿都没有']} —— 它只该在 {home}"
        for name, home in MERGED_SINGLE_HOMES.items()
        if homes[name] != {home}
    ]
    assert not bad, (
        "同一件东西又有了第二份定义（改一处另一处不会跟着变）：\n  "
        + "\n  ".join(bad)
        + "\n（读者注册表在 indexing/sources.py，台账在 adapters/sqlite.py）"
    )


def test_a_top_level_package_never_reaches_above_itself() -> None:
    offenders: list[str] = []
    for package in TOP_PACKAGES:
        for path in sorted(package.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            depth = len(path.relative_to(ROOT).parts) - 1
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level > depth:
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} 上了 {node.level} 层（最远 {depth}）"
                    )
    assert not offenders, (
        "`..` 上溯超出了顶层包本身（那一层不是包，import 会当场炸；跨包请写绝对名）：\n  "
        + "\n  ".join(offenders)
    )


def test_the_milvus_client_extra_follows_every_install_path() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    base = " ".join(project["dependencies"])
    assert "pymilvus" not in base, (
        "pymilvus 回到了基础依赖 —— agent 镜像装的是同一份 pyproject，会把它顺带捎上，"
        "出包隔离就白做了：\n  " + base
    )
    extras = project["optional-dependencies"]
    assert any(req.startswith("pymilvus") for req in extras.get("milvus") or []), (
        "`milvus` 组不见了，或组里没有 pymilvus —— 照提示装了这个组的人连 Milvus 还是会炸：\n  "
        + repr(extras.get("milvus"))
    )
    assert any("milvus" in req for req in extras.get("all") or []), (
        '`all` 组没把 `milvus` 带上（`pip install -e ".[all]"` 是 reinstall.sh 的默认装法）：\n  '
        + repr(extras.get("all"))
    )
    docker = (ROOT / "rag_service" / "Dockerfile").read_text(encoding="utf-8")
    line = next((row for row in docker.splitlines() if '".[' in row), "")
    assert line, "rag_service/Dockerfile 里找不到写死 extras 的 pip install 行，这条门禁已经空转"
    wired = line.partition('".[')[2].partition(']"')[0].split(",")
    assert "milvus" in wired, (
        "rag 镜像的 pip install 没带 `milvus` 组 —— 镜像照样构建成功，"
        "容器一起来在连 Milvus 那步才报「未安装 pymilvus」：\n  " + line.strip()
    )


def test_the_accepted_suffixes_and_the_reader_registry_stay_in_sync() -> None:
    from rag_service.indexing.sources import READER_BY_SUFFIX

    assert set(config.SOURCE_SUFFIXES) == set(READER_BY_SUFFIX), (
        "上传后缀门与读者分派表漂了 —— 两边的语义都是「管线读得懂哪些格式」："
        "后缀表多一项是收得进读不出（建库照样炸），少一项是读得出收不进。"
        f"\n  SOURCE_SUFFIXES={set(config.SOURCE_SUFFIXES)}"
        f"\n  READER_BY_SUFFIX={set(READER_BY_SUFFIX)}"
    )


def test_the_pdf_extra_follows_every_install_path() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    base = " ".join(project["dependencies"])
    assert "pypdf" not in base, (
        "pypdf 回到了基础依赖 —— agent 镜像装的是同一份 pyproject，会把它顺带捎上，"
        "出包隔离就白做了：\n  " + base
    )
    extras = project["optional-dependencies"]
    assert any(req.startswith("pypdf") for req in extras.get("pdf") or []), (
        "`pdf` 组不见了，或组里没有 pypdf：\n  " + repr(extras.get("pdf"))
    )
    assert any("pdf" in req for req in extras.get("all") or []), (
        '`all` 组没把 `pdf` 带上（`pip install -e ".[all]"` 是 reinstall.sh 的默认装法）：\n  '
        + repr(extras.get("all"))
    )
    docker = (ROOT / "rag_service" / "Dockerfile").read_text(encoding="utf-8")
    line = next((row for row in docker.splitlines() if '".[' in row), "")
    assert line, "rag_service/Dockerfile 里找不到写死 extras 的 pip install 行，这条门禁已经空转"
    wired = line.partition('".[')[2].partition(']"')[0].split(",")
    assert "pdf" in wired, (
        "rag 镜像的 pip install 没带 `pdf` 组 —— 建库通道碰到 PDF 会显式报「未安装 pypdf」：\n  "
        + line.strip()
    )


def test_the_missing_pypdf_error_names_the_pdf_extra() -> None:
    text = (PACKAGE / "indexing" / "sources.py").read_text(encoding="utf-8")
    assert "未安装 pypdf" in text and ".[pdf]" in text, (
        "pypdf 的缺失提示没有指向 `.[pdf]` —— 建库通道碰到 PDF 是有意不降级的（显式失败），"
        "提示必须写清装法：sources.py 的 ImportError 里要写明"
    )


def test_the_missing_pymilvus_error_names_the_milvus_extra() -> None:
    text = (PACKAGE / "adapters" / "milvus.py").read_text(encoding="utf-8")
    assert "未安装 pymilvus" in text and ".[milvus]" in text, (
        "pymilvus 的缺失提示没有指向 `.[milvus]` —— 新环境照着旧提示 `pip install -e .` 再装一遍，"
        "装完还是缺：milvus.py 的 ImportError 里要写明装法"
    )


SERVICE_DOCKERFILES = ("rag_service/Dockerfile", "agent_service/Dockerfile")

IMAGE_ENTRY_PACKAGES = {
    "rag_service/Dockerfile": {"rag_service", "eval", "mcp_server"},
    "agent_service/Dockerfile": {"agent_service"},
}

COPY_FACE_EXEMPT = {
    ("rag_service/Dockerfile", "agent_service"): (
        {"eval"},
        "eval.multihop 要 rag 与 agent 两边，是开发侧工具，两镜像都不跑它；rag 镜像里的评测入口只有检索臂",
    ),
    ("agent_service/Dockerfile", "rag_service"): (
        {"api_contracts"},
        "api_contracts/regen.py 是生成面（从 rag_service 生成 openapi.json），运行期不 import",
    ),
}


def _copy_top_levels(text: str) -> set[str]:
    face: set[str] = set()
    for raw in text.splitlines():
        parts = raw.split()
        if not parts or parts[0] != "COPY":
            continue
        for token in parts[1:-1]:
            if token.startswith("--"):
                continue
            face.add(token.rstrip("/").split("/")[0])
    return face


def _imported_top_names(tree: ast.AST) -> set[str]:
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    return tops


def _missing_from_face(face: set[str], needed: set[str], names: set[str]) -> list[str]:
    return sorted((needed & names) - face)


def _exempt_gaps(
    dockerfile: str,
    missing: list[str],
    entries: set[str],
    importers: dict[str, set[str]],
) -> tuple[list[str], set[tuple[str, str]]]:
    gaps: list[str] = []
    live: set[tuple[str, str]] = set()
    for name in missing:
        if name in entries:
            gaps.append(f"{name}（镜像入口包）")
            continue
        allowed, _reason = COPY_FACE_EXEMPT.get((dockerfile, name), (set(), ""))
        offenders = sorted(importers.get(name, set()) - allowed)
        if offenders:
            gaps.append(f"{name}（被这些包里 import：{offenders}）")
        else:
            live.add((dockerfile, name))
    return gaps, live


def _face_gaps(dockerfile: str) -> tuple[list[str], set[tuple[str, str]]]:
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    face = _copy_top_levels(text)
    names = {path.name for path in TOP_PACKAGES}
    entries = set(IMAGE_ENTRY_PACKAGES[dockerfile])
    needed = set(entries)
    importers: dict[str, set[str]] = {}
    for name in sorted(face & names):
        for path in (ROOT / name).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for top in _imported_top_names(ast.parse(path.read_text(encoding="utf-8"))) & names:
                importers.setdefault(top, set()).add(name)
    needed |= set(importers)
    return _exempt_gaps(dockerfile, _missing_from_face(face, needed, names), entries, importers)


def test_every_service_image_copies_the_packages_its_code_imports() -> None:
    offenders: list[str] = []
    live: set[tuple[str, str]] = set()
    for dockerfile in SERVICE_DOCKERFILES:
        gaps, seen = _face_gaps(dockerfile)
        offenders += [f"{dockerfile} 的 COPY 面缺 {gap}" for gap in gaps]
        live |= seen
    offenders += [
        f"豁免失效：{dockerfile} ← {package}（边已不存在，COPY_FACE_EXEMPT 要同步删）"
        for dockerfile, package in sorted(COPY_FACE_EXEMPT)
        if (dockerfile, package) not in live
    ]
    assert not offenders, (
        "镜像 COPY 面没包住镜像内代码要 import 的包 —— 镜像照常构建、CI 全绿，容器起来才炸：\n  "
        + "\n  ".join(offenders)
        + "\n（口径：COPY 面内全部 .py 的顶层绝对 import（含 TYPE_CHECKING 与 try/except 软导入）"
        "并入镜像入口包，都得在 COPY 面里；漏的补对应 Dockerfile 的 COPY 行；"
        "确有意的开发侧边才进 COPY_FACE_EXEMPT，且豁免精确到「允许哪些包 import 它」）"
    )


def test_the_copy_face_scan_has_teeth() -> None:
    face = _copy_top_levels(
        "COPY pyproject.toml fake_alpha/ ./fake_alpha/\n"
        "COPY --chown=app:app fake_beta/ ./fake_beta/\n"
        "RUN echo COPY fake_gamma/\n"
    )
    assert face == {"pyproject.toml", "fake_alpha", "fake_beta"}, face
    tree = ast.parse("import fake_alpha.sub\nfrom fake_beta import thing\nfrom . import sibling\nimport os\n")
    assert _imported_top_names(tree) == {"fake_alpha", "fake_beta", "os"}
    names = {"fake_alpha", "fake_beta", "fake_gamma"}
    needed = {"fake_alpha", "fake_beta", "fake_gamma", "os"}
    assert _missing_from_face(face, needed, names) == ["fake_gamma"]
    dockerfile, package = sorted(COPY_FACE_EXEMPT)[0]
    allowed, _reason = COPY_FACE_EXEMPT[(dockerfile, package)]
    assert _exempt_gaps(dockerfile, [package], set(), {package: allowed}) == ([], {(dockerfile, package)})
    assert _exempt_gaps(dockerfile, [package], set(), {package: allowed | {"fake_alpha"}})[0] == [
        f"{package}（被这些包里 import：['fake_alpha']）"
    ]
    assert _exempt_gaps("fake/Dockerfile", ["fake_gamma"], set(), {"fake_gamma": {"fake_alpha"}}) == (
        ["fake_gamma（被这些包里 import：['fake_alpha']）"],
        set(),
    )
    assert _exempt_gaps("fake/Dockerfile", ["fake_gamma"], {"fake_gamma"}, {}) == (
        ["fake_gamma（镜像入口包）"],
        set(),
    )


AGENT_PACKAGE = ROOT / "agent_service"
AGENT_SOURCES = sorted(
    path for path in AGENT_PACKAGE.rglob("*.py") if "__pycache__" not in path.parts
)
assert len(AGENT_SOURCES) > 20, (
    f"扫描根不成立：{AGENT_PACKAGE} 下只有 {len(AGENT_SOURCES)} 个 .py，存储门禁会静默放行"
)


def _sqlite_connect_lines(tree: ast.AST) -> list[int]:
    connections = {"sqlite3"}
    factories: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sqlite3":
                    connections.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
            for alias in node.names:
                if alias.name == "connect":
                    factories.add(alias.asname or alias.name)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "connect":
            if isinstance(func.value, ast.Name) and func.value.id in connections:
                lines.append(node.lineno)
        elif isinstance(func, ast.Name) and func.id in factories:
            lines.append(node.lineno)
    return sorted(lines)


def test_the_agent_side_opens_sqlite_only_in_the_storage_module() -> None:
    storage_path = AGENT_PACKAGE / "storage.py"
    probe = _sqlite_connect_lines(ast.parse(storage_path.read_text(encoding="utf-8")))
    assert probe, "探针失明：agent_service/storage.py 里认不出 sqlite3.connect 调用，本门会恒绿"
    offenders = [
        f"{_rootrel(path)}:{line}"
        for path in AGENT_SOURCES
        if path != storage_path
        for line in _sqlite_connect_lines(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, (
        "agent 侧的 sqlite 连接只许在 agent_service/storage.py 里开"
        "（import sqlite3 本身不拦，类型标注可用；开连接要过 storage 的开库函数）：\n  "
        + "\n  ".join(offenders)
    )


def test_the_agent_storage_scan_has_teeth() -> None:
    assert _sqlite_connect_lines(ast.parse("import sqlite3\nsqlite3.connect('x')\n")) == [2]
    assert _sqlite_connect_lines(ast.parse("import sqlite3 as s\ns.connect('x')\n")) == [2]
    assert _sqlite_connect_lines(ast.parse("from sqlite3 import connect\nconnect('x')\n")) == [2]
    assert _sqlite_connect_lines(ast.parse("from sqlite3 import connect as c\nc('x')\n")) == [2]
    assert _sqlite_connect_lines(ast.parse("import sqlite3\nsqlite3.Connection('x')\n")) == []
    assert _sqlite_connect_lines(ast.parse("MilvusIndexer().connect()\n")) == []


EVAL_PACKAGE = ROOT / "eval"
EVAL_SOURCES = sorted(path for path in EVAL_PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
assert len(EVAL_SOURCES) > 8, (
    f"扫描根不成立：{EVAL_PACKAGE} 下只有 {len(EVAL_SOURCES)} 个 .py，下面的门禁会静默放行"
)

EVAL_TO_RAG = {
    ("eval/harness.py", "rag_service.adapters.milvus"),
    ("eval/harness.py", "rag_service.adapters.sqlite"),
    ("eval/harness.py", "rag_service.api"),
    ("eval/harness.py", "rag_service.api.facade"),
    ("eval/harness.py", "rag_service.api.runtime"),
    ("eval/harness.py", "rag_service.indexing.chunker"),
    ("eval/harness.py", "rag_service.query.articles"),
    ("eval/multihop.py", "rag_service.adapters.sqlite"),
    ("eval/multihop.py", "rag_service.api"),
    ("eval/multihop.py", "rag_service.api.facade"),
    ("eval/multihop.py", "rag_service.api.runtime"),
    ("eval/multihop.py", "rag_service.indexing.chunker"),
    ("eval/multihop.py", "rag_service.query.articles"),
    ("eval/singlehop.py", "rag_service.api.facade"),
}


def _rag_imports_in_text(rel: str, text: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "rag_service":
                    found.add((rel, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "rag_service":
                found.add((rel, node.module))
    return found


def _rag_imports_in_eval() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in EVAL_SOURCES:
        found |= _rag_imports_in_text(_rootrel(path), path.read_text(encoding="utf-8"))
    return found


def test_the_eval_package_only_reaches_rag_through_registered_edges() -> None:
    found = _rag_imports_in_eval()
    added = sorted(found - EVAL_TO_RAG)
    gone = sorted(EVAL_TO_RAG - found)
    assert not added and not gone, (
        "eval 直取 rag_service 的边变了。eval 是仓内评测工具（在进程内起真 app、跑真检索），"
        "按设计不走 HTTP —— 所以这条边允许存在，但必须逐条登记：\n"
        f"  多出来的：{added or '无'}\n  已经不存在的：{gone or '无'}\n"
        "（新增的要么补进 EVAL_TO_RAG 并说明是哪条臂在用，要么改走 `--http URL` 那条臂：走服务端点、不 import rag）"
    )


def test_the_eval_rag_edge_scan_has_teeth() -> None:
    probe = (
        "from rag_service.api.facade import QaError\n"
        "import rag_contracts\n"
        "\n"
        "\n"
        "def f():\n"
        "    import rag_service.query.rag\n"
    )
    assert _rag_imports_in_text("eval/harness.py", probe) == {
        ("eval/harness.py", "rag_service.api.facade"),
        ("eval/harness.py", "rag_service.query.rag"),
    }
    assert ("eval/singlehop.py", "rag_service.api.facade") in _rag_imports_in_eval(), (
        "探针失明：连既有的那条 eval → rag_service 边都扫不出来，上面那条门禁恒绿"
    )


FRONTEND_SSE = ROOT / "frontend" / "src" / "api" / "sse.ts"
SSE_EMITTERS = ("rag_service/api/app.py", "agent_service/api/app.py")
SSE_FRAMES = ("delta", "done", "error", "evidence", "interrupt", "step")
SSE_FRAMES_THE_FRONTEND_IGNORES = {"evidence"}


def _frames_in_source(text: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        head = node.args[0]
        if isinstance(func, ast.Name) and func.id == "_sse":
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                names.add(head.value)
        elif isinstance(func, ast.Attribute) and func.attr == "put":
            if isinstance(head, ast.Tuple) and head.elts:
                first = head.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
    return names


def _frontend_frames_in_source(text: str) -> set[str]:
    return set(re.findall(r'frame\.event === "([a-zA-Z_]+)"', text))


def test_the_frontend_knows_exactly_the_frames_the_two_services_emit() -> None:
    emitted: set[str] = set()
    for rel in SSE_EMITTERS:
        emitted |= _frames_in_source((ROOT / rel).read_text(encoding="utf-8"))
    assert emitted == set(SSE_FRAMES), (
        "两端 SSE 帧名的并集变了。帧名是跨端契约（前端按名分派，未知帧会被丢掉）：\n"
        f"  多出来的：{sorted(emitted - set(SSE_FRAMES))}\n"
        f"  已经不发的：{sorted(set(SSE_FRAMES) - emitted)}"
    )
    known = _frontend_frames_in_source(FRONTEND_SSE.read_text(encoding="utf-8"))
    expected = set(SSE_FRAMES) - SSE_FRAMES_THE_FRONTEND_IGNORES
    assert known == expected, (
        "前端的 sse.ts 认的帧名与后端对不上（认了不发的名＝死分支；漏了发的名＝帧被静默丢掉）：\n"
        f"  前端认的：{sorted(known)}\n  该认的：{sorted(expected)}\n"
        "（确实有意忽略的帧，写进 SSE_FRAMES_THE_FRONTEND_IGNORES）"
    )


def test_the_sse_frame_scan_has_teeth() -> None:
    assert _frames_in_source('_sse("delta", payload)\nout.put(("done", payload))\n') == {"delta", "done"}
    assert _frames_in_source("_sse(name, payload)\n") == set(), (
        "探针把非常量帧名也算进来了，上面那条门禁会误报"
    )
    assert _frontend_frames_in_source('if (frame.event === "step") return null;\n') == {"step"}


DOCKERIGNORE = ROOT / ".dockerignore"
IMAGE_DATA_KEEP = frozenset(
    {
        "eval_corpus.json",
        "eval_multihop.json",
        "eval_nogold.json",
        "eval_reference.json",
        "eval_retrieval.json",
    }
)
IMAGE_MUST_IGNORE = (
    "data/documents.db",
    "data/sessions.db",
    "data/usage.db",
    "data/uploads/2026-10-11.x.docx",
    "data/traces/single82_q3next.json",
    "deploy/backup/20261009-205732/milvus.tar.gz",
    "volumes/milvus/data/flush",
)


def _dockerignore_patterns(text: str) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(("#", "!"))
    )


def _dockerignored(rel: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if rel == pattern.rstrip("/") or rel.startswith(pattern):
                return True
        elif fnmatchcase(rel, pattern) or fnmatchcase(rel.split("/")[-1], pattern):
            return True
    return False


def test_the_build_never_carries_local_state_into_the_images() -> None:
    patterns = _dockerignore_patterns(DOCKERIGNORE.read_text(encoding="utf-8"))
    leaked = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "data").rglob("*"))
        if path.is_file()
        and path.name not in IMAGE_DATA_KEEP
        and not _dockerignored(_rootrel(path), patterns)
    ]
    unguarded = [rel for rel in IMAGE_MUST_IGNORE if not _dockerignored(rel, patterns)]
    assert not leaked and not unguarded, (
        "本机运行期状态会进构建上下文／镜像层（镜像层会被 push 与 `docker history` 看到）：\n"
        f"  data/ 里没被挡掉的：{leaked or '无'}\n  规则没盖住的：{unguarded or '无'}\n"
        f"（.dockerignore 里 data/ 只该放行题集桶文件：{sorted(IMAGE_DATA_KEEP)}；"
        "确有意放行的，补进 IMAGE_DATA_KEEP 并说明为什么）"
    )


def test_the_dockerignore_scan_has_teeth() -> None:
    patterns = _dockerignore_patterns("# 注释\n!keep.txt\n  data/*.db  \ndata/uploads/\n")
    assert patterns == ("data/*.db", "data/uploads/"), patterns
    assert _dockerignored("data/sessions.db", patterns)
    assert _dockerignored("data/uploads/x/y.docx", patterns)
    assert not _dockerignored("data/eval_corpus.json", patterns)
    assert not _dockerignored("volumes/milvus/data/flush", patterns), (
        "匹配器把没写规则的东西也当成已挡掉了，上面那条门禁恒绿"
    )


OWN_SOURCES = sorted(
    path
    for root in [*TOP_PACKAGES, ROOT / "tests"]
    for path in root.rglob("*.py")
    if "__pycache__" not in path.parts
)
assert len(OWN_SOURCES) > 100, (
    f"扫描根不成立：自研 Python 只找到 {len(OWN_SOURCES)} 个，下面的门禁会静默放行"
)

PACKAGE_SOURCES = sorted(
    path for package in TOP_PACKAGES for path in package.rglob("*.py") if "__pycache__" not in path.parts
)

README_REF_SOURCES = [
    path for path in PACKAGE_SOURCES if path.relative_to(ROOT).parts[:2] != ("eval", "mutations")
]

NOQA_ONLY = re.compile(r"^#\s*noqa(?::\s*[A-Z0-9,\s]+)?$")
README_PATH = ROOT / "README.md"
README_REF = re.compile(r"README「([^」]+)」")


def _prose_comments(text: str) -> list[int]:
    return [
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT and not NOQA_ONLY.match(token.string.strip())
    ]


def _docstring_lines(text: str) -> list[int]:
    out: list[int] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node):
                out.append(getattr(node, "lineno", 1))
    return sorted(out)


def _readme_headings(text: str) -> set[str]:
    return {line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")}


def _unused_usage_modules() -> list[str]:
    offenders: list[str] = []
    for path in OWN_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declared = any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(getattr(target, "id", "") == "USAGE" for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            ))
            for node in tree.body
        )
        if not declared:
            continue
        used = any(
            isinstance(node, ast.Name) and node.id == "USAGE" and isinstance(node.ctx, ast.Load)
            for node in ast.walk(tree)
        )
        if not used:
            offenders.append(_rootrel(path))
    return offenders


def test_no_prose_comments_and_no_docstrings_in_our_own_python() -> None:
    offenders: list[str] = []
    for path in OWN_SOURCES:
        text = path.read_text(encoding="utf-8")
        offenders += [f"{_rootrel(path)}:{line} 散文注释" for line in _prose_comments(text)]
        offenders += [f"{_rootrel(path)}:{line} docstring" for line in _docstring_lines(text)]
    assert not offenders, (
        "本仓刻意零散文注释、零 docstring（用法文本放模块级 USAGE 常量，成因写进断言消息或文档）：\n  "
        + "\n  ".join(offenders)
        + "\n（`# noqa: <码>` 是机器指令，不算散文注释）"
    )


def test_every_usage_constant_is_actually_referenced() -> None:
    offenders = _unused_usage_modules()
    assert not offenders, (
        "这些模块定义了 USAGE 却没有任何地方用它 —— 用法文本会烂在文件里，"
        "`--help` 与报错提示都看不到它：\n  " + "\n  ".join(offenders) + "\n（要么用上，要么删掉）"
    )


def test_every_readme_section_reference_points_at_a_real_heading() -> None:
    headings = _readme_headings(README_PATH.read_text(encoding="utf-8"))
    offenders = [
        f"{_rootrel(path)} → README「{name}」"
        for path in README_REF_SOURCES
        for name in README_REF.findall(path.read_text(encoding="utf-8"))
        if name not in headings
    ]
    assert not offenders, (
        "自检文案按名引用了 README 里不存在的小节 —— 改 README 章节标题前先 `grep -rn \"README「\"`：\n  "
        + "\n  ".join(offenders)
        + f"\n（README 现有小节：{sorted(headings)}；"
        "扫描不含 eval/mutations/ —— 那里面是变异夹具，故意写着坏样例）"
    )


def test_the_wording_and_readme_reference_scans_have_teeth() -> None:
    probe = (
        '"""模块说明。"""\n'
        "\n"
        "import os\n"
        "\n"
        "\n"
        "def f():\n"
        '    """函数说明。"""\n'
        "    x = 1  # 这里解释一下\n"
        "    y = 2  # noqa: E402\n"
        "    return x + y\n"
    )
    assert _docstring_lines(probe) == [1, 6]
    assert _prose_comments(probe) == [8]
    assert _readme_headings("## 入口\n\n### 端点\n\n普通行 README「入口」\n") == {"入口", "端点"}
    assert README_REF.findall('raise SystemExit("清单见 README「入口」")') == ["入口"]
    assert ROOT / "eval" / "harness.py" in README_REF_SOURCES, "排除面开太大，把 eval/ 的正文也跳过了"
