from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "traffic_law_qa"

GATE_EXEMPT = {"container.py"}

ASSEMBLY_ALLOWED = {"container.py"}

ASSEMBLY_OWNER = {
    "MilvusStore": {"infra/milvus.py", "indexing/indexer.py"},
    "LocalEmbedder": {"infra/embedding.py", "indexing/indexer.py"},
    "LocalReranker": {"infra/reranking.py"},
    "OpenAILLM": {"infra/llm.py"},
}

HEAVY = ("langgraph", "pymilvus", "openai", "langfuse", "langchain_core", "torch", "transformers")

AGENT_IMPORT_EXEMPT = {"agents", "eval"}


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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
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


def _imports_agent(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        target = node.module or ""
        return node.level >= 1 and (target == "agents" or target.startswith("agents."))
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("traffic_law_qa.agents") for alias in node.names)
    return False


def _type_checking_blocks(tree: ast.AST) -> list[tuple[int, int]]:
    blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(sub, ast.Name) and sub.id == "TYPE_CHECKING"
            for sub in ast.walk(node.test)
        ):
            blocks.append((node.lineno, node.end_lineno or node.lineno))
    return blocks


def _toplevel_agent_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = _functions(tree)
    guarded = _type_checking_blocks(tree)
    bad = []
    for node in ast.walk(tree):
        if not _imports_agent(node):
            continue
        if _innermost(functions, node.lineno) is not None:
            continue
        if any(low <= node.lineno <= high for low, high in guarded):
            continue
        bad.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno}")
    return bad


def test_agent_is_imported_lazily_outside_its_own_package() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.relative_to(PACKAGE).parts[0] in AGENT_IMPORT_EXEMPT:
            continue
        offenders += _toplevel_agent_imports(path)
    assert not offenders, (
        "这些地方在模块顶层 import 了 agent（线性路的部署装不上 agent extra 就起不来）：\n  "
        + "\n  ".join(offenders)
        + "\n（确实有意的，加进 AGENT_IMPORT_EXEMPT 并写明理由）"
    )


def test_rag_is_assembled_behind_the_ready_gate() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path.name in GATE_EXEMPT:
            continue
        offenders += _assemble_before_gate(path)
    assert not offenders, (
        "这些地方装配了 RAG 却没有先过就绪门（同一函数内、且在装配之前调 readiness()）：\n  "
        + "\n  ".join(offenders)
        + "\n（确实有意不过门的，加进 GATE_EXEMPT 并写明理由）"
    )


def test_only_the_container_builds_the_swappable_parts() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = "/".join(path.relative_to(PACKAGE).parts)
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
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("openai"):
                homes.add("/".join(path.relative_to(PACKAGE).parts))
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("openai") for alias in node.names
            ):
                homes.add("/".join(path.relative_to(PACKAGE).parts))
    bad = sorted(homes - {"infra/llm.py"})
    assert not bad, f"只有 infra/llm.py 该直接碰 openai SDK，这些地方绕过了 LLM 端口：{bad}"


def test_import_package_keeps_heavy_deps_out() -> None:
    code = (
        "import sys, traffic_law_qa;"
        f"heavy = {HEAVY!r};"
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules} & set(heavy))))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", f"import traffic_law_qa 拉起了：{proc.stdout.strip()}"


def test_package_door_does_not_pull_the_agent_or_the_builder() -> None:
    code = (
        "import sys, traffic_law_qa.__main__;"
        "print(','.join(sorted(m for m in "
        "('traffic_law_qa.agents', 'traffic_law_qa.indexing.build', 'traffic_law_qa.api.app') "
        "if m in sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "", f"门拉起了：{proc.stdout.strip()}"


def test_the_agent_assembly_never_happens_before_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traffic_law_qa import container
    from traffic_law_qa.app import readiness as ready

    class _GateHit(Exception):
        pass

    def _boom(**_kwargs):
        raise _GateHit

    assembled: list[str] = []
    monkeypatch.setattr(ready, "ensure_ready", _boom)
    monkeypatch.setattr(container, "build_rag", lambda **_kwargs: assembled.append("rag"))
    with pytest.raises(_GateHit):
        container.boot_agent_runner()
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
    "稠密+BM25": "contracts/reports.py",
    "纯 BM25": "contracts/reports.py",
    "已启用": "contracts/reports.py",
    "未启用（仅 BM25）": "contracts/reports.py",
    "未知（Milvus 未连接）": "contracts/reports.py",
    "检索#": "tools/render.py",
    "网搜#": "tools/render.py",
    "材料#": "tools/render.py",
}


def _string_constants(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_shared_wording_is_written_in_exactly_one_module() -> None:
    homes: dict[str, set[str]] = {literal: set() for literal in SINGLE_SOURCE_WORDING}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = "/".join(path.relative_to(PACKAGE).parts)
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
    from traffic_law_qa import prompts
    from traffic_law_qa.agents import nodes

    names = sorted(tool["function"]["name"] for tool in nodes.TOOLS)
    assert names == ["get_article", "search_law", "search_materials"], (
        "工具面变了。下发几个就要在提示词里教几个，改这里之前先想清楚为什么改：\n  " + repr(names)
    )
    missing = [name for name in names if name not in prompts.AGENT_SYSTEM_PROMPT]
    assert not missing, f"这些工具对模型可见，提示词里却一个字没提：{missing}"
    assert "web_search" not in prompts.AGENT_SYSTEM_PROMPT, (
        "提示词教模型用 web_search，但它不在 TOOLS 里 —— 模型会去调一个看不见的工具。"
        "要恢复网搜：把 WEB_SEARCH_TOOL 加回 nodes.TOOLS，并在这里换成 4 个名字。"
    )

    rendered = prompts.AGENT_SYSTEM_PROMPT % {"max_steps": 3, "law_count": 1, "laws": "- 示例法规"}
    assert "最多 3 轮" in rendered and "- 示例法规" in rendered, "提示词是 % 模板，改文本别引入裸 %"


def test_every_offered_tool_has_something_to_run_it() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    from traffic_law_qa.agents import nodes
    from traffic_law_qa.tools import handlers

    offered = {tool["function"]["name"] for tool in nodes.TOOLS}
    missing = sorted(offered - set(handlers.HANDLERS))
    assert not missing, (
        "这些工具下发给模型了，却没有实现 —— 模型一调就拿到「未知工具」，"
        "而且是运行期才炸、测试全绿：\n  " + repr(missing) + "\n"
        "（反方向多出几个是允许的：web_search 就是刻意留着的接线位，"
        "恢复网搜时不必再写一遍实现）"
    )


RETIRED_PACKAGES = {
    "kb": "indexing/",
    "services": "infra/ + app/",
    "qa": "search/ + generation/ + app/",
    "database": "infra/sqlite.py",
    "ready.py": "app/readiness.py",
    "pipeline.py": "indexing/build.py + cli/build.py",
    "main.py": "api/app.py + api/runtime.py + cli/serve.py",
    "agents/cli.py": "cli/ask.py",
}

MERGED_SINGLE_HOMES = {
    "DocxReader": "indexing/sources.py",
    "TextReader": "indexing/sources.py",
    "reader_for": "indexing/sources.py",
    "DocumentRow": "infra/sqlite.py",
    "DDL": "infra/sqlite.py",
}


def _imported_heads(tree: ast.AST) -> set[str]:
    heads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if node.level >= 1:
                heads.add(parts[0])
            elif parts[:1] == ["traffic_law_qa"] and len(parts) > 1:
                heads.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "traffic_law_qa" and len(parts) > 1:
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
        f"traffic_law_qa/{name} 又长出来了（它归 {home}）"
        for name, home in RETIRED_PACKAGES.items()
        if (PACKAGE / name).exists()
    ]
    heads = {name.removesuffix(".py") for name in RETIRED_PACKAGES}
    for path in sorted(list(PACKAGE.rglob("*.py")) + list((PACKAGE.parent / "tests").rglob("*.py"))):
        if "__pycache__" in path.parts:
            continue
        for head in sorted(_imported_heads(ast.parse(path.read_text(encoding="utf-8"))) & heads):
            offenders.append(f"{path.relative_to(PACKAGE.parent)} import 了退役的 {head}")
    assert not offenders, (
        "上一轮的包名又回来了。现在的家：\n  "
        + "\n  ".join(f"{name} → {home}" for name, home in RETIRED_PACKAGES.items())
        + "\n出问题的：\n  "
        + "\n  ".join(offenders)
    )


def test_the_merged_readers_and_ledger_have_exactly_one_home() -> None:
    homes: dict[str, set[str]] = {name: set() for name in MERGED_SINGLE_HOMES}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = "/".join(path.relative_to(PACKAGE).parts)
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
        + "\n（读者注册表在 indexing/sources.py，台账在 infra/sqlite.py）"
    )
