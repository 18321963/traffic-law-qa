from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "traffic_law_qa"

GATE_EXEMPT = {"pipeline.py"}

HEAVY = ("langgraph", "pymilvus", "openai", "langfuse", "langchain_core")

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


def _legalrag_loads(tree: ast.AST) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if node.func.attr == "load" and isinstance(target, ast.Name) and target.id == "LegalRAG":
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
    gates = _call_lines(tree, "ensure_ready")
    bad = []
    for line in _legalrag_loads(tree):
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


def test_legalrag_is_assembled_behind_the_ready_gate() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path.name in GATE_EXEMPT:
            continue
        offenders += _assemble_before_gate(path)
    assert not offenders, (
        "这些地方装配了 LegalRAG 却没有先调 ensure_ready()（同一函数内、且在装配之前）：\n  "
        + "\n  ".join(offenders)
        + "\n（确实有意不过门的，加进 GATE_EXEMPT 并写明理由）"
    )


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
        "('traffic_law_qa.agents', 'traffic_law_qa.pipeline', 'traffic_law_qa.main') "
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


def test_agent_runner_load_actually_calls_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    graph = pytest.importorskip("traffic_law_qa.agents.graph")

    class _GateHit(Exception):
        pass

    def _boom(**_kwargs):
        raise _GateHit

    monkeypatch.setattr(graph, "ensure_ready", _boom)
    with pytest.raises(_GateHit):
        graph.AgentRunner.load()


def test_the_prompt_teaches_exactly_the_tools_the_model_gets() -> None:
    pytest.importorskip("langgraph", reason="agent extra 没装：只跑结构检查")
    nodes = pytest.importorskip("traffic_law_qa.agents.nodes")
    prompts = pytest.importorskip("traffic_law_qa.agents.prompts")

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
