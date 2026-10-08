from __future__ import annotations

import ast
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import tomllib

import rag_contracts
from rag_contracts.domain.answer import Evidence

PACKAGE = Path(rag_contracts.__file__).resolve().parent
ROOT = PACKAGE.parent
SOURCES = sorted(path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts)
assert len(SOURCES) > 10, (
    f"扫描根不成立：{PACKAGE} 下只有 {len(SOURCES)} 个 .py，下面的门禁会静默放行"
)

CONSUMERS = ("rag_service", "agent_service", "eval", "mcp_server", "api_contracts")

HEAVY = ("pymilvus", "torch", "transformers", "fastapi", "langgraph", "langfuse")

BLOCKER = """import sys

class Blocked(BaseException):
    pass

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in %r:
            raise Blocked(name)
        return None

sys.meta_path.insert(0, Blocker())
"""

TOUCHED = (
    "import rag_contracts, rag_contracts.config, rag_contracts.llm, rag_contracts.ports\n"
    "import rag_contracts.domain, rag_contracts.observability.langfuse\n"
    "import rag_contracts.observability.tracer\n"
    "print('ok')\n"
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


def _absolute_imports(tree: ast.AST) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append((node.lineno, node.module or ""))
    return out


def test_the_contracts_package_never_imports_a_consumer() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{line} import 了 {name}"
        for path in SOURCES
        for line, name in _absolute_imports(ast.parse(path.read_text(encoding="utf-8")))
        if name.split(".")[0] in CONSUMERS
    ]
    assert not offenders, (
        "契约包是叶子，不许反过来 import 它的消费者（函数内的 import 也算）：\n  "
        + "\n  ".join(offenders)
    )


def test_importing_the_contracts_package_pulls_in_no_heavy_dependency() -> None:
    proc = _run_child(BLOCKER % (HEAVY,) + TOUCHED)
    assert proc.returncode == 0, (
        "rag_contracts 在拦截了一切重依赖之后 import 不进来 —— 它是给「没有 torch 的 agent 镜像」"
        "当前提的，这些依赖得留在函数内：\n  " + proc.stderr[-800:]
    )
    assert proc.stdout.strip() == "ok", proc.stderr[-800:]

    probe = _run_child(BLOCKER % (HEAVY,) + "import langfuse\nprint('没拦住')\n")
    assert probe.returncode != 0 and "Blocked" in probe.stderr, (
        "拦截器拦不住 langfuse（或它没装），上面那条门禁恒绿：\n  " + probe.stdout + probe.stderr[-500:]
    )


def test_every_top_level_package_is_declared_for_packaging() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = data["tool"]["setuptools"]["packages"]["find"]["include"]
    dirs = sorted(
        path.name for path in ROOT.iterdir() if path.is_dir() and (path / "__init__.py").exists()
    )

    uncovered = [name for name in dirs if not any(fnmatchcase(name, p) for p in patterns)]
    assert not uncovered, (
        "这些顶层包没被 packages.find.include 覆盖 —— 本机 editable 装在，装出来的镜像里却没有"
        "这一份代码，运行期才炸：\n  " + repr(uncovered) + "\n（现有模式：" + repr(patterns) + "）"
    )

    dangling = [p for p in patterns if not any(fnmatchcase(name, p) for name in dirs)]
    assert not dangling, (
        "packages.find.include 里有模式一个目录都没匹配上（前缀打错了）：\n  " + repr(dangling)
    )


def test_evidence_does_not_carry_the_full_article_over_the_wire() -> None:
    keys = set(Evidence(label="【依据1】", citation="《X》第九十条", text="……", score=1.0).to_dict())
    assert keys == {"label", "citation", "text", "score"}, (
        "Evidence 的载荷键集变了。article 是 rag 进程内的原文对象，吐出去等于同一份 ParentChunk "
        "在一条 payload 里出现两次（retrieval.articles 一次、evidences 一次），"
        "而契约重建侧根本读不到它：\n  " + repr(sorted(keys))
    )
