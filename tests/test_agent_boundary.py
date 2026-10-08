from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent_service"

THIRD_PARTY_ALLOWED = frozenset({"fastapi", "httpx", "langgraph", "pydantic"})
CONTRACT_PACKAGES = frozenset({"rag_contracts", "api_contracts", "agent_service"})
ASSEMBLY_SYMBOLS = frozenset({"RagClient", "LegalRAG", "build_rag"})
RAG_PACKAGE = "rag_service"

BLOCKED_IMPORT_PROBE = r'''
import sys

BLOCKED = "rag_service"


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == BLOCKED or name.startswith(BLOCKED + "."):
            raise ImportError("blocked: " + name)
        return None


sys.meta_path.insert(0, Blocker())

probe = "loaded"
try:
    import rag_service
except ImportError:
    probe = "blocked"

import agent_service.api.app as agent_app

print("probe=" + probe)
print("leftover=" + str(BLOCKED in sys.modules))
print("title=" + agent_app.app.title)
print("file=" + agent_app.__file__)
'''


def _files() -> list[Path]:
    return sorted(AGENT_DIR.rglob("*.py"))


def _imported_tops(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _called_names(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            found.add(target.id)
        elif isinstance(target, ast.Attribute):
            found.add(target.attr)
    return found


def _mentioning_lines(text: str) -> list[int]:
    return [number for number, line in enumerate(text.splitlines(), start=1) if RAG_PACKAGE in line]


def test_the_agent_side_only_reaches_its_two_contract_packages() -> None:
    allowed = set(sys.stdlib_module_names) | THIRD_PARTY_ALLOWED | CONTRACT_PACKAGES
    offenders = {}
    for path in _files():
        extra = sorted(_imported_tops(path) - allowed)
        if extra:
            offenders[str(path.relative_to(ROOT))] = extra

    assert offenders == {}, (
        f"agent 出包后只许依赖 标准库 + {sorted(THIRD_PARTY_ALLOWED)} + {sorted(CONTRACT_PACKAGES)}；"
        f"这几处越界了：{offenders}"
    )


def test_the_dependency_scan_has_teeth(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import rag_service\nfrom rag_service.agents import graph\nimport httpx\n", encoding="utf-8"
    )
    relative = tmp_path / "relative.py"
    relative.write_text("from ..merge import merge_retrievals\nfrom . import state\n", encoding="utf-8")

    assert _imported_tops(probe) == {"rag_service", "httpx"}
    assert _imported_tops(relative) == set(), "相对 import 不是顶层依赖，别把它算成越界"


def test_the_agent_never_names_the_rag_package() -> None:
    hits = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in _files()
        for number in _mentioning_lines(path.read_text(encoding="utf-8"))
    ]

    assert hits == [], (
        f"agent 包里不许出现「{RAG_PACKAGE}」这个字样（一行都不行）：{hits}。"
        "运行期那条由拦截器证，这里是静态面"
    )


def test_the_mention_scan_has_teeth() -> None:
    assert _mentioning_lines("import httpx\nfrom rag_service.agents import graph\n") == [2]
    assert _mentioning_lines("from agent_service.tools import handlers\n") == []


def test_the_rag_implementation_is_constructed_in_exactly_one_module() -> None:
    sites = {}
    for path in _files():
        found = sorted(_called_names(path) & ASSEMBLY_SYMBOLS)
        if found:
            sites[str(path.relative_to(ROOT)).replace("\\", "/")] = found

    assert sites == {"agent_service/container.py": ["RagClient"]}, (
        "生产侧唯一的 rag 实现构造点只有 container.py 一处，且类型必须是 RagClient，不是 LegalRAG、"
        f"也不是 build_rag；这里是门的正面控制（扫不到就说明扫描坏了）：实际 {sites}"
    )


def test_the_agent_boots_with_the_rag_package_blocked_at_import() -> None:
    pytest.importorskip("fastapi", reason="api extra 没装：agent 服务面的导入验收跳过")

    done = subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORT_PROBE],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
    )

    assert done.returncode == 0, f"拦着 {RAG_PACKAGE} 就起不来了：\n{done.stderr}"
    fields = dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)
    assert fields["probe"] == "blocked", (
        f"正向探针：拦截器根本没拦住 {RAG_PACKAGE}，下面那条断言什么也没证明"
    )
    assert fields["leftover"] == "False", f"{RAG_PACKAGE} 还是进了 sys.modules"
    assert fields["title"] == "交通法规问答 Agent 服务"
    assert Path(fields["file"]).resolve().is_relative_to(ROOT)
