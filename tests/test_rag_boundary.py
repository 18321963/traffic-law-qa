"""结构边界：`HybridRetriever` 只准被 `qa/rag.py` 认识。

`qa/rag.py` 的自述写着「上层（pipeline / FastAPI / Agent）只依赖它」。这句话曾经是假的 ——
六处上层代码直接 import 检索器、伸手取 `.store` / `.chunks` / `.rewriter`。
这个文件把那句话变成**可执行的**：以后谁再伸手，这里先红。

**为什么必须用 AST 而不是正则**：`pipeline.py` 有一个运行时常量元组
`("retrieve ", "HybridRetriever", "Query", "RetrievalResult")`，`contracts.py` /
`pipeline.py` / `qa/rag.py` 的模块自述里也都写着这个词，`qa/retriever.py` 里还有
`class HybridRetriever` 的定义与 `-> "HybridRetriever"` 的字符串注解。
正则会把它们全算成「引用了检索器」，于是要么误报、要么只能把模式放宽到没有意义。
`ast.parse` 只看真正的 `Import` / `ImportFrom` / `Name` 节点，一个都不误伤。

**它不覆盖什么**（别把它当成万能的锁）：
- `importlib.import_module("...retriever")` —— 字符串里的模块路径看不见；
- `getattr(obj, "sto" + "re")` 这类拼出来的属性名；
- 鸭子类型地传一个「长得像检索器」的对象（测试替身就是这么干的，那是有意的）。

所以它锁的是**正常写法**：正常写法的越界会红，刻意绕过它写法的越界靠 code review。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "traffic_law_rag"
# 包内相对路径，不是文件名 —— 子包化之后 `path.name` 不再唯一（六个 __init__.py）。
FACADE = "qa/rag.py"
TARGET = "HybridRetriever"

# 只有这两个文件准提：qa/retriever.py 是定义它的地方，qa/rag.py 是唯一被允许认识它的门面。
ALLOWED = {"qa/retriever.py", FACADE}


@pytest.fixture(scope="module")
def parents(chunk_set) -> dict:
    """父块用真语料（现场跑切块）—— 这一层测的是结构，但没必要用假对象。"""
    return chunk_set.parent_map()


def _modules() -> list[Path]:
    """包内**所有**模块，含子包。

    `rglob` 不是随手写的：原来这里是 `glob("*.py")`，只扫顶层。子包化那次重构
    如果把模块搬进 `kb/` `qa/`，`glob` 会**静默地**少扫一大半 —— 测试照样全绿，
    只是边界没人守了。所以这条现在是递归的，并且 `_key()` 用相对路径做身份，
    避免子包里的同名文件（`__init__.py`）互相覆盖。
    """
    files = sorted(PACKAGE.rglob("*.py"))
    assert files, f"没找到任何模块：{PACKAGE}"
    return files


def _key(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def _imported_names(tree: ast.AST) -> set[str]:
    """这个模块通过 import 语句拿到了哪些名字。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[-1] for alias in node.names)
    return names


def _used_names(tree: ast.AST) -> set[str]:
    """这个模块在**代码位置**用到了哪些标识符（不含字符串与 docstring）。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_only_the_facade_imports_the_retriever():
    offenders = {}
    for path in _modules():
        if _key(path) in ALLOWED:
            continue
        if TARGET in _imported_names(ast.parse(path.read_text(encoding="utf-8"))):
            offenders[_key(path)] = "import"
    assert not offenders, (
        f"这些模块直接 import 了 {TARGET}，应当改走 LegalRAG：{sorted(offenders)}"
    )


def test_only_the_facade_names_the_retriever_in_code():
    """比 import 更严一层：**在代码里提到这个名字**也不行。

    只挡 import 是不够的 —— `from .retriever import HybridRetriever` 换成
    `import traffic_law_rag.qa.retriever as r` 再写 `r.HybridRetriever(...)` 就绕过去了。
    这里扫 `ast.Name` / `ast.Attribute`，把这条路也堵上。
    （类型注解也算：它是代码位置，不是字符串。）
    """
    offenders = {}
    for path in _modules():
        if _key(path) in ALLOWED:
            continue
        used = _used_names(ast.parse(path.read_text(encoding="utf-8")))
        if TARGET in used:
            offenders[_key(path)] = "name"
    assert not offenders, (
        f"这些模块在代码里提到了 {TARGET}，应当改走 LegalRAG：{sorted(offenders)}"
    )


def test_the_facade_really_does_import_it():
    """正向断言 —— 没有它，上面两条会因为「谁都别 import」而变成永真。

    一条只会在「有人真的引用了」时失败的测试，是半个测试；它还得在
    「门面自己也不引用了」时失败，否则 `qa/rag.py` 被改成空壳也没人知道。
    """
    tree = ast.parse((PACKAGE / FACADE).read_text(encoding="utf-8"))
    assert TARGET in _imported_names(tree), f"{FACADE} 应当 import {TARGET}"
    assert TARGET in _used_names(tree), f"{FACADE} 应当在代码里用上 {TARGET}"


def test_boundary_scanner_would_actually_catch_a_violation():
    """给扫描器本身做个阳性对照 —— 免得它哪天悄悄退化成永远返回空。"""
    tree = ast.parse("from .retriever import HybridRetriever\nx = HybridRetriever\n")
    assert TARGET in _imported_names(tree)
    assert TARGET in _used_names(tree)

    # 而字符串与 docstring 里的同一个词不该被算进来（这正是不能用正则的原因）
    literal = ast.parse('"""别的模块提到 HybridRetriever。"""\nLAYERS = ("HybridRetriever",)\n')
    assert TARGET not in _imported_names(literal)
    assert TARGET not in _used_names(literal)


# ================================================================== 属性那扇门
def test_the_retriever_keeps_its_innards_private(parents):
    """`store` / `chunks` / `rewriter` 是私有的；伸手进来要当场报错。

    这才是真正把门关上的那一手：AST 测试只管 import，管不到属性穿透。
    私有化之后，越界不是「静默拿到 None」，而是一次 `AttributeError`。
    """
    from traffic_law_rag.qa.retriever import HybridRetriever

    retriever = HybridRetriever(store=object(), parents=parents, chunks={})
    for name in ("store", "chunks", "rewriter"):
        assert not hasattr(retriever, name), f"{name} 不该是公开属性"
        with pytest.raises(AttributeError):
            getattr(retriever, name)


def test_boundary_covers_every_module_in_the_package():
    """确认扫的是整个包（含子包），不是恰好扫到几个文件就通过。"""
    names = {_key(path) for path in _modules()}
    assert {"qa/rag.py", "qa/retriever.py", "agent/graph.py", "api.py", "pipeline.py"} <= names
    assert len(names) >= 20, names
    # 四个子包各至少有一个模块被扫到 —— 单靠总数看不出一整个子包漏掉
    for sub in ("kb/", "qa/", "agent/", "eval/"):
        assert any(name.startswith(sub) for name in names), f"{sub} 一个模块都没扫到"
