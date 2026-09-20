"""四个入口的开关校验 —— 全部离线，一条都不触发真正的检索或 LLM 调用。

`test_server_cli.py` 钉的是「`--help` 不该把服务起起来」。这里是同一类问题的另一半：
原先每个入口各自手搓一个 `option()`，只做 `if name in args`，于是**不认识的开关被
静默忽略、取不到值的开关静默退回默认值**。

安静本身不是问题，问题是它安静地改变了要花多少钱：

- `singlehop --compare --limit 5` 把 `--limit` 敲错一个字母 → `option()` 返回 None
  → `run(limit=None)` 就是**全量 82 道**，两条臂都真调 LLM。文档里那句
  「先跑 5 道看链路」于是变成一次全额付费。
- `multihop --generate --target 20` 同理，静默退回 100 道。

现在四个入口都换成 argparse、只做校验。**`--help` 与「不带参数」仍由各自开头那个分支
打印 `USAGE`**（`add_help=False`，argparse 的自动帮助不如手写的那份全），所以这两条
路径的输出逐字节未变 —— 下面那两条帮助测试就是钉这个的。
"""

from __future__ import annotations

import pytest

from traffic_law_qa.agent import cli
from traffic_law_qa.agent.intent import INTENT_SEARCH
from traffic_law_qa.eval import harness, multihop, singlehop

ENTRIES = [cli, harness, multihop, singlehop]
IDS = [module.__name__ for module in ENTRIES]

# 三类坏参数 × 四个入口。每一类都在改动前有各自的难看法：
# 不认识的开关 = 静默忽略后照常干活，取不到值 = 静默用默认值，类型不对 = 裸 ValueError。
坏参数 = [
    pytest.param(cli, ["--maxstep", "1", "醉驾怎么处罚"], id="cli-不认识"),
    pytest.param(cli, ["--bogus", "醉驾怎么处罚"], id="cli-不认识2"),
    pytest.param(cli, ["醉驾怎么处罚", "--max-steps"], id="cli-取不到值"),
    pytest.param(cli, ["醉驾怎么处罚", "--top-k", "abc"], id="cli-类型不对"),
    pytest.param(harness, ["--bogus"], id="harness-不认识"),
    pytest.param(harness, ["--limit"], id="harness-取不到值"),
    pytest.param(harness, ["--limit", "abc"], id="harness-类型不对"),
    pytest.param(multihop, ["--check", "--bogus"], id="multihop-不认识"),
    pytest.param(multihop, ["--check", "--top-k"], id="multihop-取不到值"),
    pytest.param(multihop, ["--check", "--top-k", "abc"], id="multihop-类型不对"),
    pytest.param(singlehop, ["--compare", "--limitt", "5"], id="singlehop-敲错一个字母"),
    pytest.param(singlehop, ["--compare", "--limit"], id="singlehop-取不到值"),
    pytest.param(singlehop, ["--compare", "--limit", "abc"], id="singlehop-类型不对"),
]


@pytest.mark.parametrize("module,argv", 坏参数)
def test_坏开关报错而不是静默降级(module, argv, capsys):
    """返回 2（与 argparse 的约定一致，也和 pipeline.py 对未知子命令的处理一致）。

    stdout 必须为空 —— 它同时是「没打印 USAGE」和「没开始干活」的证据：
    argparse 的报错只走 stderr，任何真正的评测输出都会落到 stdout。
    """
    assert module.main(list(argv)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err


@pytest.mark.parametrize("module", ENTRIES, ids=IDS)
def test_帮助仍然打印各自手写的_USAGE(module, capsys):
    """`add_help=False` 是为了这个：argparse 的自动帮助覆盖不了 USAGE 里那几段例子。"""
    assert module.main(["--help"]) == 0
    assert "python -m traffic_law_qa" in capsys.readouterr().out


@pytest.mark.parametrize("module", [cli, multihop, singlehop], ids=["cli", "multihop", "singlehop"])
def test_这三个入口不带参数时打印_USAGE(module, capsys):
    """`harness` 不在此列，而且是**刻意**不在：它不带参数就是跑全量 82 道题，
    那是它的默认用法（`python -m traffic_law_qa.eval`），不是「忘了给参数」。"""
    assert module.main([]) == 0
    assert "python -m traffic_law_qa" in capsys.readouterr().out


class _记账装配:
    """顶掉 `AgentRunner`：把 `load()` 收到的关键字记下来，然后当场停住。

    记完就抛，因为装配真的跑起来要连 Milvus —— 下面两条测试都必须离线。

    它顶的是一个**类**（`AgentRunner.load` 是 classmethod），所以只要有个 `load`
    属性就够；`main` 里是按模块全局名查的，`monkeypatch.setattr(cli, ...)` 打得着。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def load(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("哨兵：走到装配就够了")


@pytest.fixture
def 装配记录(monkeypatch) -> _记账装配:
    sentinel = _记账装配()
    monkeypatch.setattr(cli, "AgentRunner", sentinel)
    return sentinel


def test_cli_的_trace_仍然被收下(装配记录, capsys):
    """全仓唯一一处**不能**让「不认识的开关」报错的地方。

    `--trace` 写在 `USAGE` 里，但从加进来那天起就没被代码读过 —— 轨迹本来就是默认
    打印的，它一直是个空开关。argparse 一上来就会把它判成「不认识的开关」，
    那等于打断一条一直在用的命令。所以收下它，并在 `USAGE` 里把这件事说清楚。

    反过来说，文档里出现的另外两个开关**不在此列**，拒绝它们才是对的：
    `harness` 的 `--in-domain` 是「域内外分开报」那套机制拆除后留下的字（harness.py
    开头写着），`multihop` 开头那句 `--reference` 指的是 harness 的开关，是散文。

    这条走 `main` 而不是解析器：**解析器是 `main` 的实现细节**，而「这条命令还能用」
    是 `main` 的行为。
    """
    assert cli.main(["醉驾怎么处罚", "--trace"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "", "argparse 把 --trace 判成了不认识的开关"
    assert "装配失败" in captured.out, "没走到装配那一步，说明中途就返回了"


def test_cli_把解析出来的开关原样交给装配(装配记录):
    """解析出来的值必须**一路透到 `load()`**，中途换成常量就算数字对不上也看不出来。

    上面那些测试只证明「坏参数会被拒」，证明不了一件事：**好参数有没有被用**。
    这里钉的就是那一跳 —— 解析器的 `--top-k 3` 与 `load(top_k=...)` 之间隔着一个
    `options.top_k`，写成 `top_k=None`（或写成任何常量）时，argparse 那半边照样全绿，
    而 `--top-k` 从此静默失效：`USAGE` 里写着「证据条数上限，默认取 RAG_TOP_K」，
    给了也不生效。

    断言落在**收到的关键字**上，与 `test_server_cli.py::test_默认值与改用_argparse_之前逐位相同`
    同一个形状（那边记的是 `host`/`port`）。**两条调用都要**：只跑带开关的那次，
    「永远传 3」这种写死也能过 —— 空值必须原样传 None，那是「默认取 RAG_TOP_K」的
    唯一实现方式。
    """
    assert cli.main([
        "醉驾怎么处罚",
        "--top-k", "3",
        "--max-steps", "4",
        "--intent", "search",
        "--no-vector",
    ]) == 1
    assert cli.main(["醉驾怎么处罚"]) == 1

    给了, 没给 = 装配记录.calls

    assert 给了["top_k"] == 3
    assert 给了["cfg"].max_steps == 4
    assert 给了["forced_intent"] == INTENT_SEARCH
    assert 给了["with_vector"] is False

    assert 没给["top_k"] is None, "不给开关时该传 None，让它去取 RAG_TOP_K"
    assert 没给["forced_intent"] is None, "没给 --intent 时不该塞一个默认意图进去"
    assert 没给["with_vector"] is True
