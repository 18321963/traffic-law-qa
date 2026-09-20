"""`tlq-serve` 的入口测试 —— 全部离线，不起服务、不连 Milvus。

与 `test_pipeline_cli.py` 同一个理由，但这一侧更重：`pipeline --help` 曾经真的跑了一次
重建（花 embedding 的钱、42 秒），`server --help` 则是**真的把服务起起来** ——
既占住 8000 端口，又跑 lifespan 里的 `_boot()`：就绪检查（索引缺失或过期时触发重建，
那要重新向量化）加稠密通道预热。想查一次用法，代价是一个跑着的服务。

根因是同一个：手搓的参数解析只做 `if name in args`，没有任何东西拦在真正干活之前。
换成 argparse 之后 `--help` 在解析阶段就被拦下，`main()` 把它的 `SystemExit(0)`
收回成返回值。所以这里断言的不是「有没有帮助文本」，而是**帮助路径绝不调到 uvicorn.run**。

`server.py` 要 `[api]` 那组依赖，没装就整组跳过（`test_agent.py` 对 langgraph 是同一做法）。
"""

from __future__ import annotations

import pytest

uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from traffic_law_qa import server  # noqa: E402


@pytest.fixture
def 起服务调用记录(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """把 `uvicorn.run` 换成记录器。

    记录而不抛异常，是为了让「正常参数」那几条也能复用同一个 fixture ——
    这里被换掉的 `uvicorn.run` 本身就是纯副作用，不会造成真连接。
    """
    calls: list[dict] = []

    def recorder(target, **kwargs):  # noqa: ANN001, ANN003 - 只做记录
        calls.append({"target": target, **kwargs})

    monkeypatch.setattr(uvicorn, "run", recorder)
    return calls


@pytest.mark.parametrize("argv", [["--help"], ["-h"]])
def test_帮助打印用法且绝不起服务(argv, 起服务调用记录, capsys):
    assert server.main(argv) == 0
    assert 起服务调用记录 == []
    out = capsys.readouterr().out
    assert "--host" in out and "--port" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["--port", "abc"],  # 曾经是裸 ValueError + traceback，不是人话
        ["--bogus", "1"],  # 曾经被静默忽略，然后照常起服务
        ["--host"],  # 曾经静默用默认值，然后照常起服务
    ],
)
def test_参数有问题时返回_2_而不是照常起服务(argv, 起服务调用记录, capsys):
    assert server.main(argv) == 2
    assert 起服务调用记录 == []
    assert capsys.readouterr().err  # argparse 的报错走 stderr


def test_默认值与改用_argparse_之前逐位相同(起服务调用记录):
    """空值即不变：不带参数仍是 127.0.0.1:8000，带了就用带的。"""
    assert server.main([]) == 0
    assert server.main(["--host", "0.0.0.0", "--port", "9000"]) == 0
    assert [(call["host"], call["port"]) for call in 起服务调用记录] == [
        ("127.0.0.1", 8000),
        ("0.0.0.0", 9000),
    ]
    assert 起服务调用记录[0]["target"] is server.app
