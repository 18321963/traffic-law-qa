"""pipeline 命令行的入口测试 —— 全部离线。

存在的理由只有一个：`python -m traffic_law_rag.pipeline --help` 曾经不是「查用法」，
而是**一次完整的重新切块 + 重新 embedding**。原因是命令推断写成
`args[0] if args and not args[0].startswith("--") else "build"`，`--help` 以 `--` 开头，
于是落进了 `else "build"`。查一次用法花掉一次 embedding 钱（约 42 秒、812 行）。

所以这里断言的不是「有没有帮助文本」，而是**帮助路径绝不能构造 RagPipeline** ——
构造它就会连 Milvus、读切块产物，进一步还会 embedding。用 monkeypatch 把它钉死。
"""

from __future__ import annotations

import pytest

from traffic_law_rag import pipeline


@pytest.fixture
def 禁止构造管道(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):  # noqa: ANN002, ANN003 - 只做哨兵
        raise AssertionError("--help 不该构造 RagPipeline（会连 Milvus / 触发重建）")

    monkeypatch.setattr(pipeline, "RagPipeline", boom)


@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["help"]])
def test_帮助直接打印用法且不构造管道(argv, 禁止构造管道, capsys):
    assert pipeline.main(argv) == 0
    out = capsys.readouterr().out
    assert "python -m traffic_law_rag.pipeline build" in out
    assert "layers | status" in out


def test_帮助文本就是模块自述():
    assert pipeline.USAGE is pipeline.__doc__
    assert "build" in pipeline.USAGE and "layers" in pipeline.USAGE
