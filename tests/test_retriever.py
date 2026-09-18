"""检索器的离线可测部分。不连 Milvus、不发网络请求。

重点覆盖 `warm()` —— 它存在的原因是一个实测出来的现象：embedding 端点的首次调用
要建 TCP+TLS 连接（实测 1.5~3s），之后的调用只要 ~150ms。不预热的话这个开销
会算在第一个用户请求头上。这条断言值得锁住：真去建连的测试没法离线跑，
但「装配时确实调用了一次」是可以离线断言的，而那正是容易被人删掉的一行。
"""

from __future__ import annotations

import pytest

from traffic_law_rag.retriever import HybridRetriever


class _FakeEmbedder:
    """记录调用次数的假 embedder。"""

    def __init__(self, *, exc: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.exc = exc

    def embed_one(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.exc is not None:
            raise self.exc
        return [0.0] * 8


def _retriever(embedder) -> HybridRetriever:
    """warm() 只用 self.embedder，其余部件给最简替身即可。"""
    return HybridRetriever(store=object(), parents={}, chunks={}, embedder=embedder)


def test_预热真的调了一次_embedder():
    embedder = _FakeEmbedder()
    ms = _retriever(embedder).warm()
    assert len(embedder.calls) == 1, "预热必须真的发一次请求，否则连接根本没建起来"
    assert isinstance(ms, float) and ms >= 0


def test_没有_embedder_时返回_None():
    assert _retriever(None).warm() is None


def test_预热失败不抛异常_而是返回_None():
    """端点挂着不是致命错误：检索层本来就会退回 BM25，预热同理。"""
    embedder = _FakeEmbedder(exc=RuntimeError("connection refused"))
    assert _retriever(embedder).warm() is None
    assert len(embedder.calls) == 1, "失败也要先真的试过"


def test_预热失败不影响后续检索的降级判断():
    """warm() 只吞自己的异常，不改变 embedder 的状态。"""
    embedder = _FakeEmbedder(exc=TimeoutError("timeout"))
    retriever = _retriever(embedder)
    assert retriever.warm() is None
    assert retriever.embedder is embedder, "预热不该把 embedder 置空"


@pytest.mark.parametrize("probe", ["预热", "x"])
def test_预热文本可覆盖(probe):
    embedder = _FakeEmbedder()
    _retriever(embedder).warm(probe=probe)
    assert embedder.calls == [probe]
