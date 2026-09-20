"""`/health` 两个通道字段的离线用例 —— 它必须说「此刻」的实话。

这里钉的是一个**真实发生过**的事故：宿主机 Ollama 没跑时，`channels` 照报
「稠密+BM25」。那个字段当时取自 `index_meta.json` 的建库快照 —— 集合确实是带
稠密建的，于是它没答错「集合支持什么」，却答错了「此刻在走什么」。容器健康检查
打的就是 `/health`，它是唯一被程序消费的出口，却是唯一说反话的那个。

现在两个字段分工，坏掉的那件事第一次变得可诊断：

- `channels`    此刻实际在走的通道（观测值 `dense_live`，随预热与每次检索更新）
- `dense_built` 集合建库时带没带稠密（`ready.dense`，落盘之后不再变）

`纯 BM25` 配 `dense_built: true` 就是「向量端点坏了」；`dense_built: false`
才是「索引本来就只建了 BM25」。只看一个字段这两种情况分不开。

全部离线：不连 Milvus、不发网络请求。
"""

from __future__ import annotations

import pytest

from traffic_law_qa import server
from traffic_law_qa.api import ReadyState


class _假检索:
    """只提供 `ask()` 真正用到的两样：`used_vector` 与 `to_dict()`。"""

    def __init__(self, used_vector: bool) -> None:
        self.used_vector = used_vector

    def to_dict(self) -> dict:
        return {"question": "问题", "used_vector": self.used_vector}


class _假Rag:
    """替掉真 `LegalRAG`。`下次走向量` 可改，用来模拟端点半路回来。

    **替身不许有 `used_vector`。** 真 `LegalRAG` 没有这个属性 —— 它挂在每次检索
    返回的 `RetrievalResult` 上（`contracts.py:411`）。替身凭空多一个同名属性，
    就等于把「从检索结果上取」和「从 rag 上取」变得**逐位相同**：写回源写成
    `rt.rag.used_vector` 时本文件照样全绿，而线上每次 `/qa` 会抛 `AttributeError`。
    所以可变状态用一个生产代码里不存在的名字，别复用 `used_vector`。
    """

    llm_ready = True

    def __init__(self, used_vector: bool) -> None:
        self.下次走向量 = used_vector

    def search(self, question: str, *, top_k: int | None = None) -> _假检索:
        return _假检索(self.下次走向量)


def _就绪(*, dense_built: bool) -> ReadyState:
    """一份形状正确、数值固定的就绪状态；这里只关心 `dense` 那一位。"""
    return ReadyState(
        action="reuse",
        reason="docx、本地产物与集合三者一致",
        rows=812,
        laws=6,
        articles=508,
        parents=508,
        dense=dense_built,
        milvus="2.6.24",
    )


def _装上(monkeypatch, *, dense_built: bool, dense_live: bool) -> server.Runtime:
    rt = server.Runtime(
        ready=_就绪(dense_built=dense_built),
        rag=_假Rag(used_vector=dense_live),
        boot_ms=1.0,
        dense_live=dense_live,
    )
    monkeypatch.setattr(server.app.state, "rt", rt, raising=False)
    return rt


@pytest.fixture
def 端点挂了(monkeypatch) -> server.Runtime:
    """这次事故的现场：集合带稠密建过，但此刻稠密通道没在工作。"""
    return _装上(monkeypatch, dense_built=True, dense_live=False)


def test_端点挂了时_channels_报纯BM25(端点挂了):
    """回归测试，就是这个 bug 本身。

    改前这里断言会失败：`channels` 报「稠密+BM25」，因为读的是建库快照。
    而同一份响应里的 `dense_built` 是 true —— 两者一比就知道是端点坏了，
    不是索引本来就是纯 BM25。
    """
    payload = server.health()

    assert payload["channels"] == "纯 BM25"
    assert payload["dense_built"] is True


def test_通道正常时_channels_报稠密(monkeypatch):
    _装上(monkeypatch, dense_built=True, dense_live=True)

    payload = server.health()

    assert payload["channels"] == "稠密+BM25"
    assert payload["dense_built"] is True


def test_ask_把本次的_used_vector_写回运行时(端点挂了):
    """状态是活的，不是启动时冻住的。

    端点半路恢复时，下一次检索就该把它翻过来 —— 这条钉住写回那一步，
    没有它 `/health` 会一直停在启动那一刻的答案上。
    """
    assert 端点挂了.dense_live is False

    端点挂了.rag.下次走向量 = True
    server.ask(server.QaRequest(question="醉驾怎么处罚", mode="search"))

    assert 端点挂了.dense_live is True


def test_假Rag_不许有真类没有的_used_vector():
    """**这条是回归**：替身凭空多一个属性，就能让写错的取法蒙混过关。

    `_假Rag` 曾经自带 `used_vector`（取值还和检索结果一致），于是把写回源写成
    `rt.rag.used_vector` —— 线上每次 `/qa` 抛 `AttributeError`、本地三条测试全绿。
    真 `LegalRAG` 没有这个属性，替身就不许有。谁把它加回来，这条当场红。
    """
    assert not hasattr(_假Rag(used_vector=True), "used_vector")
