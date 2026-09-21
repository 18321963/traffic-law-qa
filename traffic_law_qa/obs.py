"""可选的耗时观测 —— **基类本身就是空实现**。

    with tracer.span("node.reflect"):
        ...

`Tracer()` 是所有方法都 no-op 的那一个，也是默认值：不装观测时每个 span 只是两次空调用，
**行为与加这个文件之前逐位相同**（本仓一贯的判据，同 `EMBED_QUERY_PREFIX` 空串、
`retrievable` 缺省 True、`next_query` 空串）。

要真记录就给一个 `Recorder`（把 span 收进列表）或 `agent/langfuse_tracer.LangfuseTracer`
（把 span 与节点状态送到 Langfuse 云端）。**本文件自己零依赖、不 import 任何后端** ——
这里要的是一个有边界的钩子，不是一套观测体系；真后端在它自己那个文件里，且是可选的 extra。

钩子有三个，都是「基类空实现 + 子类覆写」：`span()` 计时、`observation()` 记一次 LLM 调用
或检索、`record()` 收节点的入参与出参。要接别的后端，继承 `Tracer` 覆写它们即可 ——
调用方一行都不用改。
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["Tracer", "Recorder", "traced"]


class Tracer:
    """空实现。三个钩子都返回自己，于是 `with` 与 `update` 全都落到这两个 no-op 上。"""

    def span(self, name: str) -> "Tracer":
        return self

    def observation(self, name: str, as_type: str = "span", **fields: Any) -> "Tracer":
        """一次模型调用（`as_type="generation"`）或一次检索（`"retriever"`）。

        `fields` 是**开观察时就已知**的东西（模型名、查询词）；跑完才知道的（输出、token）
        由 `update()` 补。名字与取值全由调用方定 —— 本文件不认识 state，也不认识 LLM。
        """
        return self

    def record(self, state: Any, update: Any) -> None:
        """节点跑完：把入参 state 与它返回的增量交给 span。只有真后端用得着。"""

    def update(self, **fields: Any) -> None:
        """空实现下 `observation()` 返回的正是自己，所以这个 no-op 必须存在。"""

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _Span:
    """一个计时中的 span。`__exit__` 把结果交给 sink（`Recorder.spans`）。"""

    __slots__ = ("name", "_sink", "_start")

    def __init__(self, name: str, sink: list[dict]) -> None:
        self.name = name
        self._sink = sink
        self._start = 0.0

    def __enter__(self) -> "_Span":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._sink.append({"name": self.name, "seconds": time.perf_counter() - self._start})
        return False

    def record(self, state: Any, update: Any) -> None:
        """`Recorder` 只要耗时，节点状态与它无关。"""


class Recorder(Tracer):
    def __init__(self) -> None:
        self.spans: list[dict] = []

    def span(self, name: str) -> _Span:
        return _Span(name, self.spans)

    def summary(self) -> list[dict]:
        """按名字聚合：调用次数 + 总耗时 + 单次均值。**按总耗时降序** —— 先看大头。

        均值而不是中位数：这里的 span 数只有几十个，排序看的是「哪个节点吃掉了大头」。
        """
        buckets: dict[str, list[float]] = {}
        for span in self.spans:
            buckets.setdefault(span["name"], []).append(span["seconds"])
        rows = [
            {
                "name": name,
                "count": len(seconds),
                "total": sum(seconds),
                "mean": sum(seconds) / len(seconds),
            }
            for name, seconds in buckets.items()
        ]
        rows.sort(key=lambda row: row["total"], reverse=True)
        return rows


def traced(tracer: Tracer, name: str, node: Callable) -> Callable:
    """把 langgraph 节点（`state -> dict`）包一层 span。

    包装体只做「进 span、调用、`record`、出 span」，不改返回值 —— 所以空 `Tracer` 下与
    直接 `add_node(name, node)` 等价（`record` 与 `span` 都是空实现）。

    这里**是唯一同时握着入参 state 和出参增量**的地方：节点自己只关心这两者之一，
    所以想记「这个节点吃了什么、吐了什么」只能在这一层。怎么切、切多细由 span 自己定。
    """

    def wrapped(state):
        with tracer.span(name) as span:
            update = node(state)
            span.record(state, update)
            return update

    return wrapped
