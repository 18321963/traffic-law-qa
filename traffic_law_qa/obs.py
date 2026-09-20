"""可选的耗时观测 —— **基类本身就是空实现**。

    with tracer.span("node.reflect"):
        ...

`Tracer()` 是所有方法都 no-op 的那一个，也是默认值：不装观测时每个 span 只是两次空调用，
**行为与加这个文件之前逐位相同**（本仓一贯的判据，同 `EMBED_QUERY_PREFIX` 空串、
`retrievable` 缺省 True、`next_query` 空串）。

要真记录就给一个 `Recorder`，它把 span 收进列表。**只有这两个实现，没有 exporter、
不接 LangFuse、不接 OpenTelemetry** —— 这里要的是一个有边界的钩子，不是一套观测体系。

要接真后端时，继承 `Tracer` 覆写 `span()` 返回自己的 span 对象即可 —— 调用方一行都不用改。
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["Tracer", "Recorder", "traced"]


class Tracer:
    """空实现。`span()` 返回自己，于是 `with` 与 `set` 全都落到这两个 no-op 上。"""

    def span(self, name: str) -> "Tracer":
        return self

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
        # 异常也记：崩在哪个节点比耗时更有用。**不吞异常** —— 返回 False 让它继续往上抛。
        self._sink.append({"name": self.name, "seconds": time.perf_counter() - self._start})
        return False


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

    包装体只做「进 span、调用、出 span」，不改返回值 —— 所以空 `Tracer` 下与直接
    `add_node(name, node)` 等价。
    """

    def wrapped(state):
        with tracer.span(name):
            return node(state)

    return wrapped
