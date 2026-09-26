from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["Tracer", "Recorder", "traced"]


class Tracer:

    def span(self, name: str, *, root: bool = False) -> "Tracer":
        return self

    def observation(self, name: str, as_type: str = "span", **fields: Any) -> "Tracer":
        return self

    def record(self, state: Any, update: Any) -> None:
        pass

    def update(self, **fields: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _Span:

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
        pass


class Recorder(Tracer):
    def __init__(self) -> None:
        self.spans: list[dict] = []

    def span(self, name: str, *, root: bool = False) -> _Span:
        return _Span(name, self.spans)

    def summary(self) -> list[dict]:
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

    def wrapped(state):
        with tracer.span(name) as span:
            update = node(state)
            span.record(state, update)
            return update

    return wrapped
