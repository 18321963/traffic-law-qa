from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any

from .. import config
from .tracer import Recorder

__all__ = ["LangfuseTracer", "from_env"]

_MAX_CHARS = 400
_MAX_ITEMS = 20
_MAX_DEPTH = 6


def _trim(value: Any, *, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return "…（层级太深，略）"
    if isinstance(value, str):
        if len(value) <= _MAX_CHARS:
            return value
        return f"{value[:_MAX_CHARS]}…（共 {len(value)} 字）"
    if isinstance(value, dict):
        return {str(key): _trim(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        head = [_trim(item, depth=depth + 1) for item in value[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            head.append(f"…还有 {len(value) - _MAX_ITEMS} 条")
        return head
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _trim(to_dict(), depth=depth + 1)
    return value


class _RunState(threading.local):

    root: Any = None


class _LangfuseSpan:

    __slots__ = (
        "_tracer",
        "_name",
        "_as_type",
        "_fields",
        "_is_root",
        "_anchor",
        "_timing",
        "_stack",
        "_obs",
    )

    def __init__(
        self,
        tracer: "LangfuseTracer",
        name: str,
        as_type: str,
        *,
        fields: dict | None = None,
        timing: Any = None,
        is_root: bool = False,
        anchor: bool = False,
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._as_type = as_type
        self._fields = fields or {}
        self._is_root = is_root
        self._anchor = anchor
        self._timing = timing
        self._stack = None
        self._obs = None

    def __enter__(self) -> "_LangfuseSpan":
        stack = contextlib.ExitStack()
        self._stack = stack
        if self._timing is not None:
            stack.enter_context(self._timing)
        cm = self._tracer._open(
            self._name,
            self._as_type,
            {key: _trim(value) for key, value in self._fields.items()},
            anchor=self._anchor,
        )
        if cm is not None:
            try:
                self._obs = stack.enter_context(cm)
            except Exception as exc:  # noqa: BLE001
                self._tracer._warn(f"开观察失败（{self._name}）：{exc}")
        if self._is_root and self._obs is not None:
            self._tracer._set_root(self._obs)
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._stack is not None:
            with contextlib.suppress(Exception):
                self._stack.close()
        if self._is_root:
            self._tracer._close_root(self._obs)
        return False

    def update(self, **fields: Any) -> None:
        if self._obs is not None:
            self._tracer._update(self._obs, {key: _trim(value) for key, value in fields.items()})

    def record(self, state: Any, update: Any) -> None:
        fields: dict[str, Any] = {"input": state, "output": update}
        if self._is_root:
            question = (state or {}).get("question") or ""
            fields["name"] = f"invoke · {question[:60]}"
        self.update(**fields)


class LangfuseTracer(Recorder):

    def __init__(self, cfg: config.LangfuseConfig | None = None, *, client: Any = None) -> None:
        super().__init__()
        self.cfg = cfg or config.langfuse_config()
        self._client = client if client is not None else self._connect()
        self._run = _RunState()
        self._last_root: Any = None
        self._warned: set[str] = set()

    def _connect(self) -> Any:
        from langfuse import Langfuse

        return Langfuse(
            public_key=self.cfg.public_key,
            secret_key=self.cfg.secret_key,
            host=self.cfg.host,
            tracing_enabled=True,
        )

    def _warn(self, message: str) -> None:
        if message in self._warned:
            return
        self._warned.add(message)
        print(f"[langfuse] {message}", file=sys.stderr)

    def _set_root(self, observation: Any) -> None:
        self._run.root = observation
        if observation is not None:
            self._last_root = observation

    def _close_root(self, observation: Any) -> None:
        self._run.root = None
        if self._last_root is observation:
            self._last_root = None

    def _open(self, name: str, as_type: str, fields: dict, *, anchor: bool) -> Any:
        if self._client is None:
            return None
        try:
            return self._client.start_as_current_observation(
                name=name,
                as_type=as_type,
                trace_context=self._trace_context() if anchor else None,
                **fields,
            )
        except Exception as exc:  # noqa: BLE001
            self._warn(f"建观察失败（{name}）：{exc}")
            return None

    def _update(self, observation: Any, fields: dict) -> None:
        try:
            observation.update(**fields)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"更新观察失败：{exc}")

    def _trace_context(self) -> dict | None:
        root = self._run.root or self._last_root
        if root is None:
            return None
        return {"trace_id": root.trace_id, "parent_span_id": root.id}

    def span(self, name: str, *, root: bool = False) -> _LangfuseSpan:
        if root:
            self._run.root = None
        return _LangfuseSpan(
            self, name, "span", timing=super().span(name, root=root), is_root=root, anchor=True
        )

    def observation(self, name: str, as_type: str = "span", **fields: Any) -> _LangfuseSpan:
        return _LangfuseSpan(self, name, as_type, fields=fields)

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            self._warn(f"flush 失败：{exc}")


def from_env() -> LangfuseTracer | None:
    cfg = config.langfuse_config()
    if not cfg.ready:
        return None
    try:
        tracer = LangfuseTracer(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f'[langfuse] 观测未开启：{exc}（装法：pip install -e ".[langfuse]"）', file=sys.stderr)
        return None
    print(f"[langfuse] 观测已开启：{cfg.host}（key 不打印）", file=sys.stderr)
    return tracer
