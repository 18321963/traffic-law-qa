"""把 agent 的节点状态、模型调用与检索送到 Langfuse 云端（可选，`--langfuse` 才走到这里）。

`obs.Tracer` 的空实现保住了「不观测时逐位不变」，这个文件是它的第一个真后端：
继承 `Recorder`（于是 `--timing` 与云端可以同时开），覆写 `span()` / `observation()` / `record()`。

四件事值得先知道：

**父链显式钉，不靠 OTEL 的隐式上下文。** 根 span（`AgentRunner.invoke` 那一层）建出一条新
trace 并记下 `trace_id` 与它自己的 span id，之后每个节点 span 都带
`trace_context={"trace_id": …, "parent_span_id": …}` 指回根。langgraph 会不会换线程跑节点
不在我们掌控内，隐式上下文一断，四个节点就各自变成一条孤立 trace —— 「看着有数据、其实
全散了」是最难发现的那种失败，宁可多传两个 id。**节点内部**再开的 generation / retriever
反而走隐式上下文：那已经是节点自己的调用栈，稳。

**只有一条削数据的规则**（`_trim`）：字符串截 400 字、列表留 20 条、dataclass 先 `to_dict()`。
不做按节点定制的字段表 —— 那种表没人看得住，state 加一个字段就漏一处。一条规则的效果是
「结构不丢、尺寸可控」：节点 span 的 input 是入参 state、output 是该节点返回的增量。

**观测失败不许弄坏一次提问。** 每个 SDK 调用都包了 try/except，失败只在 stderr 说一句
（同一句只说一次，免得刷屏），流程照跑。

**送出去的是题目原文、提示词、法条片段与模型输出**，这些会到 Langfuse 云端；不配 key 时
一个字节都不外发。两个 key 只进 SDK 的构造函数，既不进 trace、也绝不打印。
"""

from __future__ import annotations

import contextlib
import sys
from typing import Any

from .. import config
from ..obs import Recorder

__all__ = ["LangfuseTracer"]

_MAX_CHARS = 400
_MAX_ITEMS = 20
_MAX_DEPTH = 6


def _trim(value: Any, *, depth: int = 0) -> Any:
    """把 state 削到「云端看得懂、又不至于爆掉」的尺寸。**全套只有这一条规则。**

    一次提问的终态里有 N 轮 messages（每轮都带几百字的法条片段）、search_log 与整篇答案，
    原样送上去轻松几百 KB。削的尺度按「人在面板上愿不愿意滚」定，不按字节预算定。
    """
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


class _LangfuseSpan:
    """一个观察 + （节点才有）一笔本地计时。

    进的时候两样都进，出的时候两样都出 —— `--timing` 记的是 `invoke` / `node.*` 这几个
    老名字，云端那份可以给更长的名字，两边互不影响。
    """

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
            except Exception as exc:  # noqa: BLE001 - 观测失败不许弄坏一次提问
                self._tracer._warn(f"开观察失败（{self._name}）：{exc}")
        if self._is_root and self._obs is not None:
            self._tracer._set_root(self._obs)
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._stack is not None:
            with contextlib.suppress(Exception):
                self._stack.close()
        return False

    def update(self, **fields: Any) -> None:
        """喂进来的东西一律先过 `_trim` —— **切尺寸只有这一处**，调用方给原始数据就行。"""
        if self._obs is not None:
            self._tracer._update(self._obs, {key: _trim(value) for key, value in fields.items()})

    def record(self, state: Any, update: Any) -> None:
        """节点跑完：入参 state 与它吐出来的增量一次送出去。

        根观察在这里改名 —— 开 span 时还不知道问题是什么（问题随 state 一起来），
        而云端列表里十条 trace 都叫 `invoke` 是没法看的。改名放在这里而不是改调用方
        传的 span 名，是因为那个名字同时是 `--timing` 的行名，动不得。

        改名与 I/O **合成一次 update**：分两次写的话，第二次要是没赶上，云端就留下一条
        有名字没内容的 trace。
        """
        fields: dict[str, Any] = {"input": state, "output": update}
        if self._is_root:
            question = (state or {}).get("question") or ""
            fields["name"] = f"invoke · {question[:60]}"
        self.update(**fields)


class LangfuseTracer(Recorder):
    """`Recorder` + 云端上报。`client=` 可注入替身，离线验收因此不需要装 langfuse。"""

    def __init__(self, cfg: config.LangfuseConfig | None = None, *, client: Any = None) -> None:
        super().__init__()
        self.cfg = cfg or config.langfuse_config()
        self._client = client if client is not None else self._connect()
        self._root: Any = None
        self._opened = False
        self._warned: set[str] = set()

    def _connect(self) -> Any:
        """延迟 import：不启用时 `langfuse` 这个包根本不必装（同 `llm.py` 的 `from openai import OpenAI`）。

        版本下限钉 4.15：v3 那一套 `start_as_current_span` / `set_trace_io` 在 4 里已经换成
        `start_as_current_observation`，且 `set_trace_io` 已标 deprecated。
        """
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
        self._root = observation

    def _open(self, name: str, as_type: str, fields: dict, *, anchor: bool) -> Any:
        """`anchor=True` 才显式挂到根上（节点 span 走这条）；节点内部那几层传 False，
        由 SDK 按当前上下文挂 —— 挂到谁头上取决于「谁在 with 里」，显式钉反而会把
        generation / retriever 从它们所属的节点下面拽出来，拍成根的兄弟。"""
        if self._client is None:
            return None
        try:
            return self._client.start_as_current_observation(
                name=name,
                as_type=as_type,
                trace_context=self._trace_context() if anchor else None,
                **fields,
            )
        except Exception as exc:  # noqa: BLE001 - 观测失败不许弄坏一次提问
            self._warn(f"建观察失败（{name}）：{exc}")
            return None

    def _update(self, observation: Any, fields: dict) -> None:
        try:
            observation.update(**fields)
        except Exception as exc:  # noqa: BLE001 - 观测失败不许弄坏一次提问
            self._warn(f"更新观察失败：{exc}")

    def _trace_context(self) -> dict | None:
        """挂到根下面。根自己没开出来时返回 None —— 那几段会各自成一条 trace，只在 stderr 提醒过。"""
        root = self._root
        if root is None:
            return None
        return {"trace_id": root.trace_id, "parent_span_id": root.id}

    def span(self, name: str) -> _LangfuseSpan:
        """节点 span。**只有第一个 span 是根** —— 这个判断放在这里而不是「观察开成功了没」，
        因为 SDK 那边失败时也不该让第二个节点误以为自己是根，那会裂成六条 trace。"""
        is_root = not self._opened
        self._opened = True
        return _LangfuseSpan(
            self, name, "span", timing=super().span(name), is_root=is_root, anchor=True
        )

    def observation(self, name: str, as_type: str = "span", **fields: Any) -> _LangfuseSpan:
        return _LangfuseSpan(self, name, as_type, fields=fields)

    def flush(self) -> None:
        """短命脚本必须收尾：SDK 在后台批量导出，不 flush 就退出会丢最后几条。"""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001 - 观测失败不许弄坏一次提问
            self._warn(f"flush 失败：{exc}")
