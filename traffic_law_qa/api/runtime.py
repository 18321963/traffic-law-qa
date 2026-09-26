from __future__ import annotations

import importlib.util
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..app.rag import LegalRAG
from ..contracts.errors import QaError
from ..contracts.reports import ReadyState

if TYPE_CHECKING:
    from ..agents.graph import AgentRunner

__all__ = ["Runtime", "agent_runner", "boot_runtime"]


@dataclass
class Runtime:

    ready: ReadyState
    rag: LegalRAG
    boot_ms: float
    dense_live: bool = False
    agent_extra: bool = False
    agent: "AgentRunner | None" = None


def boot_runtime() -> Runtime:
    from .. import container

    started = time.perf_counter()
    ready = container.readiness()
    rag = container.build_rag()
    warm_ms = rag.warm()
    boot_ms = (time.perf_counter() - started) * 1000
    print(f"[serve] 装配完成（{boot_ms / 1000:.2f}s）：{ready.describe()}")
    warm_note = f"{warm_ms:.0f}ms" if warm_ms is not None else "未执行（没有嵌入权重或加载失败，检索只走 BM25）"
    print(f"[serve] 嵌入 + 重排预热 {warm_note}")
    dense_live = ready.dense and warm_ms is not None
    return Runtime(
        ready=ready, rag=rag, boot_ms=boot_ms, dense_live=dense_live, agent_extra=_agent_extra()
    )


def _agent_extra() -> bool:
    try:
        return importlib.util.find_spec("langgraph") is not None
    except (ImportError, ValueError):
        return False


_AGENT_LOCK = threading.Lock()


def agent_runner(rt: Runtime) -> "AgentRunner":
    if rt.agent is not None:
        return rt.agent
    with _AGENT_LOCK:
        if rt.agent is not None:
            return rt.agent
        if not rt.agent_extra:
            raise QaError('未安装 langgraph：mode="agent" 需要 pip install -e ".[agent]"')
        if not rt.rag.llm_ready:
            raise QaError("未配置 LLM_API_KEY：agent 这条路要走模型（只检索请用 mode=search）")
        from .. import container

        runner = container.build_agent_runner(rt.rag)
        runner.graph()
        rt.agent = runner
        return runner
