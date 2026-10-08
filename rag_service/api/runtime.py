from __future__ import annotations

import time
from dataclasses import dataclass

from rag_contracts.domain.reports import ReadyState

from ..query.rag import LegalRAG

__all__ = ["Runtime", "boot_runtime"]


@dataclass
class Runtime:

    ready: ReadyState
    rag: LegalRAG
    boot_ms: float
    dense_live: bool = False


def boot_runtime(*, rebuild: bool = False) -> Runtime:
    from .. import container

    started = time.perf_counter()
    ready = container.readiness(rebuild=rebuild)
    rag = container.build_rag()
    warm_ms = rag.warm()
    boot_ms = (time.perf_counter() - started) * 1000
    print(f"[serve] 装配完成（{boot_ms / 1000:.2f}s）：{ready.describe()}")
    warm_note = f"{warm_ms:.0f}ms" if warm_ms is not None else "未执行（没有嵌入权重或加载失败，检索只走 BM25）"
    print(f"[serve] 嵌入 + 重排预热 {warm_note}")
    dense_live = ready.dense and warm_ms is not None
    return Runtime(ready=ready, rag=rag, boot_ms=boot_ms, dense_live=dense_live)
