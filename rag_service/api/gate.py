from __future__ import annotations

import threading
from contextlib import contextmanager

from rag_contracts import config

__all__ = [
    "QUEUE_TIMEOUT_DETAIL",
    "RETRY_AFTER_SECONDS",
    "RagQueueTimeout",
    "gate",
]

QUEUE_TIMEOUT_DETAIL = "检索排队超时：处理中的请求已满，请稍后重试"
RETRY_AFTER_SECONDS = "5"

_lock = threading.Lock()
_state: dict = {"limit": -1, "sem": None}


def _semaphore() -> threading.BoundedSemaphore | None:
    limit = config.rag_max_concurrency()
    if limit <= 0:
        return None
    with _lock:
        if _state["limit"] != limit:
            _state["limit"] = limit
            _state["sem"] = threading.BoundedSemaphore(limit)
        return _state["sem"]


class RagQueueTimeout(Exception):
    pass


@contextmanager
def gate():
    sem = _semaphore()
    if sem is None:
        yield True
        return
    if not sem.acquire(timeout=config.rag_queue_timeout()):
        yield False
        return
    try:
        yield True
    finally:
        sem.release()
