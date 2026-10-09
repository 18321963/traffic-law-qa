from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
)
from langgraph.checkpoint.sqlite import SqliteSaver

__all__ = ["open_sessions"]


class _LockedSaver(BaseCheckpointSaver):

    def __init__(self, inner: SqliteSaver, conn: sqlite3.Connection) -> None:
        self._inner = inner
        self._conn = conn
        self._lock = threading.Lock()
        super().__init__(serde=inner.serde)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        with self._lock:
            return self._inner.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        with self._lock:
            return self._inner.list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._lock:
            return self._inner.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._lock:
            self._inner.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._inner.delete_thread(thread_id)

    def thread_stats(self) -> list[tuple[str, str, int]]:
        with self._lock:
            latest = self._conn.execute(
                "SELECT thread_id, MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id"
            ).fetchall()
            checkpoints = dict(
                self._conn.execute(
                    "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id"
                ).fetchall()
            )
            writes = dict(
                self._conn.execute("SELECT thread_id, COUNT(*) FROM writes GROUP BY thread_id").fetchall()
            )
        return [
            (
                str(thread_id),
                str(checkpoint_id),
                int(checkpoints.get(thread_id, 0)) + int(writes.get(thread_id, 0)),
            )
            for thread_id, checkpoint_id in latest
        ]

    def vacuum(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.execute("VACUUM")

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self._inner.get_next_version(current, channel)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def open_sessions(path: str) -> _LockedSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return _LockedSaver(saver, conn)
