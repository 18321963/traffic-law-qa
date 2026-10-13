from __future__ import annotations

import sqlite3
import sys
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

from ..storage import open_sessions_db

__all__ = ["open_sessions"]


def _prune_schema_ok(conn: sqlite3.Connection) -> bool:
    required = {"thread_id", "checkpoint_ns", "checkpoint_id"}
    try:
        for table in ("checkpoints", "writes"):
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not required <= columns:
                return False
    except sqlite3.Error:
        return False
    return True


class _LockedSaver(BaseCheckpointSaver):

    def __init__(self, inner: SqliteSaver, conn: sqlite3.Connection) -> None:
        self._inner = inner
        self._conn = conn
        self._lock = threading.Lock()
        self._prune_ok = _prune_schema_ok(conn)
        if not self._prune_ok:
            print(
                "[agent] prune disabled: sessions.db schema missing"
                " thread_id/checkpoint_ns/checkpoint_id",
                file=sys.stderr,
            )
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

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        with self._lock:
            if strategy == "delete":
                for thread_id in thread_ids:
                    self._inner.delete_thread(thread_id)
                self._conn.commit()
                return
            if strategy != "keep_latest" or not self._prune_ok:
                return
            for thread_id in thread_ids:
                kept = self._conn.execute(
                    "SELECT checkpoint_ns, MAX(checkpoint_id) FROM checkpoints"
                    " WHERE thread_id = ? GROUP BY checkpoint_ns",
                    (thread_id,),
                ).fetchall()
                for checkpoint_ns, checkpoint_id in kept:
                    self._conn.execute(
                        "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ?"
                        " AND checkpoint_id != ?",
                        (thread_id, checkpoint_ns, checkpoint_id),
                    )
                    self._conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?"
                        " AND checkpoint_id != ?",
                        (thread_id, checkpoint_ns, checkpoint_id),
                    )
            self._conn.commit()

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
    conn = open_sessions_db(path)
    saver = SqliteSaver(conn)
    saver.setup()
    return _LockedSaver(saver, conn)
