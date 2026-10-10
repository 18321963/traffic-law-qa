from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

__all__ = [
    "DB_FAIL_PREFIX",
    "day_key",
    "open_sessions_db",
    "open_usage_db",
    "record_usage",
    "used_today",
]

DB_FAIL_PREFIX = "[usage.db] 记账失败"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS usage_daily("
    " day TEXT PRIMARY KEY,"
    " tokens INTEGER NOT NULL DEFAULT 0,"
    " calls INTEGER NOT NULL DEFAULT 0)"
)


def day_key(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() if now is None else now))


def open_usage_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=3.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute(_SCHEMA)
    return conn


def open_sessions_db(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(path, check_same_thread=False)


def used_today(path: Path, *, now: float | None = None) -> tuple[int, int]:
    conn = open_usage_db(path)
    try:
        row = conn.execute(
            "SELECT tokens, calls FROM usage_daily WHERE day=?", (day_key(now),)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def record_usage(tokens: int, calls: int, path: Path, *, now: float | None = None) -> bool:
    if tokens <= 0 and calls <= 0:
        return True
    try:
        conn = open_usage_db(path)
        try:
            with conn:
                conn.execute(
                    "INSERT INTO usage_daily(day, tokens, calls) VALUES(?, ?, ?)"
                    " ON CONFLICT(day) DO UPDATE SET"
                    " tokens=tokens+excluded.tokens, calls=calls+excluded.calls",
                    (day_key(now), int(tokens), int(calls)),
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"{DB_FAIL_PREFIX}：{type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    return True
