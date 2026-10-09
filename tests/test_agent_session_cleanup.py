from __future__ import annotations

import re
import time
import uuid

import pytest

pytest.importorskip("langgraph", reason='agent 图要 pip install -e ".[agent]"')

from langgraph.checkpoint.base import empty_checkpoint  # noqa: E402

from agent_service.agents.session import open_sessions  # noqa: E402
from agent_service.agents.session_cleanup import checkpoint_time, main  # noqa: E402

TTL_DAYS = 30
DAY_SECONDS = 86400.0


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _mint_checkpoint_id(unix_seconds: float) -> str:
    stamp = int(unix_seconds * 1e7) + 0x01B21DD213814000
    raw = ((stamp >> 12) & 0xFFFFFFFFFFFF) << 80
    raw |= (stamp & 0x0FFF) << 64
    raw |= 0x8000 << 48
    raw |= 0x0123456789AB
    raw &= ~(0xF000 << 64)
    raw |= 6 << 76
    return str(uuid.UUID(int=raw))


def _open(tmp_path) -> tuple[object, str]:
    db = str(tmp_path / "sessions.db")
    return open_sessions(db), db


def _seed_fresh(saver, thread_id: str) -> None:
    saver.put(_cfg(thread_id), empty_checkpoint(), {"source": "test", "step": 0}, {})


def _seed_expired(saver, thread_id: str, *, with_writes: bool = False) -> None:
    out = saver.put(_cfg(thread_id), empty_checkpoint(), {"source": "test", "step": 0}, {})
    if with_writes:
        saver.put_writes(out, [("plan", ["旧事件"])], "task-1")
    old_id = _mint_checkpoint_id(time.time() - 40 * DAY_SECONDS)
    saver._conn.execute(
        "UPDATE checkpoints SET checkpoint_id = ? WHERE thread_id = ?", (old_id, thread_id)
    )
    saver._conn.execute("UPDATE writes SET checkpoint_id = ? WHERE thread_id = ?", (old_id, thread_id))
    saver._conn.commit()


def _thread_count(saver) -> int:
    row = saver._conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()
    return int(row[0])


def _total_rows(saver) -> int:
    row = saver._conn.execute(
        "SELECT (SELECT COUNT(*) FROM checkpoints) + (SELECT COUNT(*) FROM writes)"
    ).fetchone()
    return int(row[0])


def _rows_for(saver, thread_id: str) -> int:
    row = saver._conn.execute(
        "SELECT (SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?)"
        " + (SELECT COUNT(*) FROM writes WHERE thread_id = ?)",
        (thread_id, thread_id),
    ).fetchone()
    return int(row[0])


def test_checkpoint_time_reads_a_real_checkpoint_id(tmp_path) -> None:
    saver, _db = _open(tmp_path)
    try:
        out = saver.put(_cfg("real-1"), empty_checkpoint(), {"source": "test", "step": 0}, {})
        checkpoint_id = out["configurable"]["checkpoint_id"]
        assert abs(checkpoint_time(checkpoint_id) - time.time()) < 60
    finally:
        saver.close()


def test_checkpoint_time_rejects_a_non_uuid6() -> None:
    with pytest.raises(ValueError):
        checkpoint_time(str(uuid.uuid4()))


def test_dry_run_counts_without_touching_the_database(tmp_path, capsys) -> None:
    saver, db = _open(tmp_path)
    try:
        _seed_fresh(saver, "fresh-1")
        _seed_expired(saver, "stale-1", with_writes=True)
        rows_before = _total_rows(saver)
        assert rows_before == 3

        assert main(["--db", db, "--ttl", str(TTL_DAYS)]) == 0

        out = capsys.readouterr().out
        assert "线程总数：2｜可删：1｜预计行数：2" in out
        assert re.search(r"^\[session_cleanup\] mode=dry-run total=2 expired=1 rows=2$", out, re.M)
        assert _total_rows(saver) == rows_before
        assert _thread_count(saver) == 2
    finally:
        saver.close()


def test_apply_deletes_expired_and_keeps_fresh(tmp_path, capsys) -> None:
    saver, db = _open(tmp_path)
    try:
        _seed_fresh(saver, "fresh-1")
        _seed_expired(saver, "stale-1", with_writes=True)

        assert main(["--db", db, "--ttl", str(TTL_DAYS)]) == 0
        dry = capsys.readouterr().out
        reported = re.search(r"expired=(\d+)", dry)
        assert reported is not None
        threads_before = _thread_count(saver)

        assert main(["--db", db, "--ttl", str(TTL_DAYS), "--apply"]) == 0

        out = capsys.readouterr().out
        assert re.search(r"^\[session_cleanup\] mode=apply total=2 expired=1 rows=2 deleted=1$", out, re.M)
        assert threads_before - _thread_count(saver) == int(reported.group(1))
        assert _rows_for(saver, "stale-1") == 0
        assert _rows_for(saver, "fresh-1") == 1
        assert saver.get_tuple(_cfg("fresh-1")) is not None
        assert saver.get_tuple(_cfg("stale-1")) is None
    finally:
        saver.close()


def test_zero_ttl_exits_before_opening_the_database(tmp_path, capsys) -> None:
    db = tmp_path / "sessions.db"

    assert main(["--db", str(db), "--ttl", "0"]) == 0

    out = capsys.readouterr().out
    assert "[session_cleanup] mode=off ttl=0" in out
    assert not db.exists()


def test_the_ttl_default_comes_from_the_environment(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AGENT_SESSION_TTL_DAYS", "0")
    db = tmp_path / "sessions.db"

    assert main(["--db", str(db)]) == 0

    out = capsys.readouterr().out
    assert "[session_cleanup] mode=off ttl=0" in out
    assert not db.exists()
