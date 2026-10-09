from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone

from rag_contracts import config

from .session import open_sessions

__all__ = ["USAGE", "checkpoint_time", "main", "plan_cleanup"]

USAGE = """会话保留清理（R6）：按保留期删除过期会话线程

判活 = 该 thread 最新 checkpoint 的 checkpoint_id（langgraph uuid6，时间有序）。

dry-run（默认；可在线，agent 容器内）：
    docker compose exec agent python -m agent_service.agents.session_cleanup

--apply（必须停机：SQLite 写者假设 + VACUUM 要独占）：
    docker compose stop agent
    docker compose run --rm agent python -m agent_service.agents.session_cleanup --apply
    docker compose start agent

认得这些选项：
    --db PATH     会话库路径（默认取 AGENT_SESSION_DB）
    --ttl N       保留天数（默认取 AGENT_SESSION_TTL_DAYS；0 = 不清理，直接退出）
    --apply       真删（默认只 dry-run，不动库）

收尾一行机器可读汇总：[session_cleanup] mode=... total=... expired=... rows=... [deleted=...]
"""

EPOCH_100NS = 0x01B21DD213814000
DAY_SECONDS = 86400.0


def checkpoint_time(checkpoint_id: str) -> float:
    node = uuid.UUID(checkpoint_id)
    if node.version != 6:
        raise ValueError(f"checkpoint_id 不是 uuid6：{checkpoint_id}")
    stamp = (node.time_low << 28) | (node.time_mid << 12) | (node.time_hi_version & 0x0FFF)
    return (stamp - EPOCH_100NS) / 1e7


def plan_cleanup(saver, ttl_days: int, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    cutoff = now - ttl_days * DAY_SECONDS
    stats = saver.thread_stats()
    expired = []
    stamps = []
    for thread_id, checkpoint_id, rows in stats:
        stamp = checkpoint_time(checkpoint_id)
        stamps.append(stamp)
        if stamp < cutoff:
            expired.append((thread_id, checkpoint_id, rows))
    return {
        "total": len(stats),
        "expired": expired,
        "rows": sum(rows for _thread_id, _checkpoint_id, rows in expired),
        "oldest": min(stamps) if stamps else None,
        "newest": max(stamps) if stamps else None,
        "cutoff": cutoff,
    }


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _summary(mode: str, info: dict, deleted: int | None = None) -> str:
    line = (
        f"[session_cleanup] mode={mode} total={info['total']} "
        f"expired={len(info['expired'])} rows={info['rows']}"
    )
    if deleted is not None:
        line += f" deleted={deleted}"
    return line


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_service.agents.session_cleanup", add_help=False
    )
    parser.add_argument("--db", default=None, help="会话库路径（默认取 AGENT_SESSION_DB）")
    parser.add_argument("--ttl", type=int, default=None, help="保留天数（默认取 AGENT_SESSION_TTL_DAYS）")
    parser.add_argument("--apply", action="store_true", help="执行删除（必须停机跑；默认只 dry-run）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return 0

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    cfg = config.agent_config()
    db = options.db or cfg.session_db
    ttl_days = cfg.session_ttl_days if options.ttl is None else options.ttl

    if ttl_days <= 0:
        print(f"[session_cleanup] mode=off ttl={ttl_days}")
        return 0

    saver = open_sessions(db)
    try:
        info = plan_cleanup(saver, ttl_days)
        print(f"线程总数：{info['total']}｜可删：{len(info['expired'])}｜预计行数：{info['rows']}")
        if info["oldest"] is not None:
            print(f"最老：{_iso(info['oldest'])}｜最新：{_iso(info['newest'])}")
        if not options.apply:
            print(_summary("dry-run", info))
            return 0
        for thread_id, _checkpoint_id, _rows in info["expired"]:
            saver.delete_thread(thread_id)
        saver.vacuum()
        deleted = info["total"] - len(saver.thread_stats())
        print(_summary("apply", info, deleted=deleted))
        return 0
    finally:
        saver.close()


if __name__ == "__main__":
    raise SystemExit(main())
