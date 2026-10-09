from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

from rag_contracts import config

__all__ = [
    "BLOCKED_DETAIL",
    "DB_FAIL_PREFIX",
    "USAGE",
    "day_key",
    "db_path",
    "exhausted",
    "main",
    "record_state",
    "record_usage",
    "retry_after_seconds",
    "state_tokens",
    "used_today",
]

USAGE = """成本闸（R3）：按 UTC 日累计 LLM token 用量，超额拦问答入口

账本：data/usage.db（WAL），表 usage_daily(day TEXT PRIMARY KEY, tokens INTEGER, calls INTEGER)，
day = UTC 日期；run 出口（正常 / 中断 / 异常三路）取状态快照 usage 与 usage_extra 两个通道全部事件求和记账
（规划轮在 usage；地区识别 / 复核 / 答案生成在 usage_extra；
中断轮已记的用量由澄清节点在 resume 续跑时清零，不重复入账）。
写库失败只打 stderr（前缀 [usage.db] 记账失败），不拦请求。

环境变量：AGENT_DAILY_TOKEN_BUDGET（默认 0 = 关；>0 时进入 run 前查当日已用，已达 → 429）。

巡检（打印当日行，读失败非零退出）：
    python -m agent_service.api.budget --show
输出机读汇总：[usage.db] day=... tokens=... calls=... budget=...
"""

BLOCKED_DETAIL = "日额度已用完：预算 {budget} tokens，今日已用 {used} tokens；按 UTC 0 点重置"
DB_FAIL_PREFIX = "[usage.db] 记账失败"
READ_FAIL_PREFIX = "[usage.db] 读取失败"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS usage_daily("
    " day TEXT PRIMARY KEY,"
    " tokens INTEGER NOT NULL DEFAULT 0,"
    " calls INTEGER NOT NULL DEFAULT 0)"
)
_DAY_SECONDS = 86400.0


def db_path() -> Path:
    return Path(config.DATA_DIR) / "usage.db"


def day_key(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() if now is None else now))


def retry_after_seconds(now: float | None = None) -> int:
    stamp = time.time() if now is None else now
    return max(1, int(_DAY_SECONDS - stamp % _DAY_SECONDS))


def _open(path: Path | None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=3.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute(_SCHEMA)
    return conn


def used_today(*, path: Path | None = None, now: float | None = None) -> tuple[int, int]:
    conn = _open(path)
    try:
        row = conn.execute(
            "SELECT tokens, calls FROM usage_daily WHERE day=?", (day_key(now),)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return 0, 0
    return int(row[0] or 0), int(row[1] or 0)


def exhausted(
    budget: int, *, path: Path | None = None, now: float | None = None
) -> tuple[int, int] | None:
    if budget <= 0:
        return None
    try:
        tokens, _calls = used_today(path=path, now=now)
    except Exception as exc:  # noqa: BLE001
        print(f"{READ_FAIL_PREFIX}：{type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if tokens < budget:
        return None
    return budget, tokens


def record_usage(tokens: int, calls: int, *, path: Path | None = None, now: float | None = None) -> bool:
    if tokens <= 0 and calls <= 0:
        return True
    try:
        conn = _open(path)
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


def state_tokens(state) -> tuple[int, int]:
    if not isinstance(state, dict):
        return 0, 0
    rows = [*(state.get("usage") or ()), *(state.get("usage_extra") or ())]
    tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    return tokens, len(rows)


def record_state(state, *, path: Path | None = None, now: float | None = None) -> bool:
    tokens, calls = state_tokens(state)
    return record_usage(tokens, calls, path=path, now=now)


def _summary(tokens: int, calls: int, budget: int) -> str:
    line = f"[usage.db] day={day_key()} tokens={tokens} calls={calls} budget={budget}"
    if budget > 0:
        line += f" remaining={max(0, budget - tokens)}"
    return line


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return 0

    if args != ["--show"]:
        print(USAGE, file=sys.stderr)
        return 2

    budget = config.agent_config().daily_token_budget
    try:
        tokens, calls = used_today()
    except Exception as exc:  # noqa: BLE001
        print(f"{READ_FAIL_PREFIX}：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(_summary(tokens, calls, budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
