from __future__ import annotations

import hmac
import math
import threading

__all__ = [
    "API_KEY_HEADER",
    "EXEMPT_PATHS",
    "MinuteWindowLimiter",
    "UNAUTHORIZED_DETAIL",
    "keys_match",
    "rate_limit_detail",
]

EXEMPT_PATHS = frozenset({"/health", "/laws"})
API_KEY_HEADER = "X-API-Key"
UNAUTHORIZED_DETAIL = "API key 缺失或无效"


def keys_match(presented: str, keys: tuple[str, ...]) -> bool:
    candidate = presented.encode("utf-8")
    return any(hmac.compare_digest(candidate, key.encode("utf-8")) for key in keys)


def rate_limit_detail(rpm: int) -> str:
    return f"请求过于频繁：每分钟上限 {rpm} 次"


class MinuteWindowLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, tuple[int, int]] = {}

    def hit(self, key: str, rpm: int, now: float) -> int:
        window = int(now // 60) * 60
        with self._lock:
            if len(self._counts) > 1024:
                self._counts = {k: v for k, v in self._counts.items() if v[0] == window}
            start, count = self._counts.get(key, (window, 0))
            if start != window:
                start, count = window, 0
            count += 1
            self._counts[key] = (start, count)
        if count > rpm:
            return max(1, math.ceil(start + 60 - now))
        return 0
