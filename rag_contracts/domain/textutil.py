from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["bigrams", "cn_to_int", "sha1_of"]

CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def cn_to_int(cn: str) -> int:
    if not cn:
        raise ValueError("空的中文数字")
    section = number = 0
    for ch in cn:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        else:
            raise ValueError(f"无法解析的中文数字：{cn}")
    total = section + number
    if total == 0 and cn != "零":
        raise ValueError(f"无法解析的中文数字：{cn}")
    return total


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in text if ch.strip())
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} or {cleaned}
