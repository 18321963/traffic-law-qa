from __future__ import annotations

import json

from .schemas import (
    MATERIAL_TOP_K_MAX,
    MATERIAL_TOP_K_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    WEB_COUNT_MAX,
    WEB_COUNT_MIN,
)


def _load(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        raise ValueError("arguments 为空")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"arguments 应是 JSON 对象，实际是 {type(data).__name__}")
    return data


def _clamp(data: dict, key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def parse_tool_arguments(
    raw: str, *, default_top_k: int
) -> tuple[str, str | None, int]:
    data = _load(raw)
    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("缺少 query")
    law_name = str(data.get("law_name") or "").strip() or None
    top_k = _clamp(data, "top_k", default_top_k, TOP_K_MIN, TOP_K_MAX)
    return query, law_name, top_k


def parse_article_arguments(raw: str) -> tuple[str, str | None]:
    data = _load(raw)
    article_no = str(data.get("article_no") or "").strip()
    if not article_no:
        raise ValueError("缺少 article_no")
    return article_no, str(data.get("law_name") or "").strip() or None


def parse_web_arguments(raw: str, *, default_count: int) -> tuple[str, int]:
    data = _load(raw)
    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("缺少 query")
    count = _clamp(data, "count", default_count, WEB_COUNT_MIN, WEB_COUNT_MAX)
    return query, count


def parse_material_arguments(raw: str, *, default_top_k: int) -> tuple[str, int]:
    data = _load(raw)
    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("缺少 query")
    top_k = _clamp(data, "top_k", default_top_k, MATERIAL_TOP_K_MIN, MATERIAL_TOP_K_MAX)
    return query, top_k


def tool_message(call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}
