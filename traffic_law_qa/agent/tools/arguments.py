"""模型给的 `arguments` 字符串 → 库能用的参数。

两份 schema 各有一个解析器，标准相同：挡下**不能用**的（抛 ValueError，由调用方
翻成一句中文回灌给模型），能用的尽量救回来。`tool_message` 也在这儿 —— 它造的是
把这句中文送回模型的那条消息。
"""

from __future__ import annotations

import json

from .schemas import TOP_K_MAX, TOP_K_MIN


def parse_tool_arguments(
    raw: str, *, default_top_k: int
) -> tuple[str, str | None, int]:
    """解析模型给的 `arguments` 字符串 → (query, law_name, top_k)。

    模型给的东西什么都有可能：空串、非法 JSON、缺 query、top_k=999。
    这里只负责挡下**不能用**的（抛 ValueError，由调用方翻成一句中文回灌给模型），
    能用的就尽量救回来（top_k 夹取、law_name 原样带出交给 resolve_law_id）。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("arguments 为空")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"arguments 应是 JSON 对象，实际是 {type(data).__name__}")

    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("缺少 query")

    raw_top_k = data.get("top_k", default_top_k)
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError):
        top_k = default_top_k
    top_k = max(TOP_K_MIN, min(TOP_K_MAX, top_k))

    law_name = str(data.get("law_name") or "").strip() or None
    return query, law_name, top_k


def parse_article_arguments(raw: str) -> tuple[str, str | None]:
    """解析模型给的 get_article `arguments` → (article_no, law_name)。

    与 `parse_tool_arguments` 同一套标准：挡下**不能用**的（抛 ValueError，
    由调用方翻成一句中文回灌给模型），能用的尽量救回来。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("arguments 为空")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"arguments 应是 JSON 对象，实际是 {type(data).__name__}")

    article_no = str(data.get("article_no") or "").strip()
    if not article_no:
        raise ValueError("缺少 article_no")
    return article_no, str(data.get("law_name") or "").strip() or None


def tool_message(call_id: str, text: str) -> dict:
    """OpenAI 线上格式的 tool 消息。

    每个 tool_call 必须恰好配一条，否则下一次请求会被端点以 400 拒绝
    （assistant 的 tool_calls 必须紧跟对应的 tool 消息）。
    """
    return {"role": "tool", "tool_call_id": call_id, "content": text}
