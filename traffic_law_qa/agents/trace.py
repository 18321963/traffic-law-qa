from __future__ import annotations

from .region import REGION_UNKNOWN
from .state import AgentState

__all__ = ["render_trace", "render_timing"]

_COLORS = {"yellow": "\033[33m", "dim": "\033[2m"}
_RESET = "\033[0m"


def _paint(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_COLORS[color]}{text}{_RESET}"


def render_trace(state: AgentState, *, color: bool = False) -> str:
    messages = state.get("messages") or ()
    logs = state.get("search_log") or []
    max_steps = state.get("max_steps", 0)

    region = state.get("region") or REGION_UNKNOWN
    scope = state.get("region_scope") or ()
    route = (
        f"检索该地区条例 + 国家法，共 {len(scope)} 部"
        if scope
        else "不限地区（全库）"
    )
    lines = [
        f"[agent] 提问：{state.get('question', '')}",
        f"[agent] 地区：{region} → {route}",
    ]

    step = logged = 0
    cursor = 0
    while cursor < len(messages):
        message = messages[cursor]
        cursor += 1
        if message.get("role") != "assistant":
            continue
        step += 1
        calls = message.get("tool_calls") or []
        if not calls:
            lines.append(f"[agent] 轮 {step}/{max_steps} → 未再调用工具，进入作答")
            continue

        replies = messages[cursor : cursor + len(calls)]
        cursor += len(calls)

        for call, reply in zip(calls, replies):
            function = call.get("function") or {}
            name = function.get("name") or "?"
            text = reply.get("content") or ""
            args = function.get("arguments") or "{}"
            if text.startswith("检索#"):
                logged += 1
                row = logs[logged - 1] if logged <= len(logs) else None
                hits = len(row.get("articles") or ()) if row else 0
                lines.append(f"[agent] 轮 {step}/{max_steps} → {name}({args}) → 命中 {hits} 条")
            elif text.startswith(("网搜#", "材料#")):
                lines.append(
                    f"[agent] 轮 {step}/{max_steps} → {name}({args})"
                    f" → {text.splitlines()[0]}"
                )
            else:
                lines.append(
                    _paint(
                        f"[agent] 轮 {step}/{max_steps} → {name}({args}) → 未取到：{text[:60]}",
                        "yellow",
                        color,
                    )
                )

    answer = state.get("answer")
    if answer is not None:
        for note in answer.notes:
            if note.startswith(("Agent：", "已达", "规划轮", "本轮未取到", "共 ", "证据按", "复核")):
                lines.append(
                    _paint(f"[agent] {note}", "yellow", color)
                    if note.startswith(("已达", "本轮未取到", "复核未通过", "复核未完成"))
                    else f"[agent] {note}"
                )
    return "\n".join(lines)


def render_timing(rows: list[dict], *, color: bool = False) -> str:
    if not rows:
        return "[agent] 无计时数据（本次运行没有走到图）"
    width = max(len(row["name"]) for row in rows)
    total = sum(row["total"] for row in rows)
    lines = [_paint(f"[agent] 耗时：共 {total:.2f}s（含节点内等待，不含进程启动）", "dim", color)]
    for row in rows:
        lines.append(
            f"[agent]   {row['name'].ljust(width)}  {row['count']:>2} 次"
            f"  {row['total']:>7.2f}s  均 {row['mean']:>6.2f}s"
        )
    return "\n".join(lines)
