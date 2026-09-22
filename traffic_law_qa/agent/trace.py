"""把跑完的这次运行渲染成人看的两张脸：决策链（`render_trace`）与耗时（`render_timing`）。

**纯函数**：只读传进来的 dict，不碰图、不碰网络、不读时钟 —— 离线测试里可以直接喂
构造好的状态。

两个渲染器共用同一套约定：输出都带 `[agent] ` 前缀、都能重定向进 `data/traces/*.log`、
于是 `color` 的默认值必须是 `False` —— 上色只在人对着 TTY 看时才有意义（`cli.py` 用
`sys.stdout.isatty()` 决定）。

决策链渲染的是**终态**，所以看不出时间花在哪；耗时那张表的原料来自 `obs.Recorder.spans`，
聚合在 `Recorder.summary()` 里，这里只负责排版。
"""

from __future__ import annotations

from .region import REGION_UNKNOWN
from .state import AgentState

__all__ = ["render_trace", "render_timing"]

_COLORS = {"yellow": "\033[33m", "dim": "\033[2m"}
_RESET = "\033[0m"


def _paint(text: str, color: str, enabled: bool) -> str:
    """`enabled=False` 时**逐字节返回原串** —— 默认路径（重定向、测试）与加颜色之前相同。"""
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
            if not text.startswith("检索#"):
                lines.append(
                    _paint(
                        f"[agent] 轮 {step}/{max_steps} → {name}({args}) → 未取到：{text[:60]}",
                        "yellow",
                        color,
                    )
                )
                continue
            logged += 1
            row = logs[logged - 1] if logged <= len(logs) else None
            hits = len(row.get("articles") or ()) if row else 0
            lines.append(f"[agent] 轮 {step}/{max_steps} → {name}({args}) → 命中 {hits} 条")

    answer = state.get("answer")
    if answer is not None:
        for note in answer.notes:
            if note.startswith(("Agent：", "已达", "规划轮", "本轮未取到", "共 ", "证据按")):
                lines.append(
                    _paint(f"[agent] {note}", "yellow", color)
                    if note.startswith(("已达", "本轮未取到"))
                    else f"[agent] {note}"
                )
    return "\n".join(lines)


def render_timing(rows: list[dict], *, color: bool = False) -> str:
    """`--timing` 的表：每个名字调了几次、共几秒、单次均值，按总耗时降序。

    与 `render_trace` 分开：那个渲染的是终态（形状），这个渲染的是 `Recorder.spans`
    （时间）。两者的输入毫无重叠，合在一起只会互相将就。

    列宽按 ASCII 名字算的对齐 —— 节点名全是 `node.xxx`，没有宽字符，`ljust` 够用。
    """
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
