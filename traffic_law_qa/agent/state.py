"""Agent 循环的全部状态。

单独成文件是**硬约束**，不是排版偏好：`intent.py` / `nodes.py` 的节点工厂要给
`AgentState` 做类型标注，而 `graph.py` 要 import 那些工厂 —— 放进 `graph.py` 就是循环导入。

`total=False`：节点返回的本来就是**增量**（只带自己写的那几个字段），不是完整状态。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..contracts import Answer

__all__ = ["AgentState"]


class AgentState(TypedDict, total=False):
    # ---------- 输入：调用方写一次，所有节点只读，永不返回 ----------
    question: str
    history: list[tuple[str, str]]   # 本轮恒为空，为多轮对话预留
    top_k: int
    max_steps: int                   # 放进 state 而非闭包：轨迹能自描述、可复现

    # ---------- 循环累积：带 reducer ----------
    # 每个 reducer 都是 operator.add，语义统一为「节点返回增量」。
    # messages 不用 add_messages：那会把 dict 强转成 BaseMessage，
    # 正好丢掉「消息就是 OpenAI 线上格式」这个我们刻意选择的性质。
    messages: Annotated[list[dict], operator.add]
    search_log: Annotated[list[dict], operator.add]   # 每次检索一行 RetrievalResult.to_dict()
    steps: Annotated[int, operator.add]               # 已花掉的规划轮数（agent 与 lookup_plan 都写）
    usage: Annotated[list[dict], operator.add]
    reflections: Annotated[list[dict], operator.add]  # 每轮审核的结论；路由读 [-1]

    # ---------- 意图：classify 写一次，无 reducer（LangGraph 默认 LastValue 通道）----------
    # 意图只在**入口**定一次，回边指向 agent 而不是 classify：第二轮再分类一遍纯属浪费，
    # 而且可能分到不同意图、导致循环在两套工具之间抖动。
    intent: str

    # ---------- 输出：finalize 写一次 ----------
    answer: Answer | None
