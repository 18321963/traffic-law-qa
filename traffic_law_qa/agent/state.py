"""Agent 循环的全部状态。

单独成文件是**硬约束**，不是排版偏好：`region.py` / `nodes.py` 的节点工厂要给
`AgentState` 做类型标注，而 `graph.py` 要 import 那些工厂 —— 放进 `graph.py` 就是循环导入。

`total=False`：节点返回的本来就是**增量**（只带自己写的那几个字段），不是完整状态。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..contracts import Answer

__all__ = ["AgentState"]


class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    top_k: int
    max_steps: int

    messages: Annotated[list[dict], operator.add]
    search_log: Annotated[list[dict], operator.add]
    steps: Annotated[int, operator.add]
    usage: Annotated[list[dict], operator.add]

    region: str
    region_scope: tuple[str, ...]

    answer: Answer | None
