"""意图识别：题面里写了条号且能唯一定位 → 跳过检索，精确取条；否则走混合检索。

**纯规则，不调模型。** 题面里写没写「第X条」是正则一眼能看出来的事，用模型分类只会引入
一个新的错误源。好处也不只是省一次调用：意图落在 `state["intent"]` 上，于是它**可观测
（`render_trace` 会打出来）、可对照（`--intent` 能强行走另一边）、可单测** —— 交给模型
隐式决定的话，这三件事都做不到。

**歧义即回退**：多部法规都有这个条号时就判为「法规检索」，交给检索层的法名线索去救。
与 `resolve_law_id` 同一个取舍：宁可走通用路径，也不要猜错法规。

域外/闲聊**刻意不设意图** —— 运行时靠 `tool_choice="auto"` + finalize 兜底已经处理掉了。
"""

from __future__ import annotations

import json

from ..contracts import ParentChunk
from .state import AgentState
from .tools import GET_ARTICLE_NAME, SEARCH_LAW_NAME, find_article

__all__ = [
    "INTENT_LOOKUP",
    "INTENT_SEARCH",
    "classify_intent",
    "make_classify_node",
    "make_lookup_plan_node",
]

INTENT_LOOKUP = "条文定位"
INTENT_SEARCH = "法规检索"


def classify_intent(
    question: str,
    *,
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
) -> str:
    """题面 → 意图。纯函数，可脱离图单测。

    判据只有一个：**能不能唯一定位到一条法条**。定位得到 → 这条题不需要检索，
    用户已经把条号告诉我们了，取出来就行；定位不到 → 老老实实检索。
    """
    if find_article(question, parents=parents, index=index) is not None:
        return INTENT_LOOKUP
    return INTENT_SEARCH


def make_classify_node(
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
    forced: str | None = None,
):
    """意图识别。

    读：question
    写：{"intent": str}（无 reducer 的通道，写一次）

    `forced` 是给 A/B 对照用的：把同一道题强行按另一种意图跑一遍，看「规则判断」这一层
    到底贡献了什么。为 None 时才是真实规则。
    """

    def classify_node(state: AgentState) -> dict:
        return {
            "intent": forced
            or classify_intent(state["question"], parents=parents, index=index)
        }

    return classify_node


def make_lookup_plan_node(
    parents: dict[str, ParentChunk], index: dict[tuple[str, int], ParentChunk]
):
    """「条文定位」的规划轮 —— 不调模型，直接构造一条 get_article 调用。

    读：question
    写：{"messages": [带 tool_calls 的助手消息], "steps": 1}

    **刻意不在这里取法条，只产出一个 tool_call。** 让既有的 `tools` 节点去执行，
    换来的是四件事全部零改动：「每个 tool_call 恰好一条 tool 消息」的不变量由既有代码
    保证（不需要新的消息形状）；`reflect` 看到的仍是普通的工具问答；`search_log` 里仍是
    普通的 `RetrievalResult.to_dict()` 行（`merge_retrievals` 零改动）；路由沿用
    `route_after_agent`（有 tool_calls 就去 tools），它绝不看预算。

    写 `steps: 1` 是有意的：**由规则驱动的规划也是规划**，它消耗了一轮预算。
    代价是 `steps` 从「只有 agent 写」变成「agent 与 lookup_plan 都写」——
    在 `operator.add` 下每次 +1 仍然正确，只是单一写者的说法不再成立。
    """

    def lookup_plan_node(state: AgentState) -> dict:
        target = find_article(state["question"], parents=parents, index=index)
        if target is None:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_lookup_1",
                                "type": "function",
                                "function": {
                                    "name": SEARCH_LAW_NAME,
                                    "arguments": json.dumps(
                                        {"query": state["question"]}, ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    }
                ],
                "steps": 1,
            }

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_lookup_1",
                            "type": "function",
                            "function": {
                                "name": GET_ARTICLE_NAME,
                                "arguments": json.dumps(
                                    {
                                        "article_no": target.article_no,
                                        "law_name": target.law_name,
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            ],
            "steps": 1,
        }

    return lookup_plan_node
