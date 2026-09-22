"""两个工具给模型看的全部契约：JSON schema + 名字常量 + 取值常量。

名字常量（`SEARCH_LAW_NAME` / `GET_ARTICLE_NAME`）在这里，因为 schema 与工具轮
的分发表都以它为键；`TOP_K_MIN/MAX` 也在这里，因为 schema 的 minimum/maximum
就是它（`arguments.py` 从这里引）。渲染用的两个长度常量在 `render.py`，不在这儿。
"""

from __future__ import annotations

from typing import Any

SEARCH_LAW_NAME = "search_law"

TOP_K_MIN = 1
TOP_K_MAX = 20

SEARCH_LAW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_LAW_NAME,
        "description": (
            "在交通法规知识库（6 部法规、约 508 条）中检索法条。"
            "输入一句话或一组关键词，返回命中的法条（按相关度排序，每条给出法规名+条号+原文摘要+"
            "该条提到的其他条号）。"
            "口语词会被自动对齐成法条用语（如 醉驾→醉酒驾驶）。"
            "需要多部法规才能回答的问题，请分多次检索，每次只问一件事。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索词。**第一轮直接用用户的原话** —— 检索层会自动做口语对齐"
                        "（醉驾→醉酒驾驶）与混合召回，拆成关键词反而稀释信号。"
                        "续查时只写要补的那一块。"
                    ),
                },
                "law_name": {
                    "type": "string",
                    "description": (
                        "可选。只在某一部法规内检索，用于把结果限定到特定法规，"
                        "例如「深圳经济特区道路交通安全违法行为处罚条例」"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": TOP_K_MIN,
                    "maximum": TOP_K_MAX,
                    "description": "返回条数，默认 6",
                },
            },
            "required": ["query"],
        },
    },
}

GET_ARTICLE_NAME = "get_article"

GET_ARTICLE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GET_ARTICLE_NAME,
        "description": (
            "按条号精确取一条法条的原文。**已经知道要查哪一条时用它，比检索更准** —— "
            "检索只保证「相关」，不保证「就是那一条」。"
            "当 search_law 结果里的「相关条」提示指向某条，而你还没看过它时，也用它去取。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "article_no": {
                    "type": "string",
                    "description": "条号，中英文数字都行，例如「第九十条」或「90」",
                },
                "law_name": {
                    "type": "string",
                    "description": (
                        "可选。法规名。省略时若该条号在全库唯一就直接取；"
                        "若多部法规都有这个条号，会返回候选列表让你指明"
                    ),
                },
            },
            "required": ["article_no"],
        },
    },
}
