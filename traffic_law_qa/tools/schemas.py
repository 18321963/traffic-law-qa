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

WEB_SEARCH_NAME = "web_search"

WEB_COUNT_MIN = 1
WEB_COUNT_MAX = 20

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": WEB_SEARCH_NAME,
        "description": (
            "联网检索公开网页。**只用于时效性问题**：某地是不是刚出了新规定、现在还有效吗、"
            "最新口径是什么、政策是不是已经改了。"
            "**问「罚多少」「记几分」「这条怎么规定」一律不许用它** —— 那些答案只在法条里，"
            "用网搜答等于拿二手转述当法条，整篇答案会被判无依据。"
            "库内能查到的法条一律不用它；只有问题本身问的就是「最新 / 现在 / 是否已修改」时才用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "搜索词。要写清楚地区、年份、文件名"
                        "（如「深圳 电动自行车 管理条例 2026」），不要与上一轮重复。"
                    ),
                },
                "count": {
                    "type": "integer",
                    "minimum": WEB_COUNT_MIN,
                    "maximum": WEB_COUNT_MAX,
                    "description": "返回条数，默认 5",
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_MATERIALS_NAME = "search_materials"

MATERIAL_TOP_K_MIN = 1
MATERIAL_TOP_K_MAX = 10
MATERIAL_TOP_K_DEFAULT = 5

SEARCH_MATERIALS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_MATERIALS_NAME,
        "description": (
            "在**用户本次上传的材料**里检索（会话材料，尚未入知识库）。"
            "只在用户明确说到「我传的这份文件 / 附件 / 材料」时才用它；"
            "问法条一律用 search_law —— 材料不是法条，引用标记也不同。"
            "本次会话没上传材料时它会回一句「没有材料」，那就别再调它。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索词。用材料里真出现过的说法，别用同义词改写。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": MATERIAL_TOP_K_MIN,
                    "maximum": MATERIAL_TOP_K_MAX,
                    "description": f"返回段数，默认 {MATERIAL_TOP_K_DEFAULT}",
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
