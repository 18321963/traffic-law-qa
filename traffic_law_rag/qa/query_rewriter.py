"""查询改写层：Query → RewrittenQuery（口语词对齐 + 法名线索）。

    QueryRewriter.rewrite : Query → RewrittenQuery

为什么需要这一层（实测踩到的两个坑）：

1. **口语词命不中法条用语**。用户问"醉驾怎么处罚"，法条里写的是"醉酒驾驶"。
   BM25 只能靠"处罚"这种在整个处罚条例里满地都是的词硬凑，
   结果召回的 Top-1 是"不按交通信号灯通行"——完全跑偏。
2. **跨法规选址错误**。问深圳的事，却召回了国家法律的一般性条款。
   查询里出现"深圳""智能网联汽车"这类法名片段时，对应法规应当加权。

这两个修正都很便宜（纯字符串处理，无需模型），但对法规问答的收益极大。
"""

from __future__ import annotations

from difflib import SequenceMatcher

from ..contracts import Query, RewrittenQuery

# 口语 / 俗称 → 法条用语
QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "醉驾": ("醉酒", "醉酒驾驶", "醉酒后驾驶"),
    "酒驾": ("饮酒", "饮酒后驾驶", "饮酒驾驶"),
    "醉驾入刑": ("醉酒驾驶机动车", "依法追究刑事责任"),
    "闯红灯": ("交通信号灯", "不按交通信号灯规定通行"),
    "违停": ("违法停车", "临时停车", "停车"),
    "超速": ("行驶速度", "超过规定时速", "限速"),
    "无证驾驶": ("未取得机动车驾驶证", "驾驶证", "驾驶资格"),
    "飙车": ("追逐竞驶", "竞速"),
    "疲劳驾驶": ("连续驾驶", "休息"),
    "套牌": ("伪造", "变造", "机动车号牌"),
    "遮挡号牌": ("机动车号牌", "妨碍交通监管"),
    "电动车": ("电动自行车", "非机动车"),
    "自动驾驶": ("智能网联汽车", "自动驾驶系统"),
    "无人驾驶": ("智能网联汽车", "自动驾驶"),
    "扣分": ("记分",),
    "车祸": ("交通事故",),
    "撞人": ("交通事故", "人身伤亡"),
    "记满12分": ("记分", "重新考试"),
    # 干扰驾驶类：法条写的是"拨打接听手持电话""手动操作移动电话"
    "玩手机": ("拨打接听手持电话", "手持电话", "移动电话", "电子设备", "妨碍安全驾驶"),
    "看手机": ("拨打接听手持电话", "手持电话", "移动电话", "妨碍安全驾驶"),
    "打电话": ("拨打接听手持电话", "手持电话"),
    "接电话": ("拨打接听手持电话", "手持电话"),
    "开车": ("驾驶机动车",),
    "超载": ("超过核定载质量", "超过核定人数"),
    "超员": ("超过核定人数",),
    "不系安全带": ("使用安全带",),
    "不戴头盔": ("安全头盔",),
    "礼让行人": ("人行横道", "遇行人正在通过"),
    "占用应急车道": ("应急车道",),
    "变道": ("变更车道",),
    "乱开远光灯": ("远光灯", "不按规定使用灯光"),
}

# 这些片段虽然能和法名匹配上，但属于通用词，不能当作法规线索
GENERIC_FRAGMENTS = frozenset(
    {
        "中华人民共和国",
        "经济特区",
        "深圳经济特区",
        "实施条例",
        "条例",
        "规定",
        "办法",
        "处罚",
        "违法行为",
        "违法",
        "行为",
        "道路交通",
        "道路交通安全",
        "交通",
        "道路",
        "安全",
        "管理",
        "机动车",
        "驾驶",
        "车辆",
        "汽车",
        "法",
        # --- 2026-09 接入《道路运输条例》《交强险条例》时补的 ---
        # 新法名与旧法名共享片段，会凭空造出假线索：问「道路交通安全法」的题
        # 与「中华人民共和国道路运输条例」的最长公共片段是「中华人民共和国道路」，
        # 于是**道交法的题被 ×1.5 加权到了道路运输条例头上**（实测 90 道）。
        # 补进这里之前，先枚举过 4 部法下全语料实际产生过的 16 种线索片段，
        # 这 8 个一个都不在其中 —— 所以对旧法规行为零影响，只掐新法名的假阳性。
        "中华人民共和国道路",
        "交通事故",
        "交通事故责任",
        "机动车交通事故",
        "道路运输",
        "运输",
        "保险",
        "机动",
    }
)

MIN_HINT_LENGTH = 2


class QueryRewriter:
    """查询改写器：把口语问题变成能被法条命中的检索词。

    Input : Query
    Output: RewrittenQuery
    """

    layer = "rewrite"
    input_desc = "Query"
    output_desc = "RewrittenQuery"

    def __init__(
        self,
        *,
        aliases: dict[str, tuple[str, ...]] | None = None,
        law_names: tuple[str, ...] = (),
        max_expansions: int = 8,
    ) -> None:
        self.aliases = dict(QUERY_ALIASES if aliases is None else aliases)
        self.law_names = tuple(law_names)
        self.max_expansions = max_expansions

    # -------------------------------------------------------------- 主接口
    def rewrite(self, query: Query) -> RewrittenQuery:
        text = query.text.strip()
        matched: list[str] = []
        expansions: list[str] = []

        for term, legal_terms in self.aliases.items():
            if term not in text:
                continue
            matched.append(term)
            for legal_term in legal_terms:
                if legal_term not in text and legal_term not in expansions:
                    expansions.append(legal_term)

        expansions = expansions[: self.max_expansions]
        expanded = " ".join([text, *expansions]) if expansions else text
        hints = self.find_law_hints(text, self.law_names)

        return RewrittenQuery(
            original=text,
            expanded=expanded,
            matched_aliases=tuple(matched),
            expansions=tuple(expansions),
            law_hints=tuple(hints),
        )

    # -------------------------------------------------------------- 法名线索
    @staticmethod
    def find_law_hints(text: str, law_names: tuple[str, ...]) -> list[str]:
        """取查询与各法名的最长公共片段，作为"问题属于哪部法规"的线索。

        只取最长的那一段，且过滤掉通用词 —— 否则"…怎么处罚"会因为"处罚"
        二字把所有处罚类条例都算成线索，反而放大噪声。
        """
        hints: list[str] = []
        for name in law_names:
            matcher = SequenceMatcher(None, text, name, autojunk=False)
            block = matcher.find_longest_match(0, len(text), 0, len(name))
            fragment = text[block.a : block.a + block.size]
            if block.size >= MIN_HINT_LENGTH and fragment not in GENERIC_FRAGMENTS:
                hints.append(fragment)
        return list(dict.fromkeys(hints))

    def describe_aliases(self, rewritten: RewrittenQuery) -> str:
        if not rewritten.expansions:
            return ""
        return f"口语对齐：{'、'.join(rewritten.matched_aliases)} → {'、'.join(rewritten.expansions)}"


__all__ = ["QueryRewriter", "QUERY_ALIASES", "GENERIC_FRAGMENTS"]
