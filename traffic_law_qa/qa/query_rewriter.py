from __future__ import annotations

from difflib import SequenceMatcher

from ..contracts import Query, RewrittenQuery

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

    @staticmethod
    def find_law_hints(text: str, law_names: tuple[str, ...]) -> list[str]:
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
