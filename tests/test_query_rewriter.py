"""rewrite 层：口语词 → 法条用语对齐 + 法名线索。

这一层的两个修正都是**实测纠正过召回**的（README 第 5 节），
所以每条规则都值得一条回归断言，别让它们在后续改动中悄悄失效。
"""

from __future__ import annotations

import pytest

from traffic_law_rag.contracts import Query
from traffic_law_rag.query_rewriter import (
    GENERIC_FRAGMENTS,
    MIN_HINT_LENGTH,
    QUERY_ALIASES,
    QueryRewriter,
)

LAW_NAMES = (
    "中华人民共和国道路交通安全法",
    "中华人民共和国道路交通安全法实施条例",
    "深圳经济特区智能网联汽车管理条例",
    "深圳经济特区道路交通安全违法行为处罚条例",
)


@pytest.fixture
def rewriter() -> QueryRewriter:
    return QueryRewriter(law_names=LAW_NAMES)


# ------------------------------------------------------------------ 口语对齐
def test_醉驾展开为醉酒驾驶(rewriter):
    """README 记录的失败案例：问「醉驾」时法条写的是「醉酒驾驶」，
    未处理时 BM25 只能靠「处罚」硬凑，Top-1 召回到「不按交通信号灯通行」。"""
    result = rewriter.rewrite(Query(text="醉驾怎么处罚"))

    assert "醉驾" in result.matched_aliases
    assert "醉酒驾驶" in result.expansions
    assert "醉酒驾驶" in result.expanded
    assert result.changed


def test_展开文本保留原问题(rewriter):
    """扩写是「原文 + 法条用语」，不是替换 —— 原文里的口语词仍要参与检索。"""
    result = rewriter.rewrite(Query(text="醉驾怎么处罚"))
    assert result.expanded.startswith("醉驾怎么处罚")


def test_不重复追加已存在的法条用语(rewriter):
    """问题里已经写了「醉酒驾驶」就不该再追加一次。"""
    result = rewriter.rewrite(Query(text="醉酒驾驶怎么处罚"))
    assert "醉酒驾驶" not in result.expansions


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("酒驾", "饮酒后驾驶"),
        ("闯红灯", "交通信号灯"),
        ("无证驾驶", "未取得机动车驾驶证"),
        ("电动车", "电动自行车"),
        ("自动驾驶", "智能网联汽车"),
        ("玩手机", "拨打接听手持电话"),
        ("不系安全带", "使用安全带"),
        ("超载", "超过核定载质量"),
    ],
)
def test_常见口语词都有映射(spoken, expected):
    assert spoken in QUERY_ALIASES
    assert expected in QUERY_ALIASES[spoken]


def test_没有口语词时不做任何改写(rewriter):
    result = rewriter.rewrite(Query(text="第九十一条规定的是什么"))

    assert result.expanded == "第九十一条规定的是什么"
    assert result.expansions == ()
    assert not result.changed


def test_扩写条数有上限():
    """一次命中多个口语词时不能让检索串无限膨胀，否则 BM25 会被噪声词淹没。"""
    rewriter = QueryRewriter(
        aliases={"交通": tuple(f"法条用语{i}" for i in range(20))}, max_expansions=3
    )
    result = rewriter.rewrite(Query(text="交通"))

    assert len(result.expansions) == 3


def test_describe_aliases_只在真改写时给提示(rewriter):
    assert rewriter.describe_aliases(rewriter.rewrite(Query(text="醉驾怎么处罚")))
    assert rewriter.describe_aliases(rewriter.rewrite(Query(text="第九十一条"))) == ""


# ------------------------------------------------------------------ 法名线索
def test_深圳命中法名线索(rewriter):
    """README 记录的第二个失败案例：问深圳的事却召回国家法律一般条款。"""
    result = rewriter.rewrite(Query(text="深圳 行人 在机动车道 罚款多少"))
    assert "深圳" in result.law_hints


def test_通用词不当法名线索(rewriter):
    """「…怎么处罚」里的「处罚」二字能和两个条例的名字匹配上，
    但它显然是通用词 —— 放行会把所有处罚类条例一起放大，反而增加噪声。"""
    result = rewriter.rewrite(Query(text="醉驾怎么处罚"))
    assert "处罚" not in result.law_hints


def test_通用词表覆盖主要噪声源():
    assert {"处罚", "条例", "实施条例", "中华人民共和国", "经济特区"} <= GENERIC_FRAGMENTS


def test_线索长度门槛():
    """太短的公共片段是巧合，不是线索。"""
    assert MIN_HINT_LENGTH >= 2
    assert QueryRewriter.find_law_hints("的", LAW_NAMES) == []


def test_线索去重(rewriter):
    """「深圳」能同时匹配两部深圳法规，但只该出现一次。"""
    hints = QueryRewriter.find_law_hints("深圳 智能网联汽车", LAW_NAMES)
    assert len(hints) == len(set(hints))


def test_智能网联汽车命中对应法规(rewriter):
    result = rewriter.rewrite(Query(text="智能网联汽车道路测试需要什么条件"))
    assert result.law_hints, "问智能网联汽车却没有任何法名线索"


def test_没有法名时线索为空(rewriter):
    assert QueryRewriter.find_law_hints("醉酒驾驶怎么处理", LAW_NAMES) == []


def test_未提供法名时不产线索():
    assert QueryRewriter().rewrite(Query(text="深圳怎么罚")).law_hints == ()


# ------------------------------------------------------------------ 契约
def test_改写结果可序列化(rewriter):
    result = rewriter.rewrite(Query(text="醉驾怎么处罚"))
    payload = result.to_dict()

    assert payload["original"] == "醉驾怎么处罚"
    assert "醉酒驾驶" in payload["expansions"]
    assert isinstance(payload["matched_aliases"], list)  # 落 JSON 要 list 不是 tuple


def test_改写不改动原Query(rewriter):
    """Query 是 frozen dataclass，改写器不该有副作用。"""
    query = Query(text="醉驾怎么处罚")
    rewriter.rewrite(query)
    assert query.text == "醉驾怎么处罚"
