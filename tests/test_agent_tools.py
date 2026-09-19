"""工具层：条号解析、条号索引、精确取条、摘要窗口。

**这一层不依赖 langgraph，也不连 Milvus** —— `agent_tools` 自述就是「不认识 LangGraph、
可脱离图单测」，这个文件是把那句话兑现。父块来自 `chunk_set` fixture（现场跑切块，
不是读 `chunks/*.jsonl`，后者被 gitignore、CI 上不存在），所以断言里的法名、条号、
条文正文全是真货。

重头戏是 `find_article` 的**从严判据**：它一旦返回非 None，调用方就会跳过检索直接取条，
取错比取不到糟得多。所以「什么时候必须返回 None」和「什么时候必须命中」同等重要。
"""

from __future__ import annotations

import json

import pytest

from traffic_law_rag.agent.tools import (
    GET_ARTICLE_TOOL,
    _snippet,
    build_article_index,
    find_article,
    lookup_article,
    parse_article_arguments,
    parse_article_no,
    render_tool_result,
)
from traffic_law_rag.contracts import ParentChunk


@pytest.fixture(scope="module")
def parents(chunk_set) -> dict[str, ParentChunk]:
    return chunk_set.parent_map()


@pytest.fixture(scope="module")
def index(parents) -> dict[tuple[str, int], ParentChunk]:
    return build_article_index(parents)


# ================================================================== 条号解析
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("第九十条", 90),
        ("第十三条", 13),
        ("第一条", 1),
        ("第一百零八条", 108),
        ("第90条", 90),
        ("90", 90),
        ("九十", 90),
        ("  第九十条  ", 90),
    ],
)
def test_parse_article_no_accepts(raw, expected):
    assert parse_article_no(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "第条", "第X条", "abc", None])
def test_parse_article_no_rejects(raw):
    assert parse_article_no(raw) is None


# ================================================================== 条号索引
def test_index_is_keyed_by_law_and_number(parents, index):
    """键是 `(law_id, 条号)`，不是 `article_index`。

    后者是**位置序号**（law_parser 里 `pending_index = len(articles) + 1`），
    在本库里恰好等于条号，但那是语料巧合不是契约 —— 法规修订跳号时两者立刻分叉。
    """
    assert index
    for (law_id, number), parent in index.items():
        assert parent.law_id == law_id
        assert parse_article_no(parent.article_no) == number


def test_same_number_in_different_laws_stays_distinct(parents, index):
    """「第一条」在 6 部法里各有一条 —— 单靠条号不能当键。"""
    firsts = [(law_id, p) for (law_id, number), p in index.items() if number == 1]
    assert len(firsts) >= 2, "本库应有多部法规各有第一条"
    assert len({law_id for law_id, _ in firsts}) == len(firsts)

    # 反过来：这个条号在库内**不唯一**，所以不指名法规时必须拒绝
    assert find_article("第一条怎么规定的", parents=parents, index=index) is None


# ================================================================== 合规的法名抽取
def test_find_article_with_bracketed_law(parents, index):
    hit = find_article(
        "《中华人民共和国道路交通安全法》第九十条是什么？", parents=parents, index=index
    )
    assert hit is not None
    assert hit.law_name == "中华人民共和国道路交通安全法"
    assert hit.article_no == "第九十条"


@pytest.mark.parametrize(
    "question",
    [
        # 不带书名号 —— 真实用户的写法
        "道路交通安全法实施条例第六十六条怎么规定的",
        # 语料生成时书名号损坏（开括号被写成 _），法名文本本身是完整的
        "_中华人民共和国道路交通安全法实施条例》第六十二条禁止了哪些驾驶行为？",
    ],
)
def test_find_article_without_brackets(parents, index, question):
    hit = find_article(question, parents=parents, index=index)
    assert hit is not None
    assert hit.law_name == "中华人民共和国道路交通安全法实施条例"


def test_longer_law_name_beats_its_own_prefix(parents, index):
    """「道路交通安全法」整段出现在「道路交通安全法实施条例」里，必须由长名吃掉短名。

    这条是回归测试：此前的实现是「命中两个法名就判歧义」，于是
    「…实施条例第六十六条」被误判成提到了两部法规而退回检索 —— 实际上题面
    只提到了一部。实测这条让泄漏桶里 18 道题白白失去精确取条。
    """
    hit = find_article("道路交通安全法实施条例第六十六条", parents=parents, index=index)
    assert hit is not None
    assert hit.law_name == "中华人民共和国道路交通安全法实施条例"

    # 只说「道路交通安全法」时不能被「实施条例」抢走
    hit = find_article("道路交通安全法第六十六条", parents=parents, index=index)
    assert hit is not None
    assert hit.law_name == "中华人民共和国道路交通安全法"


def test_two_laws_really_mentioned_is_ambiguous(parents, index):
    """真提到两部法规时仍然拒绝 —— 长名吃短名不该把这条放宽。"""
    question = "《中华人民共和国道路交通安全法》和《中华人民共和国道路交通安全法实施条例》第九十条"
    assert find_article(question, parents=parents, index=index) is None


# ================================================================== 必须回退的情形
def test_find_article_returns_none_when_no_article_number(parents, index):
    """没有条号 → None。即使题面指名了法规。

    泄漏桶里有一批这种题（「在《X 法》中提到的罚款收据…」），它们**本来就该**走检索：
    题面没说问哪一条，精确取条无从谈起。
    """
    assert find_article(
        "在《中华人民共和国道路交通安全法》中提到的罚款收据怎么理解？",
        parents=parents,
        index=index,
    ) is None


def test_find_article_returns_none_when_out_of_range(parents, index):
    """条号超出该法规范围 → None（宁可检索，也不要取到一个越界的空）。"""
    assert find_article("《中华人民共和国道路交通安全法》第九千条", parents=parents, index=index) is None


def test_find_article_returns_none_for_unknown_law(parents, index):
    """题面引用了库外法规 → None，不能顺手用库内某部法顶上。"""
    assert find_article("《机动车管理办法》第九十条", parents=parents, index=index) is None


def test_find_article_ignores_false_positive_numbers(parents, index):
    """「第二条街」「第二条车道」不是条号引用 —— 靠跨法歧义兜住，不是靠正则变聪明。

    「第二条」在 6 部法里都存在，所以不指名法规时必然回退。这条钉住的是**行为**
    （回退），而不是实现（正则）—— 正则永远会把这些误当条号，只要回退兜得住就无害。
    当初是拿美国事故叙述（「第二条街」）发现的，那批题已删，但误匹配本身还在，
    真实用户也会说「从左侧第二条车道转弯」。
    """
    assert find_article("从左侧第二条车道转弯", parents=parents, index=index) is None


# ================================================================== get_article 工具
def test_get_article_schema_is_valid():
    fn = GET_ARTICLE_TOOL["function"]
    assert fn["name"] == "get_article"
    assert fn["parameters"]["required"] == ["article_no"]
    assert set(fn["parameters"]["properties"]) == {"article_no", "law_name"}


def test_parse_article_arguments():
    assert parse_article_arguments(json.dumps({"article_no": "第九十条"})) == ("第九十条", None)
    assert parse_article_arguments(json.dumps({"article_no": "90", "law_name": "X法"})) == ("90", "X法")
    assert parse_article_arguments(json.dumps({"article_no": "第九十条", "law_name": "  "})) == ("第九十条", None)


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "not json", "[]", json.dumps({"law_name": "X法"})],
)
def test_parse_article_arguments_rejects(raw):
    with pytest.raises(ValueError):
        parse_article_arguments(raw)


def test_lookup_article_success(parents, index):
    result, error = lookup_article(
        "第九十条", "中华人民共和国道路交通安全法", parents=parents, index=index
    )
    assert error == ""
    assert result is not None
    assert len(result.articles) == 1
    assert result.articles[0].citation.startswith("《中华人民共和国道路交通安全法》")
    assert "第九十条" in result.articles[0].citation


def test_lookup_article_accepts_arabic_number(parents, index):
    """「第90条」与「第九十条」取到同一条。"""
    cn, _ = lookup_article("第九十条", "中华人民共和国道路交通安全法", parents=parents, index=index)
    ar, _ = lookup_article("90", "中华人民共和国道路交通安全法", parents=parents, index=index)
    assert cn is not None and ar is not None
    assert cn.articles[0].article.parent_id == ar.articles[0].article.parent_id


def test_lookup_article_without_law_is_ambiguous(parents, index):
    """不指名法规、而多部法都有这个条号 → 取不到，并列出候选法规。"""
    result, error = lookup_article("第一条", None, parents=parents, index=index)
    assert result is None
    assert "请指明法规" in error or "部法规" in error


def test_lookup_article_unique_number_resolves_without_law(parents, index):
    """条号在全库唯一时，不指名法规也能取到。

    实测本库只有 116..124 是跨法唯一的（6 部法条号区间重叠：1..46 六部都有、
    47..64 五部、65..77 四部、78..82 三部、83..115 两部）。所以这条测试**不能**改用
    「第九十条」之类 —— 那个条号跨法重复，会走进歧义分支。
    """
    unique = sorted(
        number
        for number in {no for _, no in index}
        if sum(1 for _, no in index if no == number) == 1
    )
    assert unique, "本库应当存在跨法唯一的条号"
    result, error = lookup_article(str(unique[0]), None, parents=parents, index=index)
    assert error == ""
    assert result is not None


@pytest.mark.parametrize(
    ("article_no", "law_name", "fragment"),
    [
        ("第X条", None, "无法识别条号"),
        ("第九千条", "中华人民共和国道路交通安全法", "没有第"),
        ("第九十条", "《不存在的法》", "无法确定法规"),
    ],
)
def test_lookup_article_failure_paths(parents, index, article_no, law_name, fragment):
    """失败一律回一句中文，不抛异常 —— 条号写错是可恢复的对话，不是程序错误。"""
    result, error = lookup_article(article_no, law_name, parents=parents, index=index)
    assert result is None
    assert fragment in error


def test_out_of_range_message_reports_the_actual_range(parents, index):
    """越界时把该法规的真实范围报出来，模型下一轮就不会再瞎猜。"""
    _, error = lookup_article(
        "第九千条", "中华人民共和国道路交通安全法", parents=parents, index=index
    )
    assert "共 124 条" in error


# ================================================================== 全链路形状
def test_lookup_result_is_a_normal_retrieval_result(parents, index):
    """精确取条的产物必须与检索结果同形状 —— 这是全链路零改动的关键。

    包成 `RetrievalResult` 之后，`to_dict()` → `search_log` → `merge_retrievals`
    → `generator.generate()` 四段代码一行都不用改。这里钉住那个形状契约。
    """
    result, _ = lookup_article("第十三条", "深圳经济特区道路交通安全违法行为处罚条例", parents=parents, index=index)
    assert result is not None

    row = result.to_dict()
    assert set(row) == {"query", "used_vector", "used_bm25", "elapsed_ms", "notes", "articles"}
    assert row["used_vector"] is False and row["used_bm25"] is False   # 没走任何检索通道
    parent_id = result.articles[0].article.parent_id
    assert row["articles"][0]["parent_id"] == parent_id
    assert any("精确取条" in note for note in row["notes"])            # 说明清楚它没检索

    # 再从 parent_id 回灌原文 —— 生成层就是这么拿正文的
    assert parents[parent_id].text == result.articles[0].article.text


def test_lookup_article_scores_are_not_compared(parents, index):
    """合并走轮转交错（按排名不按分数），所以占位的 score 不会被拿去做比较。"""
    result, _ = lookup_article("第十三条", "深圳经济特区道路交通安全违法行为处罚条例", parents=parents, index=index)
    assert result.articles[0].score == 1.0
    assert result.articles[0].vector_rank is None and result.articles[0].bm25_rank is None


# ================================================================== 摘要窗口
def test_snippet_falls_back_to_head_without_overlap():
    text = "第一条　为了维护道路交通秩序，预防和减少交通事故，保护人身安全。"
    assert _snippet(text, "完全不相干的查询词", 12) == text[:12] + "…"


def test_snippet_puts_the_hat_quote_first_when_window_is_deep():
    """窗口离开头远时，帽子句 + … + 窗口两头都给。

    列举型法条里「处罚」在帽子句、「情形」在列举项里，单窗口只能看见一头。
    """
    text = (
        "第六十二条　驾驶机动车不得有下列行为："
        "（一）在车门、车厢没有关好时行车；（二）在机动车驾驶室的前后窗范围内悬挂、放置妨碍驾驶人视线的物品；"
        "（三）拨打接听手持电话；（四）下陡坡时熄火或者空挡滑行。"
    )
    out = _snippet(text, "拨打接听手持电话", 80)
    assert out.startswith("第六十二条")           # 帽子句在
    assert "…" in out                            # 中间是省略
    assert "拨打接听手持电话" in out              # 窗口也在


def test_snippet_downweights_repeated_grams():
    """按 1/词频 加权：泛词反复出现时不该把窗口拽走。

    「驾驶机动车」在这条法条里出现 4 次，若每个二元组命中都记 1，
    窗口会被拉向靠前的无关列举项（超速 / 逆行）。按词频倒数加权之后，
    稀有词「停车让行」才主导窗口位置。
    """
    text = (
        "第一条　驾驶机动车应当遵守下列规定："
        "（一）驾驶机动车不得超速；（二）驾驶机动车不得逆行；（三）驾驶机动车不得闯红灯；"
        "（四）驾驶机动车行经人行横道时，应当减速让行，遇行人正在通过时应当停车让行。"
    )
    out = _snippet(text, "停车让行", 60)
    assert "停车让行" in out


def test_snippet_uses_the_expanded_query_not_the_raw_one(parents):
    """摘要窗口按**检索层实际用的词**定位，不按用户原话。

    这条测试盯的是一个曾经真实存在、且**静默**的缺陷：口语词在法条里压根不出现，
    按原话选窗口会让窗口停在条文开头，而真正说明「这条与问题有关」的那一项
    （（七）手动操作移动电话、电子设备）落在窗口之外 —— 审核节点于是判
    「缺罚款具体数额」，尽管答案就在同一条里。表现出来像是模型的规划能力差。

    从前负责换算的是 `agent_tools.expand_query`，它靠
    `getattr(retriever, "rewriter")` 摸进检索器内部。改写器一改名，`getattr`
    返回 None，它就原样返回 —— 不报错、不失败，只是窗口悄悄退回错误位置。
    现在换算由 `HybridRetriever.expand` 承担，**用真改写器**跑真语料。
    """
    from traffic_law_rag.qa.retriever import HybridRetriever

    retriever = HybridRetriever(store=object(), parents=parents, chunks={})
    raw = "深圳开车玩手机，罚多少、扣几分"
    expanded = retriever.expand(raw)

    assert expanded != raw, "口语对齐应当真的改写了查询"
    assert "移动电话" in expanded or "手持电话" in expanded, expanded

    # 真实场景：第十三条的第七项写着「手动操作移动电话、电子设备」
    text = parents["sz_traffic_penalty_regulation@2024-05-10#a013"].text
    window = _snippet(text, expanded, 60)
    assert "移动电话" in window, f"展开后窗口应当落在那项上，实际是：{window}"

    # 反证：按原话定位就够不着 —— 这条不变量一旦失效，上面的断言就成了摆设
    assert "移动电话" not in _snippet(text, raw, 60)


def test_render_tool_result_reports_given_and_fresh(parents, index):
    result, _ = lookup_article("第十三条", "深圳经济特区道路交通安全违法行为处罚条例", parents=parents, index=index)
    text = render_tool_result(result, index=1, seen=set(), snippet_chars=200)
    assert text.startswith("检索#1")
    assert "命中 1 条（新增 1 / 已知 0）" in text

    known = {result.articles[0].article.parent_id}
    text = render_tool_result(result, index=2, seen=known, snippet_chars=200)
    assert "新增 0 / 已知 1" in text
    assert "⚠ 本轮没有新增法条" in text    # 这是模型判断「该停了」的直接依据
