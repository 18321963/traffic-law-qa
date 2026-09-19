"""跨法规多跳题集的护栏测试 —— 全部离线，不连 Milvus、不调 LLM。

护栏是这份题集唯一的可信度来源（gold 是模型给的，不是机械可验证的边），
所以**负向用例比正向更重要**：必须证明「同法」「泄漏」「不存在的条」都会被拒。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from traffic_law_rag.contracts import ChunkSet
from traffic_law_rag.eval.multihop import (
    HopError,
    Library,
    _candidates,
    build_case,
    check_question,
    cross_law_sites,
    defect_diagnostic,
    format_key,
    full_budget_summary,
    generate,
    is_preamble,
    is_pure_scope,
    load_cases,
    parse_key,
    summarize,
)

PACKAGE = Path(__file__).resolve().parent.parent / "traffic_law_rag"


@pytest.fixture(scope="module")
def library(chunk_set: ChunkSet) -> Library:
    """从现场切块结果建库视图 —— 与 conftest 的取向一致：不读 gitignore 掉的 chunks/。"""
    return Library.from_parents(list(chunk_set.parents))


# ================================================================== 键
def test_键往返一致():
    assert parse_key(format_key("road_transport_regulation", "第八条")) == (
        "road_transport_regulation",
        "第八条",
    )


@pytest.mark.parametrize("raw", ["", "没有井号", "#第八条", "road_transport_regulation#", "#"])
def test_坏键被拒(raw: str):
    with pytest.raises(HopError):
        parse_key(raw)


# ================================================================== G1 泄漏
@pytest.mark.parametrize(
    "question",
    [
        "第九十条怎么规定的？",
        "《道路交通安全法》是怎么说的？",
        "根据《深圳经济特区智能网联汽车管理条例》，责任怎么分？",
    ],
)
def test_题面泄漏被拒(question: str, library: Library):
    """条号与《法名》只要出现一样，答案就被题面自己写出来了。"""
    with pytest.raises(HopError):
        check_question(question, library)


def test_题面不加书名号写法名也算泄漏(library: Library):
    """首轮试跑真踩到过：「不是说全国道交法里规定不系安全带才罚50吗」。

    它没写书名号，`RE_LAW` 拦不住 —— 但法名简称原样在里面，答案等于被问出来了。
    """
    with pytest.raises(HopError, match="法规名"):
        check_question("道路交通安全法里说这种情况只警告，为什么深圳罚这么重", library)


def test_正常题面通过(library: Library):
    check_question("在深圳开车玩手机，罚款多少、扣几分？", library)


def test_空题面被拒(library: Library):
    with pytest.raises(HopError):
        check_question("   ", library)


def test_立法依据条不能当gold(library: Library):
    """「根据《X》，制定本条例」不含任何实质规则。

    拿它出题只会得到空转的多跳 —— 首轮试跑 5 道里踩着 3 道，全是这种序言条。
    """
    raw = {
        "question": "出了事故保险只赔一部分，剩下的该谁掏",
        "gold": [
            format_key("road_traffic_safety_regulation", "第一条"),      # 立法依据条
            format_key("traffic_insurance_regulation", "第二十一条"),
        ],
    }
    with pytest.raises(HopError, match="立法依据条"):
        build_case(raw, library)


def test_深圳适用范围只记录不拦截(library: Library):
    """gold 引了深圳条例、题面却没交代「在深圳」—— 摆出来，但绝不因此拒题。

    深圳两条条例只及于特区，题面不点明地点，gold 是可疑的。但库里相当一部分
    处罚**只有深圳条例写了**，拒掉就等于把「新法能不能被检索到」一起拒掉。
    """
    不点地点的 = build_case(
        {
            "question": "驾驶证被暂扣还开车上路，罚款和拘留会一起罚吗",
            "gold": [
                format_key("sz_traffic_penalty_regulation", "第三十条"),
                format_key("traffic_insurance_regulation", "第三十九条"),
            ],
        },
        library,
    )
    点了地点的 = build_case(
        {
            "question": "在深圳驾驶证被暂扣还开车上路，罚款和拘留会一起罚吗",
            "gold": [
                format_key("sz_traffic_penalty_regulation", "第三十条"),
                format_key("traffic_insurance_regulation", "第三十九条"),
            ],
        },
        library,
    )

    report = defect_diagnostic([不点地点的, 点了地点的], library)
    assert "2/2 题" in report
    assert "题面未提「深圳」：1 题" in report
    assert "只记录，不拦截" in report


def _trace_row(gold, rag_hits, rag_cited, agent_rounds, agent_cited):
    """造一行 trace 记录。`rag_hits` / `agent_rounds` 给 (citation, parent_id) 二元组。"""
    return {
        "gold_ids": list(gold),
        "rag": {
            "retrieved": [{"citation": c, "parent_id": p} for c, p in rag_hits],
            "cited": list(rag_cited),
        },
        "agent": {
            "searches": [
                {"articles": [{"citation": c, "parent_id": p} for c, p in rnd]} for rnd in agent_rounds
            ],
            "cited": list(agent_cited),
        },
    }


def _row_of(report: str, side: str) -> list[str]:
    """取汇总表里某一侧的字段。断言按字段比，不按空格比 —— 列宽是排版细节，不值得钉。"""
    for line in report.splitlines():
        fields = line.split()
        if fields[:1] == [side] and any("/" in f for f in fields):
            return fields
    raise AssertionError(f"汇总里没有 {side} 那一行：\n{report}")


def test_全预算汇总的引用要认得出来():
    """**这条是回归**：汇总里那张 parent_id → citation 的表一度建反了方向。

    gold 给的是 parent_id，答案的 `cited` 是引用串，得拿前者去比后者。反着建不会报错，
    只会让「引用 gold」恒等于 0 —— 一个安静到几乎看不出来的错数。
    """
    row = _trace_row(
        gold=["a#1", "b#1"],
        rag_hits=[("《A》第一条", "a#1")],
        rag_cited=["《A》第一条"],
        agent_rounds=[[("《A》第一条", "a#1"), ("《B》第一条", "b#1")]],
        agent_cited=["《B》第一条"],
    )
    report = full_budget_summary([row])
    assert _row_of(report, "rag") == ["rag", "1/2", "1/2", "1.0"], report
    assert _row_of(report, "agent") == ["agent", "2/2", "1/2", "1.0"], report


def test_全预算汇总要拆出首次检索():
    """agent 多查一轮才捞到的那条，不能算进「首次就查得更准」。"""
    row = _trace_row(
        gold=["a#1", "b#1"],
        rag_hits=[("《A》第一条", "a#1")],
        rag_cited=["《A》第一条"],
        agent_rounds=[[("《A》第一条", "a#1")], [("《B》第一条", "b#1")]],   # 第二条第二轮才有
        agent_cited=["《A》第一条", "《B》第一条"],
    )
    report = full_budget_summary([row])
    assert _row_of(report, "agent") == ["agent", "2/2", "2/2", "2.0"], report   # 全预算 2/2
    assert "rag 单次 1/2  vs  agent 首次 1/2" in report                        # 首次仍是 1/2


def test_纯适用范围条判据不能误伤交强险第二条(library: Library):
    """这条反例是整个判据的护栏。

    `is_pure_scope` 若写成「含『适用本条例』就算」，交强险条例第二条会被判成适用范围条 ——
    可它是**投保义务条款**，是全库被当作 gold 最多的一条（21 次，全部正当）。
    判据必须是「去掉管辖句后还剩多少实质内容」，不是关键字在不在。
    """
    assert is_pure_scope(library.resolve(format_key("road_traffic_safety_law", "第二条")))
    assert not is_pure_scope(library.resolve(format_key("traffic_insurance_regulation", "第二条")))


def test_纯适用范围条只报数不拦题(library: Library):
    case = build_case(
        {
            "question": "出了事故保险只赔一部分，剩下的该谁掏",
            "gold": [
                format_key("road_traffic_safety_law", "第二条"),          # 纯适用范围条
                format_key("traffic_insurance_regulation", "第二十一条"),
            ],
        },
        library,
    )
    # 关键：构建时就**没有**被拒 —— 它是诊断，不是护栏
    report = defect_diagnostic([case], library)
    assert "1/2 条" in report
    assert "只记录，不拦截" in report


def test_没有瑕疵时只印一行(library: Library):
    case = build_case(_cross_raw(), library)
    assert defect_diagnostic([case], library).strip() == "题集瑕疵：无"


def test_is_preamble_认出立法依据条(library: Library):
    assert is_preamble(library.resolve(format_key("traffic_insurance_regulation", "第一条")))
    assert not is_preamble(library.resolve(format_key("traffic_insurance_regulation", "第二条")))


def test_锚点排序跳过立法依据条(library: Library):
    """跨法引用点优先，但这些点里一大半是序言条，必须被排掉。"""
    candidates = _candidates(library)
    assert candidates, "应当还有可用的锚点"
    assert not any(is_preamble(p) for p in candidates), "立法依据条混进了锚点"
    # 跨法引用点仍然排在前面（排除序言之后剩下的那些）
    assert candidates[0].law_id == "road_transport_regulation"


# ================================================================== G2 / G3
def _cross_raw(question: str = "发生交通事故后保险能赔多少，不够的部分谁掏") -> dict:
    return {
        "question": question,
        "gold": [
            format_key("traffic_insurance_regulation", "第二十一条"),
            format_key("road_traffic_safety_law", "第七十六条"),
        ],
        "anchor": format_key("traffic_insurance_regulation", "第二十一条"),
        "why": "交强险赔多少写在条例里，不足部分的分担写在道交法里",
    }


def test_合法跨法题通过(library: Library):
    case = build_case(_cross_raw(), library)
    assert case.is_cross_law
    assert len(case.gold_laws) == 2
    assert len(case.gold_ids) == 2
    assert all(pid in library.parents for pid in case.gold_ids)


def test_同法gold被拒(library: Library):
    """**这条是本模块存在的理由**：gold 落在同一部法里就不算跨法规多跳。"""
    raw = {
        "question": "用不符合条件的驾驶员开客运车辆怎么处罚",
        "gold": [
            format_key("road_transport_regulation", "第六十四条"),
            format_key("road_transport_regulation", "第九条"),
        ],
        "anchor": format_key("road_transport_regulation", "第六十四条"),
        "why": "罚则指向资格条",
    }
    with pytest.raises(HopError, match="不算跨法规多跳"):
        build_case(raw, library)


def test_不存在的条被拒(library: Library):
    raw = {
        "question": "随便问一句",
        "gold": [
            format_key("traffic_insurance_regulation", "第九十九条"),   # 该法只有 46 条
            format_key("road_traffic_safety_law", "第七十六条"),
        ],
    }
    with pytest.raises(HopError, match="库里没有这一条"):
        build_case(raw, library)


def test_对不上的法规名被拒(library: Library):
    raw = {
        "question": "随便问一句",
        "gold": [format_key("不存在的法", "第一条"), format_key("road_traffic_safety_law", "第一条")],
    }
    with pytest.raises(HopError, match="库里没有这一条"):
        build_case(raw, library)


def test_gold为空被拒(library: Library):
    with pytest.raises(HopError, match="gold 为空"):
        build_case({"question": "随便问一句", "gold": []}, library)


# ================================================================== 读取
def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "eval_multihop.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_读取并去重(tmp_path: Path, library: Library):
    """G4：题面归一化后重复的只留第一条。"""
    first = _cross_raw()
    dup = _cross_raw()
    dup["question"] = first["question"] + "？？"     # 只差标点 → 归一化后相同
    other = _cross_raw("交强险和商业险的赔付顺序是什么")

    cases, dropped = load_cases(_write(tmp_path, [first, dup, other]), library=library)
    assert len(cases) == 2
    assert dropped == 1


def test_坏记录直接抛而不是静默跳过(tmp_path: Path, library: Library):
    """静默跳过会让「100 道」变成一句空话，所以坏记录必须让整次读取失败。"""
    bad = {
        "question": "第九十条怎么规定的",
        "gold": [
            format_key("traffic_insurance_regulation", "第二十一条"),
            format_key("road_traffic_safety_law", "第七十六条"),
        ],
    }
    with pytest.raises(HopError, match="第 1 条不合格"):
        load_cases(_write(tmp_path, [bad]), library=library)


def test_题集文件不存在时给出可操作的提示(tmp_path: Path):
    with pytest.raises(HopError, match="不存在"):
        load_cases(tmp_path / "没有这个文件.json")


# ================================================================== 跨法种子
def test_跨法种子只认别的法(library: Library):
    """种子用来优先出题；某部法在正文里提到自己（标题、修正案）不算跨法。"""
    sites = cross_law_sites(library)
    assert sites, "语料里应当存在点名了别的法的条文"

    from traffic_law_rag.agent.tools import resolve_law_id
    from traffic_law_rag.eval.harness import RE_LAW

    for parent in sites:
        cited = [
            resolve_law_id(m.group(1), library.parents)[0] for m in RE_LAW.finditer(parent.text)
        ]
        assert any(law_id is not None and law_id != parent.law_id for law_id in cited)

    # 实施条例第一条：「根据《中华人民共和国道路交通安全法》……制定本条例」—— 板上钉钉的种子
    pairs = {(p.law_id, p.article_no) for p in sites}
    assert ("road_traffic_safety_regulation", "第一条") in pairs


# ================================================================== 续跑
class _照抄锚点的假LLM:
    """照着锚点编题，每次换一部法当「另一条」，题面带流水号好让每条都不一样。"""

    available = True

    def __init__(self, library: Library):
        self.library = library
        self.calls = 0

    def chat(self, history, temperature=0.0):
        del temperature
        prompt = next(row["content"] for row in history if row["role"] == "user")
        # 「【本条】法名　条号」—— 条号那截不能进题面，会被泄漏护栏拦下
        head = next(r for r in prompt.splitlines() if r.startswith("【本条】"))
        anchor_name = head.split()[0][len("【本条】"):]
        other = next(name for name in self.library.law_names if name != anchor_name)

        self.calls += 1
        content = json.dumps(
            {
                "question": f"路上出了点事想问问，这是第{self.calls}回：保险只赔一部分，剩下的谁掏",
                "other_law": other,
                "other_article_no": "第二条",
                "why": "一条管赔偿范围，一条管责任划分",
            },
            ensure_ascii=False,
        )
        return {"content": content}, {}


def test_续跑不重挖同一个锚点(library: Library, monkeypatch):
    """**这条是回归**：续跑时只按题面去重挡不住重挖。

    `generate` 曾经判的是 `_normalize(anchor.text) in seen` —— 拿整条法条去撞题面集合，
    恒不相等，等于没判。锚点表又是固定顺序，于是「续跑补题」永远从第一个锚点重新挖起，
    补多少道都是同 gold 同考点的换皮题。这里让假 LLM 每次都吐一条合格的题面，
    所以能红的只剩锚点没被记住这一件事。
    """
    fake = _照抄锚点的假LLM(library)          # 同一个实例跨两次调用，题面才不重样
    monkeypatch.setattr("traffic_law_rag.agent.graph.ToolCallingLLM", lambda: fake)

    first = generate(target=1, library=library, verbose=False)
    second = generate(target=2, library=library, seed=first, verbose=False)

    assert len(second) == 2
    assert second[1]["anchor"] != first[0]["anchor"]


# ================================================================== 统计
def test_统计按法规分布(library: Library):
    cases = [build_case(_cross_raw(), library)]
    text = summarize(cases, library=library)
    assert "多跳题集：1 题" in text
    assert "最少 2" in text          # 只有一条记录，跨法部数恒为 2
    assert "机动车交通事故责任强制保险条例" in text


# ================================================================== 边界
def test_导入multihop不拉langgraph():
    """生成器里的 `from ..agent.graph import ToolCallingLLM` 必须是延迟导入。

    agent 层会拉 langgraph，一旦写到模块顶层，`import traffic_law_rag.eval.multihop`
    就会把整个图框架拖进来。这条与 test_rag_boundary 的取向一致。
    """
    tree = ast.parse((PACKAGE / "eval" / "multihop.py").read_text(encoding="utf-8"))
    for node in tree.body:  # 只看模块顶层，函数体里的延迟导入不算
        if isinstance(node, ast.ImportFrom):
            # 只有 `agent.graph` 拉 langgraph；`agent.tools` 顶层导入是允许的 ——
            # 它整个模块都不依赖 langgraph（test_agent_tools 另有测试钉住这一点）。
            assert node.module != "agent.graph", "graph 层会拉 langgraph，必须函数内延迟导入"
            assert not (node.module == "agent" and any(a.name == "graph" for a in node.names))
            assert node.module != "langgraph"
        if isinstance(node, ast.Import):
            assert all(alias.name != "langgraph" for alias in node.names)
