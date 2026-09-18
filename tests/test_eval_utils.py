"""eval 层的纯函数部分：法名归一、引用配对、域内判定、指标计算。

**不在这里测 `evaluate()`** —— 它每道题都走一遍真实检索（连 Milvus），
属于 `@pytest.mark.integration` 的范畴。但这个模块里被 `evaluate` 复用的
那批纯函数必须离线可测：它们是 README 里 hit@3 85.2% 这个数字的**分母定义**，
分母错了，数字再准也没有意义。
"""

from __future__ import annotations

import json

import pytest

from traffic_law_rag.chunker import ChunkStage
from traffic_law_rag.eval import (
    CITE_WINDOW,
    KS,
    OUT_OF_DOMAIN_PREFIX,
    CaseResult,
    EvalCase,
    EvalReport,
    LawResolver,
    _cited_articles,
    _is_out_of_domain,
    _kb_index,
    _normalize_law,
    build_cases,
)

LAW = "中华人民共和国道路交通安全法"
REGULATION = "中华人民共和国道路交通安全法实施条例"
SZ_ICV = "深圳经济特区智能网联汽车管理条例"
SZ_PENALTY = "深圳经济特区道路交通安全违法行为处罚条例"
ALL_LAWS = (LAW, REGULATION, SZ_ICV, SZ_PENALTY)


@pytest.fixture(scope="module")
def resolver() -> LawResolver:
    return LawResolver(list(ALL_LAWS))


# ------------------------------------------------------------------ 法名归一
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (LAW, "道路交通安全法"),
        (REGULATION, "道路交通安全法实施条例"),
        (SZ_ICV, SZ_ICV),  # 没有「中华人民共和国」前缀，原样返回
        ("  中华人民共和国道路交通安全法  ", "道路交通安全法"),
    ],
)
def test_法名归一去掉国名前缀(raw, expected):
    assert _normalize_law(raw) == expected


def test_光杆国名前缀不归一():
    """整串就是「中华人民共和国」时不该被削成空串 —— 空串会匹配上一切。"""
    assert _normalize_law("中华人民共和国") == "中华人民共和国"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (LAW, LAW),
        ("道路交通安全法", LAW),  # 简称靠后缀唯一匹配
        (SZ_ICV, SZ_ICV),
        ("智能网联汽车管理条例", SZ_ICV),
        ("道交法", None),  # 不在名字里出现的缩写解析不了 —— 承认这个边界
        ("不存在的法", None),
    ],
)
def test_法名解析(resolver, raw, expected):
    assert resolver.resolve(raw) == expected


def test_歧义简称解析为None(resolver):
    """「条例」同时是实施条例和深圳两部条例的后缀。

    解析器必须宁可放弃也不猜 —— 猜错会把引用归到别的法规上，
    这类错误在指标里表现为「本该命中的没命中」，极难定位。
    """
    assert resolver.resolve("条例") is None
    assert resolver.resolve("实施条例") == REGULATION  # 唯一后缀则正常


# ------------------------------------------------------------------ 引用配对
def test_引用配对要求法名在前(resolver):
    """条号归属于它前面最近的《法名》。"""
    pairs = _cited_articles(f"依据《{LAW}》第九十一条，应处罚。", resolver)

    assert pairs == [(LAW, "第九十一条")]


def test_没有书名号就不配对(resolver):
    """裸条号无从判断属于哪部法 —— eval 里这类答案直接算「无 gold」。"""
    assert _cited_articles("见第九十一条规定。", resolver) == []


def test_多个法名按最近归属(resolver):
    text = f"《{LAW}》第九十一条和《{SZ_ICV}》第八条，都应处罚。"
    assert _cited_articles(text, resolver) == [(LAW, "第九十一条"), (SZ_ICV, "第八条")]


def test_超出窗口的条号不归属(resolver):
    """正文里提到别的法规是常态，隔太远就不敢认（CITE_WINDOW 个字符内才认）。"""
    far = "。" * CITE_WINDOW
    assert _cited_articles(f"《{LAW}》{far}第九十一条", resolver) == []


def test_窗口是闭区间(resolver):
    """边界语义变了会悄悄改变 gold 集合 —— 分母一错，hit@k 再怎么算都不作数。

    窗口从**书名号**起算，不是从法名末尾起算（起算点选错就差两个字符）。
    """
    head = f"《{LAW}》"  # 从《到条号的距离里含这两个书名号
    at_limit = "。" * (CITE_WINDOW - len(head))

    assert _cited_articles(f"{head}{at_limit}第九十一条", resolver) == [(LAW, "第九十一条")]
    assert _cited_articles(f"{head}{at_limit}。第九十一条", resolver) == []


def test_解析不了的法名不产出配对(resolver):
    assert _cited_articles("依据《某部不存在的法》第九十一条", resolver) == []


def test_同一法条引用多次全部保留(resolver):
    """去重是 build_cases 的事，这里保原始顺序地全给出来。"""
    text = f"《{LAW}》第九十一条……又见《{LAW}》第九十一条。"
    assert _cited_articles(text, resolver) == [(LAW, "第九十一条"), (LAW, "第九十一条")]


def test_条号不匹配已归一化的法名时按全称返回(resolver):
    """答案里写简称、库里存全称 —— 返回的必须是库里的全称，否则落不到 parent_id。"""
    assert _cited_articles(f"《{_normalize_law(LAW)}》第九十一条", resolver) == [(LAW, "第九十一条")]


# ------------------------------------------------------------------ 域内判定
@pytest.mark.parametrize(
    "question",
    [
        "醉驾怎么处罚",
        "深圳 行人 在机动车道 罚款多少",
        "第九十一条讲什么",  # 全中文数字，不该被当成英文
        "智能网联汽车道路测试需要什么条件",
    ],
)
def test_中文法条问题算域内(question):
    assert _is_out_of_domain(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "Waymo 在旧金山发生事故",
        "Zoox 的自动驾驶车辆如何定责",
        "Waymo 和 Nuro 谁更安全",
        f"{OUT_OF_DOMAIN_PREFIX}：一辆卡车在高速上侧翻",
    ],
)
def test_英文专名与事故叙述算域外(question):
    """域外题占了评测集的整整一半，判据失效会把 hit@3 从 85.2% 拉到 45.1%。"""
    assert _is_out_of_domain(question) is True


def test_两个字母的英文不算域外():
    """「≥3 个连续字母」是有意选的阈值：小于它的多是型号缩写，不足以判定域外。"""
    assert _is_out_of_domain("AB 型车牌怎么规定") is False
    assert _is_out_of_domain("ABC 型车牌怎么规定") is True


# ------------------------------------------------------------------ build_cases
def _corpus(tmp_path, items: list[dict]):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def test_语料切分成三类(tmp_path, resolver):
    """合法题 / 无 gold / 题面泄漏 —— 三类计数的差定义了 hit@k 的分母。"""
    path = _corpus(
        tmp_path,
        [
            {"instruction": "醉驾怎么处罚", "output": f"依据《{LAW}》第九十一条，处拘役。"},
            {"instruction": "第九十一条讲什么", "output": f"《{LAW}》第九十一条规定…"},
            {"instruction": "美国自动驾驶怎么管", "output": "本库没有相关规定。"},
        ],
    )
    id_of = {(LAW, "第九十一条"): "road_traffic_safety_law@2021-04-29#a091"}

    cases, no_gold, leaked = build_cases(path, resolver, id_of)

    assert [case.question for case in cases] == ["醉驾怎么处罚"]
    assert no_gold == 1
    assert leaked == 1


def test_题面含法名也算泄漏(tmp_path, resolver):
    path = _corpus(
        tmp_path,
        [{"instruction": f"《{LAW}》里怎么规定醉驾", "output": f"《{LAW}》第九十一条…"}],
    )
    _, _, leaked = build_cases(path, resolver, {(LAW, "第九十一条"): "x"})

    assert leaked == 1


def test_无gold优先于泄漏计数(tmp_path, resolver):
    """两个条件同时成立时只计一次，且算「无 gold」—— 先判 gold 再判泄漏。

    这不是吹毛求疵：两条计数会一起进报告的题头，重复计数就凑不回语料总数。
    """
    path = _corpus(
        tmp_path,
        [{"instruction": f"《{LAW}》第九十一条怎么规定", "output": "没有引用任何条号。"}],
    )
    cases, no_gold, leaked = build_cases(path, resolver, {(LAW, "第九十一条"): "x"})

    assert (len(cases), no_gold, leaked) == (0, 1, 0)


def test_库外法条不算gold(tmp_path, resolver):
    """答案里引用的条号必须真在本库 —— 否则 ground truth 永远命不中。"""
    path = _corpus(
        tmp_path,
        [{"instruction": "醉驾怎么处罚", "output": f"依据《{LAW}》第九十二条，应处罚。"}],
    )
    cases, no_gold, _ = build_cases(path, resolver, {(LAW, "第九十一条"): "x"})

    assert cases == []
    assert no_gold == 1


def test_ground_truth取全部引用条号(tmp_path, resolver):
    """一条答案合法引用多条是常态，只认第一条会低估命中率。"""
    path = _corpus(
        tmp_path,
        [
            {
                "instruction": "醉驾还撞了人怎么处理",
                "output": f"依据《{LAW}》第九十一条和《{SZ_PENALTY}》第二十六条，应处罚。",
            }
        ],
    )
    id_of = {(LAW, "第九十一条"): "a091", (SZ_PENALTY, "第二十六条"): "sz026"}

    cases, _, _ = build_cases(path, resolver, id_of)

    assert cases[0].gold_ids == ("a091", "sz026")
    assert cases[0].gold_laws == (LAW, SZ_PENALTY)
    assert cases[0].gold_citations == (f"《{LAW}》第九十一条", f"《{SZ_PENALTY}》第二十六条")


def test_域外题仍进入cases并被标记(tmp_path, resolver):
    """域外题不是被剔除，而是标记后单独统计 —— 否则域内数字会被它们稀释。"""
    path = _corpus(
        tmp_path,
        [{"instruction": "Waymo 的事故怎么判", "output": f"依据《{LAW}》第九十一条。"}],
    )
    cases, _, leaked = build_cases(path, resolver, {(LAW, "第九十一条"): "a091"})

    assert leaked == 0
    assert len(cases) == 1
    assert cases[0].in_domain is False


def test_空题面被跳过(tmp_path, resolver):
    path = _corpus(tmp_path, [{"instruction": "   ", "output": f"《{LAW}》第九十一条"}])
    cases, no_gold, _ = build_cases(path, resolver, {(LAW, "第九十一条"): "x"})

    assert cases == [] and no_gold == 0


# ------------------------------------------------------------------ 指标
def _result(rank: int | None, *, in_domain: bool = True, laws=(LAW,)) -> CaseResult:
    return CaseResult(
        case=EvalCase(
            question="q",
            gold_ids=("g",),
            gold_citations=(f"《{LAW}》第九十一条",),
            gold_laws=tuple(laws),
            in_domain=in_domain,
        ),
        hit_ids=("h1", "h2"),
        hit_citations=(f"《{LAW}》第九十一条", f"《{LAW}》第九十二条"),
        rank=rank,
        used_vector=True,
    )


def _report(rows: list[CaseResult]) -> EvalReport:
    return EvalReport(
        results=tuple(rows),
        total_raw=10,
        skipped_no_gold=2,
        skipped_leak=1,
        top_k=6,
        used_vector=True,
        elapsed_ms=1000.0,
    )


def test_hit_at与mrr():
    report = _report([_result(1), _result(2), _result(None)])

    assert report.hit_at(1) == pytest.approx(1 / 3)
    assert report.hit_at(3) == pytest.approx(2 / 3)
    assert report.hit_at(6) == pytest.approx(2 / 3)
    assert report.mrr() == pytest.approx((1 + 0.5 + 0) / 3)


def test_空子集返回0而不是崩():
    """域外题被筛空（--in-domain）时不能除零。"""
    assert _report([]).hit_at(3) == 0.0
    assert _report([]).mrr() == 0.0


def test_域内外分开统计():
    report = _report([_result(1), _result(1, in_domain=False), _result(None, in_domain=False)])

    assert len(report.in_domain()) == 1
    assert len(report.out_of_domain()) == 2
    assert report.hit_at(3, report.in_domain()) == 1.0
    assert report.hit_at(3, report.out_of_domain()) == 0.5


def test_按法规分组只统计域内且按hit3():
    report = _report(
        [
            _result(1, laws=(LAW,)),
            _result(4, laws=(LAW,)),          # 名次 4 → hit@3 不算命中
            _result(None, laws=(SZ_ICV,)),
            _result(1, laws=(SZ_ICV,), in_domain=False),  # 域外不计入
        ]
    )
    assert report.by_law() == {LAW: (1, 2), SZ_ICV: (0, 1)}


def test_向量标签说实话():
    """请求了向量不等于走了向量 —— A/B 结论的题头不能按命令行开关写。"""

    def report_with(flags: list[bool]) -> EvalReport:
        rows = [
            CaseResult(
                case=_result(1).case,
                hit_ids=(),
                hit_citations=(),
                rank=1,
                used_vector=flag,
            )
            for flag in flags
        ]
        return _report(rows)

    assert report_with([False, False]).vector_label == "纯 BM25"
    assert report_with([True, True]).vector_label == "稠密+BM25"
    assert report_with([True, False]).vector_label == "部分降级（1/2 题走稠密）"


def test_to_dict的hit_at键是字符串():
    """落 JSON 后键必是字符串，别处按 int 取会 KeyError。"""
    payload = _report([_result(1)]).to_dict()

    assert set(payload["in_domain"]["hit_at"]) == {str(k) for k in KS}
    assert payload["total_raw"] == 10
    assert payload["by_law"][LAW] == {"hit3": 1, "cases": 1}


def test_render_包含关键口径():
    """render 是给人读的，题头必须同时出现域内外题数与实际检索通道。"""
    text = _report([_result(1), _result(None, in_domain=False)]).render()

    assert "域内 1 / 域外 1" in text
    assert "稠密+BM25" in text
    assert "按法规" in text


def test_未命中清单可展开():
    text = _report([_result(1), _result(None)]).render(show_misses=5)

    assert "域内未命中 1 题" in text
    assert "期望：" in text


# ------------------------------------------------------------------ 全库索引
def test_kb_index_覆盖全库(chunk_set, monkeypatch):
    """`_kb_index` 是工具层 `check_citation` 的数据源，也是 gold 合法性的判据。

    这里把 `ChunkStage.load` 换成现场切块的产物，避免依赖被 gitignore 的
    `chunks/parents.jsonl` —— 顺带也验证了「从 docx 现场跑一遍」能得出同样的索引。
    """
    monkeypatch.setattr(ChunkStage, "load", lambda self: chunk_set)

    _, known, id_of = _kb_index()

    assert len(known) == len(id_of) == len(chunk_set.parents) == 380
    assert (LAW, "第九十一条") in known
    assert id_of[(LAW, "第九十一条")].endswith("#a091")


def test_引用配对能落到真实parent_id上(chunk_set, monkeypatch):
    """引用配对的窗口、法名归一若改坏了，gold 会集体落到库外 ——
    表现为 hit@3 暴跌，但根因在评测层不在检索层。这里守住那个根因。"""
    monkeypatch.setattr(ChunkStage, "load", lambda self: chunk_set)

    resolver, known, _ = _kb_index()
    pairs = _cited_articles(f"依据《{LAW}》第九十一条，处拘役。", resolver)

    assert pairs and all(pair in known for pair in pairs)
