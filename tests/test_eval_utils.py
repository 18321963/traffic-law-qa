"""eval 层的纯函数部分：法名归一、引用配对、语料切桶、指标计算。

**不在这里测 `evaluate()`** —— 它每道题都走一遍真实检索（连 Milvus），
属于 `@pytest.mark.integration` 的范畴。但这个模块里被 `evaluate` 复用的
那批纯函数必须离线可测：它们是 README 里那组 hit@k 的**分母定义**，
分母错了，数字再准也没有意义。（这里**故意不写具体数值** —— 分母的定义与
向量模型无关，把某一版的 hit@3 抄进来只会跟着模型一起腐坏。）
"""

from __future__ import annotations

import json

import pytest

from traffic_law_rag.eval.harness import (
    CITE_WINDOW,
    KS,
    CaseResult,
    EvalCase,
    EvalReport,
    LawResolver,
    ReferenceCase,
    ReferenceReport,
    _cited_articles,
    _kb_index,
    _normalize_law,
    build_cases,
)
from traffic_law_rag.kb.chunker import ChunkStage

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


# ------------------------------------------------- include_leaked（规则取条臂的题集）
def test_默认剔除泄漏桶_行为逐字节不变(tmp_path, resolver):
    """不传 `include_leaked` 时，泄漏题既不进 cases 也不改变计数 —— 既有调用方不受影响。

    这条是「`evaluate()` 零改动」的看门测试：改了默认行为，基线 hit@k 的分母就变了。
    """
    path = _corpus(
        tmp_path,
        [
            {"instruction": "醉驾怎么处罚", "output": f"依据《{LAW}》第九十一条，处拘役。"},
            {"instruction": "第九十一条讲什么", "output": f"《{LAW}》第九十一条规定…"},
        ],
    )
    id_of = {(LAW, "第九十一条"): "a091"}

    cases, no_gold, leaked = build_cases(path, resolver, id_of)

    assert [case.question for case in cases] == ["醉驾怎么处罚"]
    assert (no_gold, leaked) == (0, 1)


def test_include_leaked把泄漏桶收进题集(tmp_path, resolver):
    """开了之后泄漏题**进 cases**，而 `leaked` 计数照旧 —— 计数是语料体检，不是筛选结果。

    这对规则取条臂是必要的：题面自己写着条号，检索指标对它毫无价值（BM25 必然命中），
    但它正是「正则抽条号 → 条号索引定位」这条路的探针，而评测集那 135 题里含条号的是 0 道。
    """
    path = _corpus(
        tmp_path,
        [
            {"instruction": "醉驾怎么处罚", "output": f"依据《{LAW}》第九十一条，处拘役。"},
            {"instruction": "第九十一条讲什么", "output": f"《{LAW}》第九十一条规定…"},
        ],
    )
    id_of = {(LAW, "第九十一条"): "a091"}

    cases, no_gold, leaked = build_cases(path, resolver, id_of, include_leaked=True)

    assert [case.question for case in cases] == ["醉驾怎么处罚", "第九十一条讲什么"]
    assert (no_gold, leaked) == (0, 1)
    # gold 照常解析 —— 收进来的是完整评测题，不是只有题面的残次品
    assert cases[1].gold_ids == ("a091",)


def test_include_leaked不改变无gold的判定(tmp_path, resolver):
    """无 gold 的题仍然被剔除 —— 它对两条臂都没有意义，没人能判它答得对不对。"""
    path = _corpus(
        tmp_path,
        [{"instruction": f"《{LAW}》第九十一条怎么规定", "output": "没有引用任何条号。"}],
    )
    cases, no_gold, leaked = build_cases(
        path, resolver, {(LAW, "第九十一条"): "x"}, include_leaked=True
    )

    assert (len(cases), no_gold, leaked) == (0, 1, 0)


# ------------------------------------------------- 规则取条臂的统计口径
def _ref(question: str, *, gold: bool, located: bool, rank: int | None) -> ReferenceCase:
    """造一条评测结果。`gold` 是「规则答对了」，**没触发就不可能答对** —— 这里强制这条不变量。

    真实数据里 `located_is_gold=True` 蕴含 `located is not None`（没取到条就无从谈对错）。
    让替身也能造出「没取到却答对了」这种状态，测出来的就是替身的 bug 而不是代码的。
    """
    return ReferenceCase(
        question=question,
        gold_citations=("《X法》第九条",),
        located="《X法》第九条" if located else None,
        located_is_gold=bool(located and gold),
        baseline_rank=rank,
    )


def test_规则取条臂的三类互斥且穷尽():
    """触发/回退、答对/答错 —— 报告里的每个数都是这三类的加减，不能有第四个去处。"""
    report = ReferenceReport(
        results=(
            _ref("a", gold=True, located=True, rank=2),    # 规则答对，基线第 1 名不是它
            _ref("b", gold=False, located=True, rank=1),   # 规则错取
            _ref("c", gold=False, located=False, rank=1),  # 回退，基线第 1 名就是 gold
            _ref("d", gold=False, located=False, rank=None),  # 回退，基线也没命中
        ),
        total_raw=4,
        skipped_no_gold=0,
        elapsed_ms=0.0,
        baseline_ran=True,
    )

    assert [c.question for c in report.located] == ["a", "b"]
    assert [c.question for c in report.fell_back] == ["c", "d"]
    assert [c.question for c in report.correct] == ["a"]
    assert [c.question for c in report.wrong] == ["b"]
    assert len(report.located) + len(report.fell_back) == report.cases
    assert len(report.correct) + len(report.wrong) == len(report.located)


def test_回退不算答错但也是分母():
    """回退的题**不算规则答对** —— 否则 93.9% 会被说成 99.5%，那是两回事。

    触发率（93.9%）与触发后的精确率（99.5%）必须分开报：前者是覆盖率，
    后者是「它开口时有多可靠」，合成一个数就会把回退偷偷算成成功。
    """
    report = ReferenceReport(
        results=(
            _ref("a", gold=True, located=True, rank=1),
            _ref("b", gold=False, located=False, rank=1),   # 回退，基线能答对
            _ref("c", gold=False, located=False, rank=None),
            _ref("d", gold=False, located=False, rank=None),
        ),
        total_raw=4,
        skipped_no_gold=0,
        elapsed_ms=0.0,
        baseline_ran=True,
    )

    assert report.cases == 4
    assert len(report.correct) == 1                 # 不是 4，也不是 3
    assert len(report.correct) / report.cases == 0.25
    assert len(report.correct) / len(report.located) == 1.0   # 触发时 1/1 精确


def test_互补性与合起来的上界():
    """「触发就用、回退走检索」这个上线形态的收益，只能由互补性算出来。

    规则独得 = 基线第 1 名拿不到而规则答对的；基线独得 = 规则回退而基线第 1 名就是 gold 的。
    两个方向都要数 —— 只数一个方向就会把这一臂说成净赚。
    """
    report = ReferenceReport(
        results=(
            _ref("a", gold=True, located=True, rank=None),   # 规则独得（基线漏了）
            _ref("b", gold=True, located=True, rank=1),      # 两者都行
            _ref("c", gold=False, located=False, rank=1),    # 基线独得
            _ref("d", gold=False, located=False, rank=None),  # 都没辙
        ),
        total_raw=4,
        skipped_no_gold=0,
        elapsed_ms=0.0,
        baseline_ran=True,
    )

    assert [c.question for c in report.rule_only] == ["a"]
    assert [c.question for c in report.baseline_only] == ["c"]
    assert report.baseline_hits(1) == 2                  # b 和 c
    assert report.combined_hits == 3                     # 规则答对 2 + 基线独得 1
    # 合起来严格优于任何单独一条臂
    assert report.combined_hits > len(report.correct)
    assert report.combined_hits > report.baseline_hits(1)


def _报告(规则对: int, 基线对: int) -> ReferenceReport:
    """造一份「规则答对 N 题、基线 hit@1 命中 M 题」的报告，两者互不重叠。"""
    rows = [_ref(f"rule{i}", gold=True, located=True, rank=None) for i in range(规则对)]
    rows += [_ref(f"base{i}", gold=False, located=False, rank=1) for i in range(基线对)]
    return ReferenceReport(
        results=tuple(rows), total_raw=len(rows), skipped_no_gold=0,
        elapsed_ms=0.0, baseline_ran=True,
    )


@pytest.mark.parametrize(
    "规则对,基线对,词,差",
    [
        (199, 192, "多", 7),    # 换本地 bge 之后的真实情形
        (199, 202, "少", 3),    # 换 bge 之前（云端 v4）的真实情形
        (5, 5, "打平", None),
    ],
)
def test_结论句跟着数字走(规则对, 基线对, 词, 差):
    """结论句必须是**算出来的**，不能是写死的常量。

    它原本写死成「没有命中率增益，这一臂比基线还少 3 题」—— 3 是拿当时的基线 202
    减出来的。后来向量模型从 v4 换成 bge，基线掉到 192，规则臂反倒**多** 7 题，
    方向整个反了，而那句常量仍然印在**重新算出来的**数字正下方，自相矛盾。
    所以这里断言的是最终渲染出来的那一行，不是某个函数 —— 数字与解释必须同源。
    """
    text = _报告(规则对, 基线对).render()
    结论 = text.split("结论：")[1].split("──")[0]

    assert 词 in 结论, f"结论没跟上数字：{结论!r}"
    if 差 is not None:
        assert str(差) in 结论, f"差值写错了：{结论!r}"


def test_基线未命中计入分母():
    """`rank is None` 是**未命中**，不是「没跑」。混为一谈会把基线命中率抬高。"""
    report = ReferenceReport(
        results=(
            _ref("a", gold=True, located=True, rank=1),
            _ref("b", gold=False, located=False, rank=None),
        ),
        total_raw=2,
        skipped_no_gold=0,
        elapsed_ms=0.0,
        baseline_ran=True,
    )

    assert report.baseline_hit_at(1) == 0.5    # 1/2，不是 1/1


def test_基线没跑时不编造数字():
    """Milvus 不可用时基线那一列整体缺席，不能悄悄按 0 算。"""
    report = ReferenceReport(
        results=(_ref("a", gold=True, located=True, rank=None),),
        total_raw=1,
        skipped_no_gold=0,
        elapsed_ms=0.0,
        baseline_ran=False,
    )

    assert report.baseline_hit_at(1) == 0.0
    assert report.to_dict()["baseline_hit_at"] is None
    rendered = report.render()
    assert "基线对照未跑" in rendered
    assert "hit@1" not in rendered   # 没跑就不该出现任何命中率数字


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


def test_空题面被跳过(tmp_path, resolver):
    path = _corpus(tmp_path, [{"instruction": "   ", "output": f"《{LAW}》第九十一条"}])
    cases, no_gold, _ = build_cases(path, resolver, {(LAW, "第九十一条"): "x"})

    assert cases == [] and no_gold == 0


# ------------------------------------------------------------------ 指标
def _result(rank: int | None, *, laws=(LAW,)) -> CaseResult:
    return CaseResult(
        case=EvalCase(
            question="q",
            gold_ids=("g",),
            gold_citations=(f"《{LAW}》第九十一条",),
            gold_laws=tuple(laws),
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
    """空子集不能除零（命中率分母是题数，没有题时直接返回 0）。"""
    assert _report([]).hit_at(3) == 0.0
    assert _report([]).mrr() == 0.0


def test_按法规分组且按hit3():
    report = _report(
        [
            _result(1, laws=(LAW,)),
            _result(4, laws=(LAW,)),          # 名次 4 → hit@3 不算命中
            _result(None, laws=(SZ_ICV,)),
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

    assert set(payload["overall"]["hit_at"]) == {str(k) for k in KS}
    assert payload["total_raw"] == 10
    assert payload["by_law"][LAW] == {"hit3": 1, "cases": 1}


def test_render_包含关键口径():
    """render 是给人读的，题头必须同时出现题数与实际检索通道。"""
    text = _report([_result(1), _result(None)]).render()

    assert "检索评测：2 题" in text
    assert "稠密+BM25" in text
    assert "按法规" in text


def test_未命中清单可展开():
    text = _report([_result(1), _result(None)]).render(show_misses=5)

    assert "未命中 1 题" in text
    assert "期望：" in text


# ------------------------------------------------------------------ 全库索引
def test_kb_index_覆盖全库(chunk_set, monkeypatch):
    """`_kb_index` 是工具层 `check_citation` 的数据源，也是 gold 合法性的判据。

    这里把 `ChunkStage.load` 换成现场切块的产物，避免依赖被 gitignore 的
    `chunks/parents.jsonl` —— 顺带也验证了「从 docx 现场跑一遍」能得出同样的索引。
    """
    monkeypatch.setattr(ChunkStage, "load", lambda self: chunk_set)

    _, known, id_of = _kb_index()

    assert len(known) == len(id_of) == len(chunk_set.parents) == 508
    assert (LAW, "第九十一条") in known
    assert id_of[(LAW, "第九十一条")].endswith("#a091")


def test_引用配对能落到真实parent_id上(chunk_set, monkeypatch):
    """引用配对的窗口、法名归一若改坏了，gold 会集体落到库外 ——
    表现为 hit@3 暴跌，但根因在评测层不在检索层。这里守住那个根因。"""
    monkeypatch.setattr(ChunkStage, "load", lambda self: chunk_set)

    resolver, known, _ = _kb_index()
    pairs = _cited_articles(f"依据《{LAW}》第九十一条，处拘役。", resolver)

    assert pairs and all(pair in known for pair in pairs)
