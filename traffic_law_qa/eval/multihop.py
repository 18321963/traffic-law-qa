"""跨法规多跳题集：从条文反向造题 + 机械护栏。

**为什么需要它。** 现有两臂（默认 `hit@k`、`--reference` 规则取条）的 gold 要么是单条、
要么是同法内几条 —— 没有任何一题要求**跨部法规**。「一个问题的答案横跨两部法」这件事，
两臂在结构上永远测不到。

**为什么不从 refs 构造。** 实测全库 27 条带 refs、共 42 条边，**100% 同法**，跨法边 0 条。
跨法引用只落在 **9 条**条文上（正文里共 22 处出现），且一律是概括性转致
（「依照《道路交通安全法》的规定处十五日以下拘留」）—— **只点名法规，不点名条号**。
所以跨法规多跳没有机械 gold 源，只能从条文反向生成。

**护栏**（不过就重试，连续失败丢弃该条）：

    G1  题面不含条号、不含《法名》   —— 否则题面自带定位信息，检索必然命中，测不出检索
    G2  gold 每一条都能解析到真实存在的条文
    G3  gold 必须跨 >= 2 部法规      —— 硬闸，就是本模块存在的理由
    G4  题面去重

**诚实交代两件事，别被数字骗了：**

1. **gold 是模型给的，不是机械可验证的边。** 这与单跳那 82 道题同一个噪声来源
   （语料的 gold 也是从模型 output 里解析的）。护栏能保证「条存在」「真跨法」，
   保证不了「这一条真的必要」。
2. **所以基线可检索性只做诊断，不做筛选。** 若把「基线捞不到」当成入选条件，
   等于把题集调成「rag 必输」，agent/rag 对比就失去意义。`--check` 会打印每条 gold
   在基线 top-k 里的位置分布，供判断，但不据此增删。

    python -m traffic_law_qa.eval.multihop --check                 # 全离线，复跑护栏
    python -m traffic_law_qa.eval.multihop --generate --target 100 # 花钱调 LLM
    python -m traffic_law_qa.eval.multihop --trace --limit 10 --out data/traces/hop10.json
    python -m traffic_law_qa.eval.multihop --compare --limit 100 --out data/traces/hop100.json
                                                                    # = --trace + 全预算汇总

`data/traces/` 被 gitignore —— 轨迹是跑出来的，不是手工维护的，且随时能重跑。
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..agent.llm import ToolCallingLLM
from ..agent.tools import build_article_index, parse_article_no, resolve_law_id
from ..agent.trace import render_trace
from ..contracts import ParentChunk
from .harness import RE_ARTICLE, RE_LAW

__all__ = [
    "HopError",
    "HopCase",
    "Library",
    "load_cases",
    "check_question",
    "build_case",
    "summarize",
    "trace",
    "full_budget_summary",
    "main",
]

SEP = "#"

GEN_TEMPERATURE = 0.8
GEN_ATTEMPTS = 3


class HopError(ValueError):
    """护栏不通过，或题集文件本身有问题。"""


def _save(path: Path, payload: Any) -> None:
    """落盘。建父目录是因为 `data/traces/` 被 gitignore 了 —— 新克隆里它不存在，
    直接 write_text 会 FileNotFoundError。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def format_key(law_id: str, article_no: str) -> str:
    return f"{law_id}{SEP}{article_no}"


def parse_key(raw: str) -> tuple[str, str]:
    """`law_id#条号` → `(law_id, 条号)`。"""
    law_id, sep, article_no = (raw or "").partition(SEP)
    if not sep or not law_id.strip() or not article_no.strip():
        raise HopError(f"引用格式应为 `law_id{SEP}条号`，收到：{raw!r}")
    return law_id.strip(), article_no.strip()


@dataclass(frozen=True, eq=False)
class Library:
    """题集要用到的最小库视图：条文、`(law_id, 条号)` 索引、全部法名。

    `eq=False`：字段里有 dict，生成出来的 `__hash__` 会在被调用时炸掉。
    """

    parents: dict[str, ParentChunk]
    index: dict[tuple[str, int], ParentChunk]
    law_names: tuple[str, ...]

    @classmethod
    def from_parents(cls, parents: list[ParentChunk]) -> Library:
        mapping = {p.parent_id: p for p in parents}
        return cls(
            parents=mapping,
            index=build_article_index(mapping),
            law_names=tuple(dict.fromkeys(p.law_name for p in parents)),
        )

    @classmethod
    def load(cls) -> Library:
        """读切块产物（`法规知识库/chunks/`），与 `eval._kb_index()` 同一条路。"""
        from ..kb.chunker import ChunkStage

        return cls.from_parents(list(ChunkStage(verbose=False).load().parents))

    def resolve(self, raw: str) -> ParentChunk:
        """`law_id#条号` → 条文。解析不到就是题集坏了，直接抛。"""
        law_id, article_no = parse_key(raw)
        number = parse_article_no(article_no)
        if number is None:
            raise HopError(f"条号解析不出数字：{article_no!r}（{raw}）")
        parent = self.index.get((law_id, number))
        if parent is None:
            raise HopError(f"库里没有这一条：{raw}（{law_id} 第 {number} 条不存在）")
        return parent


RE_PREAMBLE = re.compile(r"制定(?:本|该)(?:条例|法|规定|办法)")


def is_preamble(parent: ParentChunk) -> bool:
    """是不是立法依据条 —— 不参与出题，也不许当目标条。"""
    return bool(RE_PREAMBLE.search(parent.text))


def _law_aliases(library: Library) -> tuple[str, ...]:
    """法名 + 去掉「中华人民共和国」的通用简称，长的在前。

    简称这一条是必须的：试跑里模型写出过「不是说全国道交法里规定不系安全带才罚50吗」，
    它没写书名号，`RE_LAW` 拦不住，但「道路交通安全法」这七个字原样在里面。
    """
    aliases: set[str] = set()
    for name in library.law_names:
        aliases.add(name)
        short = name.removeprefix("中华人民共和国")
        if len(short) >= 4:
            aliases.add(short)
    return tuple(sorted(aliases, key=len, reverse=True))


def check_question(question: str, library: Library) -> None:
    """G1：题面不得含条号、不得含《法名》或其简称。

    只要带上一样，题面就把定位信息自己写出来了 —— 检索必然命中，指标就成了
    在测「正则能不能匹配中文数字」。单跳那套题也是这样剔出来的。
    """
    text = (question or "").strip()
    if not text:
        raise HopError("题面为空")
    if RE_ARTICLE.search(text):
        raise HopError("题面含条号，自带定位信息")
    if RE_LAW.search(text):
        raise HopError("题面含《法名》，自带定位信息")
    for alias in _law_aliases(library):
        if alias in text:
            raise HopError(f"题面含法规名「{alias}」，自带定位信息")


def build_case(raw: dict, library: Library) -> HopCase:
    """一条原始记录 → `HopCase`，沿途跑 G1–G5。不过就抛 `HopError`。"""
    if not isinstance(raw, dict):
        raise HopError(f"记录应为对象，收到 {type(raw).__name__}")

    question = (raw.get("question") or "").strip()
    check_question(question, library)

    gold_raw = raw.get("gold") or []
    if not isinstance(gold_raw, list) or not gold_raw:
        raise HopError("gold 为空")

    parents = [library.resolve(item) for item in gold_raw]
    laws = tuple(dict.fromkeys(p.law_id for p in parents))
    if len(laws) < 2:
        raise HopError(f"gold 全在《{parents[0].law_name}》里，不算跨法规多跳")
    for parent in parents:
        if is_preamble(parent):
            raise HopError(f"{parent.citation} 是立法依据条，不含实质规则，不能当 gold")

    anchor_raw = (raw.get("anchor") or "").strip()
    if anchor_raw:
        library.resolve(anchor_raw)
    anchor_key = parse_key(anchor_raw) if anchor_raw else None

    return HopCase(
        question=question,
        gold_keys=tuple(parse_key(item) for item in gold_raw),
        anchor_key=anchor_key,
        why=(raw.get("why") or "").strip(),
        gold_ids=tuple(p.parent_id for p in parents),
        gold_laws=laws,
        gold_citations=tuple(p.citation for p in parents),
    )


@dataclass(frozen=True)
class HopCase:
    """一道跨法规多跳题：问题 + 答案需要的全部条文。"""

    question: str
    gold_keys: tuple[tuple[str, str], ...]
    anchor_key: tuple[str, str] | None
    why: str
    gold_ids: tuple[str, ...]
    gold_laws: tuple[str, ...]
    gold_citations: tuple[str, ...]

    @property
    def is_cross_law(self) -> bool:
        return len(self.gold_laws) >= 2

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "gold": [format_key(law, no) for law, no in self.gold_keys],
            "anchor": format_key(*self.anchor_key) if self.anchor_key else "",
            "why": self.why,
        }


def _normalize(question: str) -> str:
    """去重用的归一化：只留中日韩文字与字母数字，其余（空白/标点）全丢。"""
    return re.sub(r"[^\w一-鿿]+", "", question)


def load_cases(
    path: Path | None = None, *, library: Library | None = None
) -> tuple[list[HopCase], int]:
    """读题集文件并跑全部护栏；返回（题目, 被去重丢掉几条）。

    文件不存在或护栏不过都直接抛 —— 静默跳过坏记录会让「100 道」变成一句空话。
    """
    path = Path(path or config.EVAL_MULTIHOP_PATH)
    if not path.exists():
        raise HopError(f"多跳题集不存在：{path}（先跑 --generate）")
    library = library or Library.load()

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise HopError(f"{path} 顶层应是数组")

    cases: list[HopCase] = []
    seen: set[str] = set()
    dropped = 0
    for position, item in enumerate(raw, start=1):
        try:
            case = build_case(item, library)
        except HopError as exc:
            raise HopError(f"{path} 第 {position} 条不合格：{exc}") from exc
        key = _normalize(case.question)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        cases.append(case)
    return cases, dropped


def summarize(cases: list[HopCase], *, library: Library | None = None) -> str:
    """题集体检：条数、法规分布、每道题的 gold 跨了几部法。"""
    lines = [f"多跳题集：{len(cases)} 题"]
    if not cases:
        return "\n".join(lines)

    per_law: dict[str, int] = {}
    for case in cases:
        for law_id in case.gold_laws:
            per_law[law_id] = per_law.get(law_id, 0) + 1
    names = {p.law_id: p.law_name for p in (library.parents.values() if library else [])}

    widths = [len(case.gold_laws) for case in cases]
    lines.append(f"  gold 跨法部数：最少 {min(widths)}、最多 {max(widths)}"
                 f"、平均 {sum(widths) / len(widths):.2f}")
    lines.append(f"  gold 条数：{sum(len(c.gold_ids) for c in cases)}"
                 f"（平均 {sum(len(c.gold_ids) for c in cases) / len(cases):.2f} 条/题）")
    lines.append("  法规分布（该法的条文被多少道题的 gold 需要）：")
    for law_id, count in sorted(per_law.items(), key=lambda kv: -kv[1]):
        label = names.get(law_id, law_id)
        lines.append(f"    {count:>4}  {label}")
    return "\n".join(lines)


def overlap_diagnostic(cases: list[HopCase], library: Library) -> str:
    """题面与 gold 原文的最长公共片段 —— 越长说明题面越像在**抄条文**而不是提问。

    G1 拦得住条号和法名，拦不住「把条文内容改写进题面」。这个数就是那份残余重合的
    可见化：中位数要是到了十几个字，说明题面在复述答案，题集就白造了。
    """

    def longest_common(a: str, b: str) -> int:
        previous = [0] * (len(b) + 1)
        best = 0
        for char_a in a:
            current = [0] * (len(b) + 1)
            for j, char_b in enumerate(b, start=1):
                if char_a == char_b:
                    current[j] = previous[j - 1] + 1
                    if current[j] > best:
                        best = current[j]
            previous = current
        return best

    rows = [
        (max(longest_common(case.question, library.parents[pid].text) for pid in case.gold_ids),
         case.question)
        for case in cases
    ]
    values = sorted(row[0] for row in rows)
    lines = [
        f"  题面与 gold 原文的最长公共片段：中位 {values[len(values) // 2]} 字、"
        f"最长 {values[-1]} 字（越小越像「提问」而不是「抄条文」）"
    ]
    for length, question in sorted(rows, reverse=True)[:3]:
        lines.append(f"    {length:>3} 字  {question[:46]}")
    return "\n".join(lines)


RE_APPLY = re.compile(r"(适用|遵守)(本条例|本法)")


def is_pure_scope(parent: ParentChunk) -> bool:
    """纯适用范围条：「……适用本条例。」，去掉管辖句后一个字规则都不剩。

    与 G5 的立法依据条是同一类毛病 —— 那种条不含规则，拿它当 gold 只是空转。
    判据是**去掉含「适用/遵守本条例」的整句后还剩多少字**，不是「有没有这句话」：

        《道路交通安全法》第二条   「……都应当遵守本法。」            → 剩 0 字  ← 是
        《交强险条例》第二条       「……应当……投保交强险。……适用本条例。」→ 剩 71 字 ← 不是

    这条反例必须留着：交强险条例第二条含「适用本条例」，但它的前半句是**投保义务**，
    是全库被引用最多的 gold（21 次，全部正当）。只按关键字拦会把它们一次杀光。
    """
    if not RE_APPLY.search(parent.text):
        return False
    rest = "".join(s for s in re.split(r"[。；]", parent.text) if s and not RE_APPLY.search(s))
    return len(rest) < 20


def defect_diagnostic(cases: list[HopCase], library: Library) -> str:
    """题集瑕疵 —— 两类**值得知道、但绝不拦截**的情况，合成一份报告。

    它们都不能当护栏，因为都有正当反例，拦下去会误伤好题：

    - **gold 引了深圳经济特区法规，题面却没交代「在深圳」。** 深圳两条条例的效力只及于
      特区，同一个「罚三百」在别处未必是这个数。但库里相当一部分处罚**只有深圳条例写了**，
      拒掉就等于把「新法能不能被检索到」这个要害问题一起拒掉。
    - **gold 落在纯适用范围条上**（判据见 `is_pure_scope`）。这类条只有「谁受管辖」、
      没有规则。但同为第二条，《交强险条例》第二条是投保义务条款，只按关键字拦会一次杀光。

    所以只报数、只摆样，让人知道这 100 道里各有多少带这个前提。
    """
    law_name = {p.law_id: p.law_name for p in library.parents.values()}
    sz_total = 0
    sz_samples: list[str] = []
    scope_hits: list[str] = []
    for case in cases:
        if any("深圳" in law_name.get(law, "") for law in case.gold_laws):
            sz_total += 1
            if "深圳" not in case.question:
                sz_samples.append(case.question)
        for pid in case.gold_ids:
            parent = library.parents[pid]
            if is_pure_scope(parent):
                scope_hits.append(
                    f"{case.question[:44]}  ← gold 含 {parent.citation}{parent.article_no}"
                )

    gold_total = sum(len(c.gold_ids) for c in cases)
    if not sz_total and not scope_hits:
        return "  题集瑕疵：无"

    lines = ["  题集瑕疵（只记录，不拦截）"]
    lines.append(
        f"    ① gold 含深圳经济特区法规：{sz_total}/{len(cases)} 题"
        f"（其中题面未提「深圳」：{len(sz_samples)} 题）"
    )
    lines.extend(f"        {q[:46]}" for q in sz_samples[:3])
    lines.append(f"    ② gold 落在纯适用范围条上：{len(scope_hits)}/{gold_total} 条")
    lines.extend(f"        {row}" for row in scope_hits[:3])
    return "\n".join(lines)


def baseline_diagnostic(cases: list[HopCase], *, top_k: int = 6, verbose: bool = True) -> dict:
    """【只诊断，不筛选】看每条 gold 在基线 top-k 里的位置。

    存在意义是让人**看见**题集难度：如果大多数 gold 本来就在 top-6 里，
    那这份题集对 rag 就不算难，agent 赢了也说明不了什么。**不据此增删题目**。
    """
    from ..api import qa
    from ..contracts import RetrievalResult

    hit = miss = 0
    rows: list[dict] = []
    for case in cases:
        result = qa(case.question, mode="search", top_k=top_k, with_vector=True, debug=False)
        assert isinstance(result, RetrievalResult), "mode='search' 应返回检索结果"
        hits = [a.article.parent_id for a in result.articles]
        ranks = {pid: (hits.index(pid) + 1 if pid in hits else None) for pid in case.gold_ids}
        for rank in ranks.values():
            if rank is None:
                miss += 1
            else:
                hit += 1
        rows.append(
            {
                "question": case.question,
                "gold": list(case.gold_citations),
                "ranks": {c: r for c, r in zip(case.gold_citations, ranks.values(), strict=True)},
                "全中": all(r is not None for r in ranks.values()),
            }
        )
        if verbose:
            marks = "、".join(f"{r or '未命中'}" for r in ranks.values())
            print(f"[hop] {case.question[:40]}  gold 排名：{marks}", flush=True)

    total = hit + miss
    payload = {
        "cases": len(cases),
        "gold_total": total,
        "gold_in_topk": hit,
        "gold_missed": miss,
        "gold_hit_rate": round(hit / total, 4) if total else 0.0,
        "all_gold_in_topk": sum(1 for r in rows if r["全中"]),
        "rows": rows,
    }
    print()
    print(f"[诊断] gold 落在基线 top-{top_k} 的比例：{hit}/{total}"
          f"（{hit / total:.1%}）；{payload['all_gold_in_topk']}/{len(cases)} 题 gold 全中")
    return payload


GEN_SYSTEM = (
    "你是交通法规评测集的出题人。你只输出 JSON，不写任何解释性文字。"
)

GEN_TEMPLATE = """下面是一部法规的条文原文。请写**一个**自然的中文用户问题。

硬性要求：
1. 这必须是**多跳**题：**只问一件事**，而这件事要答全，必须把两条串起来用。
   - 对的形状：本条给出规则，规则里的某个概念/资格/范围/标准得看另一条才定得下来。
     例：本条说「不得超过核定的人数」，而「核定的人数」怎么算写在另一条里。
   - 错的形状：把两件事并排问进一句（「这样算不算违规？是不是还得买保险？」）——
     两条各答一半，去掉哪条都还能答，这是拼盘，不是多跳。
2. 问题要像一个真实用户会问的话（口语一点、具体一点），不要写成法律文书。
3. **不许**出现「第X条」「本法」「本条例」这类字样，**不许**出现任何《法规名称》。
4. 另一条**必须来自另一部法规**，不能和下面这条同属一部。

【本条】{law_name}　{article_no}
{text}

可选的另一部法规（只能从这 6 部里挑，且不能是本条所在的那一部）：
{catalog}

只输出这个 JSON，不要有别的内容（why 用一句话说明这两条各自补上了哪一块，
**要写成链**：本条定了什么 → 其中哪个词/资格/标准要靠另一条才落地）：
{{"question": "……", "other_law": "另一条的法规全名", "other_article_no": "第X条", "why": "……"}}"""


def _extract_json(content: str) -> dict:
    """从模型回复里抠出 JSON 对象。容忍 ```json 围栏与前后废话。"""
    text = (content or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise HopError(f"回复里没有 JSON：{content[:120]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise HopError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(data, dict):
        raise HopError(f"JSON 顶层应为对象，收到 {type(data).__name__}")
    return data


def cross_law_sites(library: Library) -> list[ParentChunk]:
    """点名了库内**另一部**法规的条文 —— 天然的跨法种子。

    实测全库只有 9 条（如深圳处罚条例「依照《道路交通安全法》的规定处十五日以下拘留」）。
    它们是概括性转致、给不出条级 gold，但正因如此，拿它们当锚点最可能问出真跨法的问题。
    只认**别的**法：某部法在正文里写自己的名字（序言里的标题、修正案说明）不算。
    """
    sites: list[ParentChunk] = []
    for parent in library.parents.values():
        for match in RE_LAW.finditer(parent.text):
            law_id, _all_names = resolve_law_id(match.group(1), library.parents)
            if law_id is not None and law_id != parent.law_id:
                sites.append(parent)
                break
    return sites


def _candidates(library: Library) -> list[ParentChunk]:
    """出题锚点的顺序：跨法引用点优先，其余**按法规轮转**。

    两个刻意的设计：

    **立法依据条一律出局**（见 `is_preamble`）：跨法引用点里一大半正是
    「根据《X》，制定本条例」这种序言条，不排除的话头几道题全落在它上面。

    **其余按法规轮转交错，不按 (法规, 条号) 字典序。** 字典序会把道交法 124 条
    整段排在前面，100 道题全落在同一部法上，「6 个法规之间多跳」就成了空话。
    轮转后每部法都能持续供题。

    排序固定是为了可复现：同样的库 + 同样的模型，跑出来的顺序应当一致。
    """
    seeds = [p for p in cross_law_sites(library) if not is_preamble(p)]
    seed_ids = {p.parent_id for p in seeds}

    by_law: dict[str, list[ParentChunk]] = {}
    for parent in library.parents.values():
        if parent.parent_id in seed_ids or is_preamble(parent):
            continue
        by_law.setdefault(parent.law_id, []).append(parent)
    for bucket in by_law.values():
        bucket.sort(key=lambda p: parse_article_no(p.article_no) or 0)

    rest: list[ParentChunk] = []
    depth = max((len(bucket) for bucket in by_law.values()), default=0)
    for index in range(depth):
        for law_id in sorted(by_law):
            if index < len(by_law[law_id]):
                rest.append(by_law[law_id][index])
    return seeds + rest


def generate(
    *,
    target: int = 100,
    out: Path | None = None,
    library: Library | None = None,
    top_k: int = 6,
    seed: list[dict] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """从条文反向造题，直到攒够 `target` 道过了护栏的题。

    `seed` 是已经有过的记录（续跑用）：先装进来，再往后补。**补的是没挖过的锚点** ——
    同一个锚点出过的题已经进了 `seed`，再挖一遍只会得到换了说法的同 gold 同考点。
    每接受一道就落一次盘，所以中途断了重跑时把 `out` 读回来当 `seed` 即可接着补。
    """
    library = library or Library.load()
    llm = ToolCallingLLM()
    if not llm.available:
        raise HopError("未配置 LLM_API_KEY，无法生成题集")

    catalog = "\n".join(f"- {name}" for name in library.law_names)
    accepted: list[dict] = list(seed or [])
    seen = {_normalize(item.get("question", "")) for item in accepted}
    mined = {item.get("anchor", "") for item in accepted}
    attempts = failures = 0

    for anchor in _candidates(library):
        if len(accepted) >= target:
            break
        if format_key(anchor.law_id, anchor.article_no) in mined:
            continue

        history: list[dict] = [
            {"role": "system", "content": GEN_SYSTEM},
            {
                "role": "user",
                "content": GEN_TEMPLATE.format(
                    law_name=anchor.law_name,
                    article_no=anchor.article_no,
                    text=anchor.text,
                    catalog=catalog,
                ),
            },
        ]
        for _ in range(GEN_ATTEMPTS):
            attempts += 1
            reply, _usage = llm.chat(history, temperature=GEN_TEMPERATURE)
            content = reply.get("content") or ""
            try:
                data = _extract_json(content)
                other_law = (data.get("other_law") or "").strip()
                other_no = (data.get("other_article_no") or "").strip()
                law_id, _names = resolve_law_id(other_law, library.parents)
                if law_id is None:
                    raise HopError(f"另一条的法规名对不上库：{other_law!r}")
                raw = {
                    "question": (data.get("question") or "").strip(),
                    "gold": [format_key(anchor.law_id, anchor.article_no), format_key(law_id, other_no)],
                    "anchor": format_key(anchor.law_id, anchor.article_no),
                    "why": (data.get("why") or "").strip(),
                }
                case = build_case(raw, library)
                if _normalize(case.question) in seen:
                    raise HopError("题面与已有题目重复")
            except HopError as exc:
                failures += 1
                if verbose:
                    print(f"[gen] 不合格（{exc}），重试", flush=True)
                history += [
                    {"role": "assistant", "content": content[:400]},
                    {"role": "user", "content": f"上一条不合格：{exc}。请重写，只输出 JSON。"},
                ]
                continue

            seen.add(_normalize(case.question))
            mined.add(format_key(anchor.law_id, anchor.article_no))
            accepted.append(case.to_dict())
            if out is not None:
                _save(out, accepted)
            if verbose:
                print(f"[gen] {len(accepted):>3}/{target}  {case.question[:44]}"
                      f"  ← {case.gold_citations[-1]}", flush=True)
            break
        else:
            continue

    if out is not None:
        _save(out, accepted)

    if verbose:
        print(f"\n[gen] 尝试 {attempts} 次 → 采用 {len(accepted)} 道（失败 {failures} 次）")
        if len(accepted) < target:
            print(f"[gen] **没凑够 {target} 道**：候选锚点用完了。换 seed 或放宽 target。")
    return accepted


def _why_of(case: Any) -> str:
    """多跳题带 `why`（这道题为什么算跨法），单跳题没有 —— 一份行格式要能吃两种 case。"""
    return getattr(case, "why", "")


def trace(
    *,
    cases: Sequence[Any] | None = None,
    limit: int | None = 10,
    out: Path | None = None,
    top_k: int = 6,
    verbose: bool = True,
) -> list[dict]:
    """把同一批题在 **rag** 与 **agent** 两侧的原始产物并排 dump 出来。

    **这里刻意不算任何指标。** 先看两边到底产出了什么，再决定怎么比 ——
    先定指标容易把真问题盖掉。两侧都要真调 LLM。

    `cases` 不给就是多跳题集；单跳题集（`eval.singlehop`）复用同一份跑法 ——
    两条臂怎么跑、行里放什么，只此一处实现，改一次两边同时生效。
    case 只要求有 `question` / `gold_ids` / `gold_citations` 三个字段。
    """
    from ..agent.graph import AgentRunner
    from ..api import qa
    from ..contracts import Answer, RetrievalResult

    if cases is None:
        cases, _dropped = load_cases()
    cases = list(cases)[:limit]
    runner = AgentRunner.load(top_k=top_k)

    rows: list[dict] = []
    for position, case in enumerate(cases, start=1):
        if verbose:
            print(f"[trace] {position}/{len(cases)}  {case.question[:44]}", flush=True)

        retrieval = qa(case.question, mode="search", top_k=top_k, with_vector=True, debug=False)
        assert isinstance(retrieval, RetrievalResult)
        answer = qa(case.question, mode="ask", top_k=top_k, with_vector=True, debug=False)
        assert isinstance(answer, Answer)

        state = runner.invoke(case.question)
        agent_answer = state.get("answer")

        rows.append(
            {
                "question": case.question,
                "gold": list(case.gold_citations),
                "gold_ids": list(case.gold_ids),
                "why": _why_of(case),
                "rag": {
                    "answer": answer.text,
                    "cited": [e.citation for e in answer.evidences],
                    "retrieved": [
                        {"citation": a.citation,
                         "parent_id": a.article.parent_id,
                         "score": round(a.score, 4)}
                        for a in retrieval.articles
                    ],
                    "model": answer.model,
                    "usage": answer.usage,
                    "elapsed_ms": round(answer.elapsed_ms, 1),
                    "notes": list(answer.notes),
                },
                "agent": {
                    "answer": agent_answer.text if agent_answer else "",
                    "cited": [e.citation for e in agent_answer.evidences] if agent_answer else [],
                    "steps": state.get("steps"),
                    "intent": state.get("intent"),
                    "searches": state.get("search_log") or [],
                    "reflections": state.get("reflections") or [],
                    "usage": state.get("usage") or [],
                    "trace": render_trace(state),
                },
            }
        )

        if out is not None:
            _save(out, rows)

    if out is not None:
        _save(out, rows)
        if verbose:
            print(f"\n[trace] → {out}")
    return rows


def full_budget_summary(rows: list[dict]) -> str:
    """**各自全预算**：rag 走它的完整一次，agent 走它全部的检索轮次。

    这是产品实际形态的对照 —— agent 本来就允许多轮，把它砍成一轮等于测一个不存在的系统。
    但只报全预算数会误导：agent 会多查几轮，rag 只查一次，多打几枪本来就会多中。
    （这个倍数**随题集与 AGENT_MAX_STEPS 走，不是常数** —— 100 题集上曾是 2.4 次，
    63 题集上是 1.7 次。所以下面那句倍数不要写死，写成"agent 平均检索 N 次"的形态。）
    所以同一份输出里**必须**带上同预算分解（双方都只看第一次检索），
    否则「agent 更强」这个结论分不清是"第一次就查得更准"还是"单纯多查了几轮"。
    """
    index: dict[str, str] = {}
    for row in rows:
        for hit in row["rag"]["retrieved"]:
            index.setdefault(hit["parent_id"], hit["citation"])
        for search in row["agent"]["searches"]:
            for hit in search["articles"]:
                index.setdefault(hit["parent_id"], hit["citation"])

    total = sum(len(row["gold_ids"]) for row in rows)
    tally = {
        "rag": {"found": 0, "cited": 0, "rounds": 0},
        "agent": {"found": 0, "cited": 0, "rounds": 0},
        "agent_first": {"found": 0},
    }
    for row in rows:
        gold = set(row["gold_ids"])
        gold_citations = {index[g] for g in gold if g in index}

        rag_found = {h["parent_id"] for h in row["rag"]["retrieved"]} & gold
        tally["rag"]["found"] += len(rag_found)
        tally["rag"]["cited"] += len(set(row["rag"]["cited"]) & gold_citations)
        tally["rag"]["rounds"] += 1

        searches = row["agent"]["searches"]
        agent_found: set[str] = set()
        for search in searches:
            agent_found |= {h["parent_id"] for h in search["articles"]}
        agent_found &= gold
        tally["agent"]["found"] += len(agent_found)
        tally["agent"]["cited"] += len(set(row["agent"]["cited"]) & gold_citations)
        tally["agent"]["rounds"] += len(searches)

        if searches:
            tally["agent_first"]["found"] += len(
                {h["parent_id"] for h in searches[0]["articles"]} & gold
            )

    n = len(rows)
    lines = [
        f"各自全预算对照（{n} 题，共 {total} 条 gold）",
        "",
        f"  {'':6}{'查到 gold':>12}{'引用 gold':>12}{'平均检索次数':>14}",
    ]
    for side in ("rag", "agent"):
        t = tally[side]
        found = f"{t['found']}/{total}"
        cited = f"{t['cited']}/{total}"
        rounds = t["rounds"] / max(1, n)
        lines.append(f"  {side:6}{found:>12}{cited:>12}{rounds:>14.1f}")
    rag_first = tally["rag"]["found"]
    agent_first = tally["agent_first"]["found"]
    lines += [
        "",
        "  同预算分解（两边都只看第一次检索）",
        f"    rag 单次 {rag_first}/{total}  vs  agent 首次 {agent_first}/{total}",
        "",
        "  ⚠ 「引用 gold」不是「答对了」：gold 由模型断言，读了答案才知道那条是不是真必要。",
    ]
    return "\n".join(lines)


USAGE = __doc__


def _parser() -> argparse.ArgumentParser:
    """只做校验的解析器（`--help` 与「不带参数」由 main 开头那个分支打印 `USAGE`，
    所以 `add_help=False`）。

    模式是四个**平级的布尔开关**，不是子命令 —— 所以谁都没给时会落到末尾打印说明书，
    这个形状与改动前一致。解析器只负责把「不认识的开关」和「取不到值的开关」变成错误：
    `--target` 敲错一个字母静默退回 100 道，和 singlehop 那边 `--limit` 敲错退回全量
    是同一类错误，只是贵在 LLM 那一步。
    """
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa.eval.multihop", add_help=False)
    parser.add_argument("--check", action="store_true", help="复跑护栏，全离线")
    parser.add_argument("--generate", action="store_true", help="调 LLM 造题并落盘（花钱）")
    parser.add_argument("--trace", action="store_true", help="跑 rag/agent 两臂并落盘轨迹（花钱）")
    parser.add_argument("--compare", action="store_true", help="= --trace + 全预算汇总")
    parser.add_argument("--diagnose", action="store_true", help="在 --check 里追加基线诊断")
    parser.add_argument("--quiet", action="store_true", help="不打印逐题进度")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 道（默认 10）")
    parser.add_argument("--target", type=int, default=None, help="--generate 的目标题数（默认 100）")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖默认召回条数（6）")
    parser.add_argument("--out", default=None, help="产物落盘路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args or not args:
        print(USAGE)
        return 0

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    out = Path(options.out) if options.out else None
    try:
        if options.check:
            library = Library.load()
            cases, dropped = load_cases(out or config.EVAL_MULTIHOP_PATH, library=library)
            print(summarize(cases, library=library))
            print(overlap_diagnostic(cases, library))
            print(defect_diagnostic(cases, library))
            if dropped:
                print(f"  （去重丢掉 {dropped} 条重复题面）")
            if options.diagnose:
                baseline_diagnostic(cases, top_k=options.top_k or 6)
            return 0

        if options.generate:
            generate(
                target=options.target or 100,
                out=out or config.EVAL_MULTIHOP_PATH,
                verbose=not options.quiet,
            )
            return 0

        if options.trace or options.compare:
            rows = trace(
                limit=options.limit or 10,
                out=out,
                top_k=options.top_k or 6,
                verbose=not options.quiet,
            )
            if options.compare:
                print()
                print(full_budget_summary(rows))
            return 0
    except HopError as exc:
        print(f"错误：{exc}")
        return 1

    print(USAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
