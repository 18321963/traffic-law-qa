"""离线检索评测：data/eval_corpus.json（指令语料）→ hit@1/@3/@6 + MRR。

    python -m traffic_law_rag.eval                    # 跑全部可用评测题（135 条）
    python -m traffic_law_rag.eval --limit 20         # 先跑 20 条看链路
    python -m traffic_law_rag.eval --no-vector        # 只走 BM25，用来 A/B 对比混合检索
    python -m traffic_law_rag.eval --show-misses 10   # 打印没命中的题，便于定位
    python -m traffic_law_rag.eval --json             # 机器可读

另一条臂（测「条文定位」而不是检索）：

    python -m traffic_law_rag.eval --reference        # 题面含条号那批题 → 规则取条能否唯一定位
    python -m traffic_law_rag.eval --reference --no-compare   # 连基线对照都不跑，纯离线

为什么需要第二条臂：构造评测题时会**剔除**题面含条号或法名的题（答案泄漏，检索必然命中），
所以那 135 题里含「第…条」的是 0 道 —— 检索指标在结构上永远衡量不到规则取条这条新路径。
被剔除的那批反而是一份现成的、已标注的探针集，`--reference` 就是拿它来测。

语料本身不是评测集，475 条 instruction/output 里只有一部分能当检索题用：

- **132 条**的答案里没有能定位到本库的条号 → 构造不出 ground truth
- **208 条**的题面自己就写着「第X条」或《法名》→ 答案泄漏，检索必然"命中"，测不出东西
- 剩下 **135 条**既能定位 gold、题面又不泄漏 → 本模块跑这些
  → hit@1 75.6% / hit@3 85.2% / hit@6 88.1% / MRR 0.805

**这 475 条是删过的。** 原语料 779 条里有 **296 条美国自动驾驶事故叙述**
（Waymo / Cruise / Zoox 在旧金山、洛杉矶的碰撞叙述），本库是中国交通法规，
一条也覆盖不了，而且它们的 gold 是硬凑的 —— 检索一律返回深圳条例第五十三条
反而是合理行为，域外 hit@3 仅 4.5%，混进合计会把 hit@3 从 85.2% 拉到 45.1%。
另删 8 条英文版深圳条例题（跨语言检索不是本评测要测的东西）。
两类合起来 304 条，2026-09 一次性删净，删法见 docs/DESIGN.md。

原先为此设的**「域内外分开报」机制已一并拆除**：域外题清零后，`in_domain` 标记、
`--in-domain` 开关、报告里的域外行全都失去了真实调用，留着就是一份没人执行的契约。

ground truth 取答案里引用的**全部**本库条号，不是只取第一条：一条答案合法引用
多条是常态，只认第一条会低估命中率。

每条都用 `qa(..., mode="search")` 跑，所以测的就是对外那个接口本身。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..contracts import RetrievalResult

__all__ = ["main", "evaluate", "build_cases", "EvalCase", "EvalReport"]

RE_LAW = re.compile(r"《([^》]{2,40})》")
RE_ARTICLE = re.compile(r"第[零一二三四五六七八九十百千]+条")
# 条号必须紧跟在法名后面才算数，隔太远就不敢认了（正文里可能提到别的法规）
CITE_WINDOW = 50
KS = (1, 3, 6)


@dataclass(frozen=True)
class EvalCase:
    """一条评测题：问题 + 期望命中的法条（parent_id）。"""

    question: str
    gold_ids: tuple[str, ...]
    gold_citations: tuple[str, ...]   # 人读，用于打印
    gold_laws: tuple[str, ...]        # 去重后的法规名，用于分组统计


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    hit_ids: tuple[str, ...]                  # 检索到的 parent_id，按排名
    hit_citations: tuple[str, ...]            # 同上，人读版（含条号）
    rank: int | None                          # 第一个 gold 的名次；None = 没命中
    used_vector: bool                         # 实际是否走了稠密通道（没 key / 纯 BM25 集合会降级）

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass(frozen=True)
class EvalReport:
    results: tuple[CaseResult, ...]
    total_raw: int                    # 语料原始条数
    skipped_no_gold: int              # 构造不出 gold 的
    skipped_leak: int                 # 题面泄漏的
    top_k: int
    used_vector: bool
    elapsed_ms: float

    @property
    def cases(self) -> int:
        return len(self.results)

    def hit_at(self, k: int, subset: tuple[CaseResult, ...] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.rank is not None and r.rank <= k) / len(rows)

    def mrr(self, subset: tuple[CaseResult, ...] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return sum(1.0 / r.rank for r in rows if r.rank is not None) / len(rows)

    def _line(self, label: str, rows: tuple[CaseResult, ...]) -> str:
        if not rows:
            return f"  {label}：（无）"
        metrics = "  ".join(f"hit@{k} {self.hit_at(k, rows):>5.1%}" for k in KS)
        return f"  {label}：{metrics} ｜ MRR {self.mrr(rows):.3f}  （{len(rows)} 题）"

    def by_law(self) -> dict[str, tuple[int, int]]:
        """法规 → （命中 top-3 的题数, 该法规总题数）。"""
        grouped: dict[str, list[bool]] = {}
        for item in self.results:
            for law in item.case.gold_laws:
                grouped.setdefault(law, []).append(item.rank is not None and item.rank <= 3)
        return {law: (sum(flags), len(flags)) for law, flags in sorted(grouped.items())}

    def to_dict(self) -> dict:
        def metrics(rows: tuple[CaseResult, ...]) -> dict:
            return {
                "cases": len(rows),
                "hit_at": {str(k): round(self.hit_at(k, rows), 4) for k in KS},
                "mrr": round(self.mrr(rows), 4),
            }

        return {
            "cases": self.cases,
            "total_raw": self.total_raw,
            "skipped_no_gold": self.skipped_no_gold,
            "skipped_leak": self.skipped_leak,
            "top_k": self.top_k,
            "used_vector": self.used_vector,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "overall": metrics(self.results),
            "by_law": {
                law: {"hit3": hit, "cases": total} for law, (hit, total) in self.by_law().items()
            },
        }

    @property
    def vector_label(self) -> str:
        """按实际检索用到的通道标注，而不是按命令行开关。

        请求了向量不等于真走了向量：没配 EMBED_API_KEY、或集合是纯 BM25 建的，
        检索层会降级成只用 BM25。这个标签是 A/B 结论的题头，必须说实话。
        """
        used = sum(1 for row in self.results if row.used_vector)
        if used == 0:
            return "纯 BM25"
        if used == len(self.results):
            return "稠密+BM25"
        return f"部分降级（{used}/{len(self.results)} 题走稠密）"

    def render(self, *, show_misses: int = 0) -> str:
        mode = self.vector_label
        lines = [
            f"检索评测：{self.cases} 题"
            f" ｜ {mode} ｜ top_k={self.top_k} ｜ 耗时 {self.elapsed_ms / 1000:.1f}s",
            f"来源：{self.total_raw} 条语料 → 剔除 {self.skipped_no_gold} 条无 gold、"
            f"{self.skipped_leak} 条题面泄漏",
            "",
            self._line("合计", self.results),
            "",
            "  按法规（hit@3）：",
        ]
        for law, (hit, total) in self.by_law().items():
            lines.append(f"    {hit / total:>6.1%}  {hit:>4}/{total:<4}  {law}")

        if show_misses:
            missed = [r for r in self.results if not r.hit]
            lines.append("")
            lines.append(f"  未命中 {len(missed)} 题，前 {min(show_misses, len(missed))} 条：")
            for item in missed[:show_misses]:
                lines.append(f"    Q: {item.case.question[:56]}")
                lines.append(f"       期望：{'、'.join(item.case.gold_citations)}")
                got = "、".join(item.hit_citations[:3]) or "（空）"
                lines.append(f"       实得：{got}")
        return "\n".join(lines)


# ================================================================== 语料 → 评测题
def _normalize_law(name: str) -> str:
    """去掉「中华人民共和国」前缀，让简称和全称能对上。

    不这么做的话，「道路交通安全法」会同时是「…道路交通安全法」和
    「…道路交通安全法实施条例」的子串，简称会被解析到错的那一部。
    """
    name = name.strip()
    prefix = "中华人民共和国"
    return name[len(prefix):] if name.startswith(prefix) and len(name) > len(prefix) else name


class LawResolver:
    """把答案里写的《法名》对到本库的法规上。"""

    def __init__(self, law_names: list[str]) -> None:
        self.by_norm = {_normalize_law(name): name for name in law_names}

    def resolve(self, name: str) -> str | None:
        norm = _normalize_law(name)
        if norm in self.by_norm:
            return self.by_norm[norm]
        candidates = [full for key, full in self.by_norm.items() if key.endswith(norm)]
        # 只认唯一候选：「条例」这种对到多部的，宁可放弃也不猜
        return candidates[0] if len(candidates) == 1 else None


def _cited_articles(text: str, resolver: LawResolver) -> list[tuple[str, str]]:
    """答案里（法名, 条号）的配对：条号归属于它前面最近的那个《法名》。"""
    laws = [(m.start(), m.group(1)) for m in RE_LAW.finditer(text)]
    if not laws:
        return []

    pairs: list[tuple[str, str]] = []
    for match in RE_ARTICLE.finditer(text):
        preceding = [(start, name) for start, name in laws if 0 <= match.start() - start <= CITE_WINDOW]
        if not preceding:
            continue
        law = resolver.resolve(preceding[-1][1])
        if law:
            pairs.append((law, match.group(0)))
    return pairs


def build_cases(
    data_path: Path,
    resolver: LawResolver,
    id_of: dict[tuple[str, str], str],
    *,
    include_leaked: bool = False,
) -> tuple[list[EvalCase], int, int]:
    """语料 → 评测题；返回（题目, 无 gold 条数, 题面泄漏条数）。

    `include_leaked=True` 时把「题面泄漏」那批**收进**评测集而不是剔除。
    它们对检索指标毫无价值（题面自己写着条号，BM25 必然命中），但正是
    **规则取条**那条路的探针：题面给出的条号就是 gold 的比例高达 94%，
    是一份已经标注好、规模比手搓探针集大一个量级的现成测试集。

    默认 `False` —— 既有调用方的行为逐字节不变。
    """
    raw = json.loads(Path(data_path).read_text(encoding="utf-8"))
    known = set(id_of)
    cases: list[EvalCase] = []
    no_gold = leaked = 0

    for item in raw:
        question = (item.get("instruction") or "").strip()
        output = item.get("output") or ""
        if not question:
            continue

        pairs = [(law, art) for law, art in _cited_articles(output, resolver) if (law, art) in known]
        if not pairs:
            no_gold += 1
            continue
        # 题面自己写着条号或法名 → 答案泄漏，检索必然命中，测不出东西
        if RE_ARTICLE.search(question) or RE_LAW.search(question):
            leaked += 1
            if not include_leaked:
                continue

        gold_pairs = list(dict.fromkeys(pairs))  # 去重保序
        cases.append(
            EvalCase(
                question=question,
                gold_ids=tuple(id_of[(law, art)] for law, art in gold_pairs),
                gold_citations=tuple(f"《{law}》{art}" for law, art in gold_pairs),
                gold_laws=tuple(dict.fromkeys(law for law, _ in gold_pairs)),
            )
        )
    return cases, no_gold, leaked


def _kb_index() -> tuple[LawResolver, set[tuple[str, str]], dict[tuple[str, str], str]]:
    from ..kb.chunker import ChunkStage

    parents = ChunkStage(verbose=False).load().parents
    resolver = LawResolver(sorted({p.law_name for p in parents}))
    known = {(p.law_name, p.article_no) for p in parents}
    id_of = {(p.law_name, p.article_no): p.parent_id for p in parents}
    return resolver, known, id_of


# ================================================================== 跑评测
def evaluate(
    *,
    data_path: Path | None = None,
    limit: int | None = None,
    top_k: int | None = None,
    with_vector: bool = True,
    verbose: bool = True,
) -> EvalReport:
    """逐题跑检索，统计 hit@k 与 MRR。"""
    from ..api import qa

    data_path = Path(data_path or config.EVAL_CORPUS_PATH)
    if not data_path.exists():
        from ..api import QaError

        raise QaError(f"评测语料不存在：{data_path}")

    resolver, _known, id_of = _kb_index()
    cases, no_gold, leaked = build_cases(data_path, resolver, id_of)
    if limit:
        cases = cases[:limit]
    if not cases:
        from ..api import QaError

        raise QaError(f"{data_path} 里没有可用的评测题（无 gold {no_gold} 条 / 题面泄漏 {leaked} 条）")

    top_k = top_k or max(KS)
    started = time.perf_counter()
    results: list[CaseResult] = []

    for index, case in enumerate(cases, start=1):
        result: RetrievalResult = qa(
            case.question, mode="search", top_k=top_k, with_vector=with_vector, debug=False
        )
        hits = tuple(hit.article.parent_id for hit in result.articles)
        rank = next((i for i, parent_id in enumerate(hits, start=1) if parent_id in case.gold_ids), None)
        results.append(
            CaseResult(
                case=case,
                hit_ids=hits,
                hit_citations=tuple(hit.citation for hit in result.articles),
                rank=rank,
                used_vector=result.used_vector,
            )
        )
        if verbose and (index % 10 == 0 or index == len(cases)):
            marked = "ok " if rank else "miss"
            print(f"[eval] {index:>4}/{len(cases)}  {marked}  {case.question[:36]}", flush=True)

    return EvalReport(
        results=tuple(results),
        total_raw=len(json.loads(data_path.read_text(encoding="utf-8"))),
        skipped_no_gold=no_gold,
        skipped_leak=leaked,
        top_k=top_k,
        used_vector=any(row.used_vector for row in results),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


# ================================================================== 规则取条臂
@dataclass(frozen=True)
class ReferenceCase:
    """一条「题面点名了某条」的题：规则取条拿到了什么，基线检索又拿到了什么。"""

    question: str
    gold_citations: tuple[str, ...]
    located: str | None      # 规则取条拿到的引用；None = 没走这条路（回退检索）
    located_is_gold: bool    # 拿到的那条是不是 gold
    baseline_rank: int | None  # 基线检索里第一个 gold 的名次；None = 未命中或没跑


@dataclass(frozen=True)
class ReferenceReport:
    """规则取条（意图识别 + get_article）在「题面含条号」那批题上的表现。

    **这一臂完全离线**：它测的能力（正则抽条号 + 条号索引定位）只需要 `parents`，
    不连 Milvus、不调 LLM。基线那一列才需要检索，没跑时如实标出来。
    """

    results: tuple[ReferenceCase, ...]
    total_raw: int
    skipped_no_gold: int
    elapsed_ms: float
    baseline_ran: bool = False   # 基线那一列是否真的跑了（Milvus 没起时为 False）

    @property
    def cases(self) -> int:
        return len(self.results)

    @property
    def located(self) -> tuple[ReferenceCase, ...]:
        return tuple(r for r in self.results if r.located is not None)

    @property
    def fell_back(self) -> tuple[ReferenceCase, ...]:
        return tuple(r for r in self.results if r.located is None)

    @property
    def correct(self) -> tuple[ReferenceCase, ...]:
        return tuple(r for r in self.results if r.located_is_gold)

    @property
    def wrong(self) -> tuple[ReferenceCase, ...]:
        """规则取条拿到了**不是** gold 的那一条 —— 这是唯一真正有害的一类。"""
        return tuple(r for r in self.results if r.located is not None and not r.located_is_gold)

    def baseline_hit_at(self, k: int) -> float:
        """基线命中率。`baseline_rank is None` 就是**未命中** —— 分母是全部题，不是命中的那些。"""
        if not self.baseline_ran or not self.results:
            return 0.0
        return sum(
            1 for r in self.results if r.baseline_rank is not None and r.baseline_rank <= k
        ) / len(self.results)

    def baseline_hits(self, k: int) -> int:
        return sum(1 for r in self.results if r.baseline_rank is not None and r.baseline_rank <= k)

    @property
    def rule_only(self) -> tuple[ReferenceCase, ...]:
        """规则答对了、而基线**第 1 名不是它**的题 —— 规则这条路真正赚到的部分。"""
        return tuple(r for r in self.results if r.located_is_gold and r.baseline_rank != 1)

    @property
    def baseline_only(self) -> tuple[ReferenceCase, ...]:
        """规则回退、基线第 1 名就是 gold 的题 —— 规则交给基线反而更好的部分。"""
        return tuple(r for r in self.results if r.located is None and r.baseline_rank == 1)

    @property
    def combined_hits(self) -> int:
        """合起来（规则触发就用规则，回退就用基线）在「第 1 名」上的上界。"""
        return len(self.correct) + len(self.baseline_only)

    def render(self) -> str:
        total = self.cases
        lines = [
            f"规则取条评测：{total} 题（题面含条号或法名的「泄漏桶」）"
            f" ｜ 离线 ｜ 耗时 {self.elapsed_ms / 1000:.1f}s",
            f"来源：{self.total_raw} 条语料 → 剔除 {self.skipped_no_gold} 条无 gold",
            "",
            f"  规则定位到唯一一条：{len(self.located)}/{total}"
            f"（{len(self.located) / total:.1%}）"
            if total else "  （无题）",
        ]
        if total:
            lines.append(
                f"    其中就是 gold：{len(self.correct)}/{len(self.located)}"
                f"（{len(self.correct) / max(1, len(self.located)):.1%}）"
                f" ｜ 疑似错取：{len(self.wrong)}"
            )
            lines.append(
                f"  回退到检索：{len(self.fell_back)}/{total}"
                f"（{len(self.fell_back) / total:.1%}）—— 回退不是错，是判据从严的另一面"
            )
            lines.append("")
            lines.append(
                f"  成本：定位到的 {len(self.located)} 题每题 **0 次检索调用**"
                f"（无 embedding、无 BM25）、0 轮 LLM 规划"
            )
        if not self.baseline_ran:
            lines.append("  基线对照未跑（Milvus 不可用或 --no-compare）。以上是纯离线部分。")
            return "\n".join(lines)

        n = total
        lines += [
            "",
            f"  基线（同一批题走检索）：hit@1 {self.baseline_hits(1)}/{n}"
            f"（{self.baseline_hit_at(1):.1%}）  hit@3 {self.baseline_hits(3)}/{n}"
            f"（{self.baseline_hit_at(3):.1%}）",
            "",
            "  ── 结论：**没有命中率增益**，这一臂比基线还少 3 题 ──",
            f"    规则答对 {len(self.correct)}，基线 hit@1 {self.baseline_hits(1)}。"
            f"两者不是同一件事：规则回退 {len(self.fell_back)} 题，基线在其中的 "
            f"{len(self.baseline_only)} 题上第 1 名就是 gold。",
            f"    反过来规则也独得 {len(self.rule_only)} 题（基线第 1 名不是它）。"
            f"合起来「触发就用、回退就走检索」上界 {self.combined_hits}/{n}"
            f"（{self.combined_hits / n:.1%}）。",
            "",
            "  ── 那它赚在哪 ──",
            f"    ① 成本：触发的 {len(self.located)} 题 **0 次检索调用**、0 轮 LLM 规划；"
            f"基线每条都要 embedding + BM25。",
            f"    ② 确定性：触发时给的是唯一一条法条原文，不是一份排名 —— "
            f"精确率 {len(self.correct)}/{len(self.located)}"
            f"（{len(self.correct) / max(1, len(self.located)):.1%}）。",
            "    ③ 判据从严：回退的题交给检索，**不是丢了** —— 合起来才是上线形态。",
            "",
            "  已知脏数据：错取的那条是语料本身矛盾（题面写「第九十五条」、gold 是第九十六条），"
            "不是定位逻辑错。",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "cases": self.cases,
            "total_raw": self.total_raw,
            "skipped_no_gold": self.skipped_no_gold,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "located": len(self.located),
            "correct": len(self.correct),
            "wrong": len(self.wrong),
            "fell_back": len(self.fell_back),
            "baseline_hit_at": (
                {str(k): round(self.baseline_hit_at(k), 4) for k in KS} if self.baseline_ran else None
            ),
        }


def evaluate_reference(
    *,
    data_path: Path | None = None,
    limit: int | None = None,
    top_k: int | None = None,
    with_vector: bool = True,
    compare_baseline: bool = True,
    verbose: bool = True,
) -> ReferenceReport:
    """跑「题面点名了某一条」那批题，看规则取条能不能唯一定位到它。

    与 `evaluate()` 的分工：那个测**检索**（hit@k / MRR），这个测**定位**。
    两者测不到同一件事 —— 评测集那 135 题里含「第…条」的是 0 道，所以
    检索指标在结构上永远衡量不到本轮新增的能力，这一臂才是它的探针。

    `compare_baseline=True` 时额外跑一遍基线检索做对照；Milvus 不可用时
    如实跳过基线那一列，**不**让整个评测失败（离线部分本来就跑得完）。
    """
    from ..agent.tools import build_article_index, find_article
    from ..kb.chunker import ChunkStage

    data_path = Path(data_path or config.EVAL_CORPUS_PATH)
    if not data_path.exists():
        from ..api import QaError

        raise QaError(f"评测语料不存在：{data_path}")

    resolver, _known, id_of = _kb_index()
    cases, no_gold, _leaked = build_cases(data_path, resolver, id_of, include_leaked=True)
    # 只留「题面真的点名了某一条」的题：既没条号又没法名的那些，题面给不出可定位的目标，
    # 拿它们算分母会把定位率稀释成一个没有意义的数。
    cases = [c for c in cases if RE_ARTICLE.search(c.question) or RE_LAW.search(c.question)]
    if limit:
        cases = cases[:limit]
    if not cases:
        from ..api import QaError

        raise QaError(f"{data_path} 里没有「题面含条号或法名」的题")

    parents = ChunkStage(verbose=False).load().parent_map()
    index = build_article_index(parents)
    citation_of = {p.parent_id: p.citation for p in parents.values()}

    baseline: dict[str, int | None] = {}
    baseline_ran = False
    if compare_baseline:
        try:
            from ..api import qa

            top_k = top_k or max(KS)
            for item in cases:
                # mode="search" 回的是 RetrievalResult（不是 Answer）—— 只检索、不花 LLM 的钱。
                # `qa` 的注明类型是 `Answer | RetrievalResult`，这里断言一下把它收窄，
                # 顺便把「mode 传错就会拿到没有 .articles 的对象」这件事变成一句明确的话。
                retrieval = qa(
                    item.question, mode="search", top_k=top_k, with_vector=with_vector, debug=False
                )
                assert isinstance(retrieval, RetrievalResult), "mode='search' 应返回检索结果"
                hits = tuple(hit.article.parent_id for hit in retrieval.articles)
                baseline[item.question] = next(
                    (i for i, pid in enumerate(hits, start=1) if pid in item.gold_ids), None
                )
            baseline_ran = True
        except Exception as exc:  # noqa: BLE001 - Milvus 没起不该让离线部分也失败
            print(f"[eval] 基线对照跳过（{type(exc).__name__}：{exc}）", flush=True)

    started = time.perf_counter()
    results: list[ReferenceCase] = []
    for position, item in enumerate(cases, start=1):
        hit = find_article(item.question, parents=parents, index=index)
        located = citation_of.get(hit.parent_id) if hit is not None else None
        results.append(
            ReferenceCase(
                question=item.question,
                gold_citations=item.gold_citations,
                located=located,
                located_is_gold=hit is not None and hit.parent_id in item.gold_ids,
                baseline_rank=baseline.get(item.question),
            )
        )
        if verbose and (position % 25 == 0 or position == len(cases)):
            mark = "取条" if located else "回退"
            print(f"[eval] {position:>4}/{len(cases)}  {mark}  {item.question[:36]}", flush=True)

    return ReferenceReport(
        results=tuple(results),
        total_raw=len(json.loads(data_path.read_text(encoding="utf-8"))),
        skipped_no_gold=no_gold,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        baseline_ran=baseline_ran,
    )


# ------------------------------------------------------------------ 命令行
USAGE = __doc__


def main(argv: list[str] | None = None) -> int:
    import sys

    from ..api import QaError

    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0

    def option(name: str, cast: Callable[[str], Any] = int) -> Any:
        """取 `--name value` 里的 value 并按 `cast` 转一下；没给这个开关就返回 None。

        返回 `Any` 是有意的：这一个 helper 同时服务 `--limit`（int）与 `--data`（str），
        用联合类型标注只会让每个调用点都要再窄化一次。
        """
        if name not in args or args.index(name) + 1 >= len(args):
            return None
        return cast(args[args.index(name) + 1])

    def data_path() -> Path | None:
        raw = option("--data", str)
        return Path(raw) if raw else None

    try:
        if "--reference" in args:
            # 规则取条臂：测的是**定位**，不是检索。默认那一支的行为逐字节不变。
            reference = evaluate_reference(
                data_path=data_path(),
                limit=option("--limit"),
                top_k=option("--top-k"),
                with_vector="--no-vector" not in args,
                compare_baseline="--no-compare" not in args,
                verbose="--quiet" not in args,
            )
            print()
            if "--json" in args:
                print(json.dumps(reference.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(reference.render())
            return 0

        report = evaluate(
            data_path=data_path(),
            limit=option("--limit"),
            top_k=option("--top-k"),
            with_vector="--no-vector" not in args,
            verbose="--quiet" not in args,
        )
    except QaError as exc:
        print(str(exc))
        return 1

    if "--json" in args:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(report.render(show_misses=option("--show-misses") or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
