"""离线检索评测：data/eval_corpus.json（指令语料）→ hit@1/@3/@6 + MRR。

    python -m traffic_law_rag.eval                    # 跑全部可用评测题（域内外分开报）
    python -m traffic_law_rag.eval --in-domain        # 只跑域内 135 题，快一半
    python -m traffic_law_rag.eval --limit 20         # 先跑 20 条看链路
    python -m traffic_law_rag.eval --no-vector        # 只走 BM25，用来 A/B 对比混合检索
    python -m traffic_law_rag.eval --show-misses 10   # 打印没命中的题，便于定位
    python -m traffic_law_rag.eval --json             # 机器可读

语料本身不是评测集，779 条 instruction/output 里只有一部分能当检索题用：

- **298 条**的答案里没有能定位到本库的条号 → 构造不出 ground truth（多为库外法规）
- **213 条**的题面自己就写着「第X条」或《法名》→ 答案泄漏，检索必然"命中"，测不出东西
- 剩下 **268 条**既能定位 gold、题面又不泄漏 → 本模块跑这些

这 268 条还得分域内外**分开报**，否则数字没法看：

- **域内 135 条**是法条问答，本库能覆盖 → hit@1 75.6% / hit@3 85.2% / MRR 0.805
- **域外 133 条**是美国自动驾驶事故叙述（Waymo / Zoox / Nuro 在旧金山…），
  本库是中国交通法规，根本覆盖不了；它们的 gold 是硬凑的，检索一律返回
  深圳条例第五十三条反而是合理行为 → hit@3 仅 4.5%

域外占了整整一半，混在一起会把 hit@3 从 85.2% 拉到 45.1%，让人误判成"检索很差"。
判据：题面以「事故叙述」开头，或含连续英文专名（≥3 个字母）。

ground truth 取答案里引用的**全部**本库条号，不是只取第一条：一条答案合法引用
多条是常态，只认第一条会低估命中率。

每条都用 `qa(..., mode="search")` 跑，所以测的就是对外那个接口本身。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .contracts import RetrievalResult

__all__ = ["main", "evaluate", "build_cases", "EvalCase", "EvalReport"]

RE_LAW = re.compile(r"《([^》]{2,40})》")
RE_ARTICLE = re.compile(r"第[零一二三四五六七八九十百千]+条")
RE_LATIN = re.compile(r"[A-Za-z]{3,}")
# 条号必须紧跟在法名后面才算数，隔太远就不敢认了（正文里可能提到别的法规）
CITE_WINDOW = 50
KS = (1, 3, 6)
OUT_OF_DOMAIN_PREFIX = "事故叙述"


def _is_out_of_domain(question: str) -> bool:
    """题面是否超出本库覆盖范围（判据见模块 docstring）。"""
    return question.startswith(OUT_OF_DOMAIN_PREFIX) or bool(RE_LATIN.search(question))


@dataclass(frozen=True)
class EvalCase:
    """一条评测题：问题 + 期望命中的法条（parent_id）。"""

    question: str
    gold_ids: tuple[str, ...]
    gold_citations: tuple[str, ...]   # 人读，用于打印
    gold_laws: tuple[str, ...]        # 去重后的法规名，用于分组统计
    in_domain: bool                   # False = 域外题（本库覆盖不了，单独统计）


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

    def in_domain(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.case.in_domain)

    def out_of_domain(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.case.in_domain)

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
        """法规 → （命中 top-3 的题数, 该法规总题数）；只统计域内题。"""
        grouped: dict[str, list[bool]] = {}
        for item in self.in_domain():
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
            "in_domain": metrics(self.in_domain()),
            "out_of_domain": metrics(self.out_of_domain()),
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
        out = self.out_of_domain()
        lines = [
            f"检索评测：{self.cases} 题（域内 {len(self.in_domain())} / 域外 {len(out)}）"
            f" ｜ {mode} ｜ top_k={self.top_k} ｜ 耗时 {self.elapsed_ms / 1000:.1f}s",
            f"来源：{self.total_raw} 条语料 → 剔除 {self.skipped_no_gold} 条无 gold、"
            f"{self.skipped_leak} 条题面泄漏",
            "",
            self._line("域内（法条问答）", self.in_domain()),
        ]
        if out:
            lines.append(self._line("域外（本库覆盖不了）", out))
        lines += [
            self._line("合计", self.results),
            "",
            "  按法规（hit@3，仅域内）：",
        ]
        for law, (hit, total) in self.by_law().items():
            lines.append(f"    {hit / total:>6.1%}  {hit:>4}/{total:<4}  {law}")

        if show_misses:
            missed = [r for r in self.in_domain() if not r.hit]
            lines.append("")
            lines.append(f"  域内未命中 {len(missed)} 题，前 {min(show_misses, len(missed))} 条：")
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
) -> tuple[list[EvalCase], int, int]:
    """语料 → 评测题；返回（题目, 无 gold 条数, 题面泄漏条数）。"""
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
            continue

        gold_pairs = list(dict.fromkeys(pairs))  # 去重保序
        cases.append(
            EvalCase(
                question=question,
                gold_ids=tuple(id_of[(law, art)] for law, art in gold_pairs),
                gold_citations=tuple(f"《{law}》{art}" for law, art in gold_pairs),
                gold_laws=tuple(dict.fromkeys(law for law, _ in gold_pairs)),
                in_domain=not _is_out_of_domain(question),
            )
        )
    return cases, no_gold, leaked


def _kb_index() -> tuple[LawResolver, set[tuple[str, str]], dict[tuple[str, str], str]]:
    from .chunker import ChunkStage

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
    in_domain_only: bool = False,
    verbose: bool = True,
) -> EvalReport:
    """逐题跑检索，统计 hit@k 与 MRR。in_domain_only 只跑本库覆盖得了的题。"""
    from .api import qa

    data_path = Path(data_path or config.EVAL_CORPUS_PATH)
    if not data_path.exists():
        from .api import QaError

        raise QaError(f"评测语料不存在：{data_path}")

    resolver, _known, id_of = _kb_index()
    cases, no_gold, leaked = build_cases(data_path, resolver, id_of)
    if in_domain_only:
        cases = [case for case in cases if case.in_domain]
    if limit:
        cases = cases[:limit]
    if not cases:
        from .api import QaError

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


# ------------------------------------------------------------------ 命令行
USAGE = __doc__


def main(argv: list[str] | None = None) -> int:
    import sys

    from .api import QaError

    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0

    def option(name: str, cast=int):
        return cast(args[args.index(name) + 1]) if name in args and args.index(name) + 1 < len(args) else None

    try:
        report = evaluate(
            data_path=Path(option("--data", str)) if option("--data", str) else None,
            limit=option("--limit"),
            top_k=option("--top-k"),
            with_vector="--no-vector" not in args,
            in_domain_only="--in-domain" in args,
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
