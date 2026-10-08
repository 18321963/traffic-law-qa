from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from rag_contracts import config
from rag_contracts.domain.reports import channel_label
from rag_contracts.domain.retrieval import RetrievalResult

from .corpus import BucketError, EvalCase, display_path, load_bucket

USAGE = """离线检索评测：data/eval_retrieval.json → 池子召回 / hit@1@3@6 / 单通道召回。

    python -m eval                    # 跑全部可用评测题（82 道）
    python -m eval --limit 20         # 先跑 20 条看链路
    python -m eval --pool 50          # 换池子大小（默认取 RAG_CANDIDATES）
    python -m eval --no-vector        # 只走 BM25，用来 A/B 对比混合检索
    python -m eval --show-misses 10   # 打印没命中的题，便于定位
    python -m eval --json             # 机器可读
    python -m eval --http asgi        # 默认：本进程里真起一个 app，每题过真 HTTP
    python -m eval --http http://127.0.0.1:8000   # 打别人起的服务

另一条臂（测「条文定位」而不是检索）：

    python -m eval --reference        # 换跑 data/eval_reference.json（112 道）
    python -m eval --reference --no-compare   # 连基线对照都不跑，纯离线

`--http` 那条臂把每题发到 `POST /qa`（`mode=search`、`pool=<池子>`、`debug=true`），
回来的载荷经 `api_contracts.client` 重建后走**同一套统计** —— 量的是「过 HTTP 那条路」
与进程内是否等价。默认的 `asgi` 不需要另起服务：eval 自己 `boot_runtime()` 一个真 app，
用 starlette `TestClient` 走真 ASGI（真路由、真校验、真 `from_dict`，只是不过 socket）；
给 URL 则打那个真服务，这时服务得先起着，`RAG_POOL_MAX` 要放得下 `--pool`。

**同口径的前提，别不看**：端点的 `pool` 同时是候选池与返回深度，进程内那条臂的候选池是
`RAG_CANDIDATES`、`--pool` 只改返回深度。所以两条臂逐题可比的条件是
`--pool` 等于服务端的 `RAG_CANDIDATES`（两边默认都是 20，这时的默认跑法就是同一把尺子）；
`--pool` 给了别的值，两边的池子深度就不一样，读数只能各自看。另外三处本来就不可比：
`asgi` 臂由 eval 自己 boot，用不了 `--no-vector`（两个一起给就落回进程内并打印一行，
但显式指定了 URL 就报错退出 —— 那是明确指定的目标，落回等于无视它）；耗时含网络与服务端排队。

为什么需要第二条臂：题面自带条号或法名的题**进不了**检索题集 —— 查询里已经给了定位信息，
检索必然命中，衡量不出检索能力。所以那 82 道里含「第…条」的是 0 道，检索指标在结构上
永远衡量不到规则取条这条新路径。被剔出去的那批反而是一份现成的、已标注的探针集，
`--reference` 就是拿它来测：其中写了条号的 109 道里 107 道那个条号就是 gold。

**题集不是本模块造的。** 三份桶文件（检索 82 / 点名 112 / 无 gold 45）都由 `eval.corpus`
从源语料 data/eval_corpus.json 切出来，各是什么、gold 怎么取交集，文档在那个模块里。
本模块只读文件 —— 换题集是换文件，不是改代码。

指标口径（四条都影响读数，改口径等于换了一把尺子）：

- 每条都用 `qa(..., mode="search")` 跑，所以测的就是对外那个接口本身
- ground truth 取答案里引用的**全部**本库条号，不是只取第一条：一条答案合法引用多条
  是常态，只认第一条会低估命中率
- 命中 = 期望的 parent_id 出现在返回列表里，名次取**最靠前**的那个；报告里的
  82 道 → hit@1 75.6% / hit@3 86.6% / hit@6 91.5% / MRR@6 0.819、池子召回 95.1%、
  单通道召回 BM25 90.2% / 稠密 93.9%（本地 bge-m3 + 交叉编码器重排）
  是**当前配置的属性**，换向量模型或关掉重排这组数就会动，别当成模型的属性。
- **三段是同一趟读出来的**：检索按池子大小（默认 `RAG_CANDIDATES`）取回整个候选池，
  重排后在 `top_k` 处截断，所以一份返回列表同时给出三段 ——
  `单通道召回 ≤ 池子召回 ≤ hit@6`。池子召回 - 单通道最高那路 = RRF 融合赚的；
  池子召回 - hit@6 = 重排排序丢的。两条边界要说清：
  「单通道召回」是「gold 出现在**池子里**、且这一路给了它名次」的口径 ——
  池子本身是融合后截断的，所以它不是通道各自 top-N 的原始召回；
  MRR 锁在 **@6**，否则池子放大后第 7 名往后的名次会开始贡献 `1/rank`，
  与已发布的那组数不可比。

原先设过的**「域内外分开报」机制已拆除**：域外题清零后，`in_domain` 标记、
`--in-domain` 开关、报告里的域外行全都失去了真实调用，留着就是一份没人执行的契约。
"""

__all__ = ["main", "evaluate", "evaluate_reference", "EvalReport", "ReferenceReport"]

KS = (1, 3, 6)


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    hit_ids: tuple[str, ...]
    hit_citations: tuple[str, ...]
    rank: int | None
    used_vector: bool
    in_dense: bool | None
    in_bm25: bool | None

    def hit_within(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass(frozen=True)
class EvalReport:
    results: tuple[CaseResult, ...]
    source: str
    top_k: int
    pool: int
    used_vector: bool
    elapsed_ms: float

    @property
    def cases(self) -> int:
        return len(self.results)

    def hit_at(self, k: int, subset: tuple[CaseResult, ...] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.hit_within(k)) / len(rows)

    def mrr(self, subset: tuple[CaseResult, ...] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        cutoff = max(KS)
        return sum(1.0 / r.rank for r in rows if r.hit_within(cutoff)) / len(rows)

    def pool_hits(self, subset: tuple[CaseResult, ...] | None = None) -> int:
        rows = self.results if subset is None else subset
        return sum(1 for r in rows if r.hit)

    def pool_recall(self, subset: tuple[CaseResult, ...] | None = None) -> float:
        rows = self.results if subset is None else subset
        if not rows:
            return 0.0
        return self.pool_hits(rows) / len(rows)

    def channel_recall(
        self, channel: str, subset: tuple[CaseResult, ...] | None = None
    ) -> tuple[int, int]:
        rows = self.results if subset is None else subset
        flags = [r.in_dense if channel == "dense" else r.in_bm25 for r in rows]
        known = [flag for flag in flags if flag is not None]
        return sum(1 for flag in known if flag), len(known)

    def _line(self, label: str, rows: tuple[CaseResult, ...]) -> str:
        if not rows:
            return f"  {label}：（无）"
        metrics = "  ".join(f"hit@{k} {self.hit_at(k, rows):>5.1%}" for k in KS)
        return f"  {label}：{metrics} ｜ MRR@6 {self.mrr(rows):.3f}  （{len(rows)} 题）"

    def by_law(self) -> dict[str, tuple[int, int]]:
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
                "pool_recall": round(self.pool_recall(rows), 4),
            }

        bm25_hits, bm25_cases = self.channel_recall("bm25")
        dense_hits, dense_cases = self.channel_recall("dense")

        return {
            "cases": self.cases,
            "source": self.source,
            "top_k": self.top_k,
            "pool": self.pool,
            "used_vector": self.used_vector,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "overall": metrics(self.results),
            "channel_recall": {
                "bm25": {"hits": bm25_hits, "cases": bm25_cases},
                "dense": (
                    {"hits": dense_hits, "cases": dense_cases} if dense_cases else None
                ),
            },
            "by_law": {
                law: {"hit3": hit, "cases": total} for law, (hit, total) in self.by_law().items()
            },
        }

    @property
    def vector_label(self) -> str:
        used = sum(1 for row in self.results if row.used_vector)
        if used == 0:
            return channel_label(False)
        if used == len(self.results):
            return channel_label(True)
        return f"部分降级（{used}/{len(self.results)} 题走稠密）"

    def render(self, *, show_misses: int = 0) -> str:
        mode = self.vector_label
        bm25_hits, bm25_cases = self.channel_recall("bm25")
        dense_hits, dense_cases = self.channel_recall("dense")
        dense_line = (
            f"稠密 {dense_hits}/{dense_cases}（{dense_hits / dense_cases:.1%}）"
            if dense_cases
            else "稠密 未开"
        )
        lines = [
            f"检索评测：{self.cases} 题"
            f" ｜ {mode} ｜ 池子 {self.pool} 截到 top_k={self.top_k}"
            f" ｜ 耗时 {self.elapsed_ms / 1000:.1f}s",
            f"题集：{self.source}",
            "",
            self._line("合计", self.results),
            f"  池子召回：{self.pool_recall():>5.1%}"
            f"  （{self.pool_hits()}/{self.cases} 题进了候选池：检索+融合捞到的上界）",
            f"  单通道召回：BM25 {bm25_hits}/{bm25_cases}"
            f"（{bm25_hits / max(1, bm25_cases):.1%}） ｜ {dense_line}"
            "（池子里的 gold 有没有拿到这一路的名次）",
            "",
            "  按法规（hit@3）：",
        ]
        for law, (hit, total) in self.by_law().items():
            lines.append(f"    {hit / total:>6.1%}  {hit:>4}/{total:<4}  {law}")

        if show_misses:
            missed = [r for r in self.results if not r.hit_within(max(KS))]
            lines.append("")
            lines.append(f"  未命中 {len(missed)} 题，前 {min(show_misses, len(missed))} 条：")
            for item in missed[:show_misses]:
                lines.append(f"    Q: {item.case.question[:56]}")
                lines.append(f"       期望：{'、'.join(item.case.gold_citations)}")
                got = "、".join(item.hit_citations[:3]) or "（空）"
                lines.append(f"       实得：{got}")
        return "\n".join(lines)


ASGI = "asgi"


def _remote(http: str | None):
    if not http:
        return None

    from api_contracts.client import RagClient

    if http != ASGI:
        return RagClient(http)

    from fastapi.testclient import TestClient

    from rag_service.adapters.sqlite import init_database
    from rag_service.api import app as server
    from rag_service.api.runtime import boot_runtime

    init_database()
    server.app.state.rt = boot_runtime()
    server.app.state.boot_error = None
    return RagClient("http://asgi", client=TestClient(server.app))


def _search(
    question: str, *, top_k: int, with_vector: bool, debug: bool, remote=None
) -> RetrievalResult:
    if remote is not None:
        from api_contracts.client import to_retrieval

        return to_retrieval(remote.qa(question, mode="search", pool=top_k, debug=debug))

    from rag_service.api.facade import qa

    result = qa(question, mode="search", top_k=top_k, with_vector=with_vector, debug=debug)
    assert isinstance(result, RetrievalResult), "mode='search' 应返回检索结果"
    return result


def evaluate(
    *,
    data_path: Path | None = None,
    limit: int | None = None,
    pool: int | None = None,
    with_vector: bool = True,
    verbose: bool = True,
    http: str | None = None,
) -> EvalReport:
    data_path = Path(data_path or config.EVAL_RETRIEVAL_PATH)
    if not data_path.exists():
        from rag_service.api.facade import QaError

        raise QaError(f"题集不存在：{data_path}（先跑 python -m eval.corpus build）")

    cases = load_bucket(data_path)
    if limit:
        cases = cases[:limit]
    if not cases:
        from rag_service.api.facade import QaError

        raise QaError(f"{display_path(data_path)} 里没有题")

    pool = pool or config.retrieve_config().candidates
    cutoff = max(KS)
    remote = _remote(http)
    started = time.perf_counter()
    results: list[CaseResult] = []

    for index, case in enumerate(cases, start=1):
        result = _search(
            case.question, top_k=pool, with_vector=with_vector, debug=True, remote=remote
        )
        hits = tuple(hit.article.parent_id for hit in result.articles)
        rank = next((i for i, parent_id in enumerate(hits, start=1) if parent_id in case.gold_ids), None)
        gold = [hit for hit in result.articles if hit.article.parent_id in case.gold_ids]
        results.append(
            CaseResult(
                case=case,
                hit_ids=hits,
                hit_citations=tuple(hit.citation for hit in result.articles),
                rank=rank,
                used_vector=result.used_vector,
                in_dense=(
                    any(hit.vector_rank is not None for hit in gold) if result.used_vector else None
                ),
                in_bm25=(
                    any(hit.bm25_rank is not None for hit in gold) if result.used_bm25 else None
                ),
            )
        )
        if verbose and (index % 10 == 0 or index == len(cases)):
            marked = "ok" if rank is not None and rank <= cutoff else ("pool" if rank else "miss")
            print(f"[eval] {index:>4}/{len(cases)}  {marked:<4} {case.question[:36]}", flush=True)

    return EvalReport(
        results=tuple(results),
        source=display_path(data_path),
        top_k=cutoff,
        pool=pool,
        used_vector=any(row.used_vector for row in results),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _verdict(rule_hits: int, base_hits: int) -> str:
    delta = rule_hits - base_hits
    if delta > 0:
        return f"这一臂比基线 hit@1 **多** {delta} 题"
    if delta < 0:
        return f"这一臂比基线 hit@1 **少** {-delta} 题"
    return f"这一臂与基线 hit@1 **打平**（各 {rule_hits} 题）"


@dataclass(frozen=True)
class ReferenceCase:

    question: str
    gold_citations: tuple[str, ...]
    located: str | None
    located_is_gold: bool
    baseline_rank: int | None


@dataclass(frozen=True)
class ReferenceReport:

    results: tuple[ReferenceCase, ...]
    source: str
    elapsed_ms: float
    baseline_ran: bool = False

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
        return tuple(r for r in self.results if r.located is not None and not r.located_is_gold)

    def baseline_hit_at(self, k: int) -> float:
        if not self.baseline_ran or not self.results:
            return 0.0
        return sum(
            1 for r in self.results if r.baseline_rank is not None and r.baseline_rank <= k
        ) / len(self.results)

    def baseline_hits(self, k: int) -> int:
        return sum(1 for r in self.results if r.baseline_rank is not None and r.baseline_rank <= k)

    @property
    def rule_only(self) -> tuple[ReferenceCase, ...]:
        return tuple(r for r in self.results if r.located_is_gold and r.baseline_rank != 1)

    @property
    def baseline_only(self) -> tuple[ReferenceCase, ...]:
        return tuple(r for r in self.results if r.located is None and r.baseline_rank == 1)

    @property
    def combined_hits(self) -> int:
        return len(self.correct) + len(self.baseline_only)

    def render(self) -> str:
        total = self.cases
        lines = [
            f"规则取条评测：{total} 题（题面含条号或法名的「点名桶」）"
            f" ｜ 离线 ｜ 耗时 {self.elapsed_ms / 1000:.1f}s",
            f"题集：{self.source}",
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
            f"  ── 结论：{_verdict(len(self.correct), self.baseline_hits(1))} ──",
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
            "source": self.source,
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
    http: str | None = None,
) -> ReferenceReport:
    from rag_service.adapters.milvus import MilvusError
    from rag_service.api.facade import QaError
    from rag_service.indexing.chunker import ChunkStage
    from rag_service.query.articles import build_article_index, find_article

    data_path = Path(data_path or config.EVAL_REFERENCE_PATH)
    if not data_path.exists():
        from rag_service.api.facade import QaError

        raise QaError(f"题集不存在：{data_path}（先跑 python -m eval.corpus build）")

    cases = load_bucket(data_path)
    if limit:
        cases = cases[:limit]
    if not cases:
        from rag_service.api.facade import QaError

        raise QaError(f"{display_path(data_path)} 里没有题")

    parents = ChunkStage(verbose=False).load().parent_map()
    index = build_article_index(parents)
    citation_of = {p.parent_id: p.citation for p in parents.values()}

    remote = _remote(http)
    baseline: dict[str, int | None] = {}
    baseline_ran = False
    if compare_baseline:
        try:
            top_k = top_k or max(KS)
            for item in cases:
                retrieval = _search(
                    item.question, top_k=top_k, with_vector=with_vector, debug=False, remote=remote
                )
                hits = tuple(hit.article.parent_id for hit in retrieval.articles)
                baseline[item.question] = next(
                    (i for i, pid in enumerate(hits, start=1) if pid in item.gold_ids), None
                )
            baseline_ran = True
        except (QaError, MilvusError) as exc:
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
        source=display_path(data_path),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        baseline_ran=baseline_ran,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval", add_help=False)
    parser.add_argument("--reference", action="store_true", help="规则取条臂：测定位，不是检索")
    parser.add_argument("--no-compare", action="store_true", help="连基线对照都不跑")
    parser.add_argument("--no-vector", action="store_true", help="只走 BM25，做 A/B 对照")
    parser.add_argument("--quiet", action="store_true", help="不打印逐题进度")
    parser.add_argument("--json", action="store_true", help="输出机器可读的报告")
    parser.add_argument(
        "--data",
        default=None,
        help="换一份题集文件（默认 data/eval_retrieval.json；--reference 时默认 data/eval_reference.json）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 道")
    parser.add_argument("--pool", type=int, default=None, help="覆盖池子大小（默认取 RAG_CANDIDATES）")
    parser.add_argument("--top-k", type=int, default=None, help="--reference 的基线召回深度（默认 6）")
    parser.add_argument("--show-misses", type=int, default=None, help="打印 N 道没命中的题")
    parser.add_argument(
        "--http",
        default=ASGI,
        help=f"改走服务端点：{ASGI}（默认，本进程里真起一个 app 过真 ASGI）或 URL（打真服务）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    from rag_service.api.facade import QaError

    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if options.no_vector and options.http == ASGI:
        print("--no-vector 与 asgi 臂不搭，落回进程内（同一个 LegalRAG，只是不过 HTTP）")
        options.http = None
    elif options.no_vector and options.http:
        print("--http 臂上没有 --no-vector：POST /qa 没这个开关，只走 BM25 得用进程内那条臂")
        return 1
    elif options.http:
        print(
            f"[eval] HTTP 臂：{options.http}"
            + ("（本进程里真起一个 app）" if options.http == ASGI else "（打真服务）"),
            flush=True,
        )

    data_path = Path(options.data) if options.data else None

    try:
        if options.reference:
            reference = evaluate_reference(
                data_path=data_path,
                limit=options.limit,
                top_k=options.top_k,
                with_vector=not options.no_vector,
                compare_baseline=not options.no_compare,
                verbose=not options.quiet,
                http=options.http,
            )
            print()
            if options.json:
                print(json.dumps(reference.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(reference.render())
            return 0

        report = evaluate(
            data_path=data_path,
            limit=options.limit,
            pool=options.pool,
            with_vector=not options.no_vector,
            verbose=not options.quiet,
            http=options.http,
        )
    except (QaError, BucketError) as exc:
        print(str(exc))
        return 1

    if options.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(report.render(show_misses=options.show_misses or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit("已收口：请用 python -m eval（清单见 README「入口」）")
