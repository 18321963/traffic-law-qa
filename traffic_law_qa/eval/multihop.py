from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config
from ..agents.trace import render_trace
from ..contracts import ParentChunk
from ..services.llm import ToolCallingLLM
from ..tools.articles import build_article_index, parse_article_no, resolve_law_id
from .corpus import RE_ARTICLE, RE_LAW

USAGE = """跨法规多跳题集：从条文反向造题 + 机械护栏。

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
    pass


def _save(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def format_key(law_id: str, article_no: str) -> str:
    return f"{law_id}{SEP}{article_no}"


def parse_key(raw: str) -> tuple[str, str]:
    law_id, sep, article_no = (raw or "").partition(SEP)
    if not sep or not law_id.strip() or not article_no.strip():
        raise HopError(f"引用格式应为 `law_id{SEP}条号`，收到：{raw!r}")
    return law_id.strip(), article_no.strip()


@dataclass(frozen=True, eq=False)
class Library:

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
        from ..kb.chunker import ChunkStage

        return cls.from_parents(list(ChunkStage(verbose=False).load().parents))

    def resolve(self, raw: str) -> ParentChunk:
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
    return bool(RE_PREAMBLE.search(parent.text))


def _law_aliases(library: Library) -> tuple[str, ...]:
    aliases: set[str] = set()
    for name in library.law_names:
        aliases.add(name)
        short = name.removeprefix("中华人民共和国")
        if len(short) >= 4:
            aliases.add(short)
    return tuple(sorted(aliases, key=len, reverse=True))


def check_question(question: str, library: Library) -> None:
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
    return re.sub(r"[^\w一-鿿]+", "", question)


def load_cases(
    path: Path | None = None, *, library: Library | None = None
) -> tuple[list[HopCase], int]:
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
    if not RE_APPLY.search(parent.text):
        return False
    rest = "".join(s for s in re.split(r"[。；]", parent.text) if s and not RE_APPLY.search(s))
    return len(rest) < 20


def defect_diagnostic(cases: list[HopCase], library: Library) -> str:
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
    sites: list[ParentChunk] = []
    for parent in library.parents.values():
        for match in RE_LAW.finditer(parent.text):
            law_id, _all_names = resolve_law_id(match.group(1), library.parents)
            if law_id is not None and law_id != parent.law_id:
                sites.append(parent)
                break
    return sites


def _candidates(library: Library) -> list[ParentChunk]:
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
    return getattr(case, "why", "")


def trace(
    *,
    cases: Sequence[Any] | None = None,
    limit: int | None = 10,
    out: Path | None = None,
    top_k: int = 6,
    verbose: bool = True,
) -> list[dict]:
    from ..agents.graph import AgentRunner
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
                    "region": state.get("region"),
                    "region_scope": list(state.get("region_scope") or ()),
                    "searches": state.get("search_log") or [],
                    "usage": state.get("usage") or [],
                    "trace": render_trace(state),
                    "review": (
                        None
                        if agent_answer is None or agent_answer.review is None
                        else agent_answer.review.to_dict()
                    ),
                },
            }
        )

        if out is not None:
            _save(out, rows)

    runner.tracer.flush()

    if out is not None:
        _save(out, rows)
        if verbose:
            print(f"\n[trace] → {out}")
    return rows


def full_budget_summary(rows: list[dict]) -> str:
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


def _parser() -> argparse.ArgumentParser:
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
