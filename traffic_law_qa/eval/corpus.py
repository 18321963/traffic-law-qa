"""题集文件的唯一生成者与读者：一份源语料 → 三份桶文件。

    python -m traffic_law_qa.eval.corpus build    # 从源语料重建三份桶文件
    python -m traffic_law_qa.eval.corpus check    # 只读：内存里重算一遍，逐字节比对

**为什么要有这个模块。** 三桶原先是在 `harness.build_cases` 里每次跑评测现算的，
磁盘上只有一份源语料 —— 想知道「这套语料分几类题」必须去读代码或跑一次。
现在一个桶一个文件，名字即用途：

    data/eval_retrieval.json   82 道  题面不带定位信息、又能定位 gold → 检索指标
    data/eval_reference.json  112 道  题面自带条号或《法名》 → 规则取条那一臂的探针
    data/eval_nogold.json      45 道  构造不出 ground truth → 存档，每道带一句成因

（第四份 `data/eval_multihop.json`（63 道跨法多跳）是从条文反向造出来的，
不归这里管，`--generate` 仍写它。）

**源语料 `data/eval_corpus.json`** 是 475 条 `instruction/output` 记录，每道题**存了两遍**
（同一句题面、两份不同的 `output` —— 两次生成的答案），所以往下一切筛选与计数都按
**题面**走：先把两个变体合并成一道、再分类。不合并的话同一句检索词会被算两遍、
分母虚高四成，而两次生成引的条不一致时还会拿两套互相矛盾的标准去判同一次检索。

合并规则：gold 取两次生成所引**本库**条号的**交集** —— 只认两次都引了的条。交集为空
就是「无 gold」，它有两个成因，`why` 字段如实记下是哪一个：答案里没引本库条号、
或两次生成引的条不一致（后者把两组条都列出来，便于人复查）。

题面自带条号或《法名》的题**不进检索题集**：查询里已经写明要哪一条，BM25 必然命中，
衡量不出检索能力。它们改当规则取条那一臂的题集 —— 这是同一批题换一个用途，
不是两份复制品。其中写了条号的 109 道里 107 道那个条号就是 gold（另 3 道只写法名）。

**gold 写 `law_id#条号`，不写 parent_id**：parent_id 形如
`road_traffic_safety_law@2021-04-29#a001`，是切块的产物、重切一次就变；条号才是法规的
身份。形状与 `data/eval_multihop.json` 一致，加载时经 `multihop.Library` 换回条文。

**源语料是删过的**：原 779 条里有 296 条美国自动驾驶事故叙述（Waymo / Cruise / Zoox
在旧金山、洛杉矶的碰撞叙述），本库是中国交通法规，一条也覆盖不了，而且它们的 gold
是硬凑的 —— 检索一律返回深圳条例第五十三条反而是合理行为，域外 hit@3 仅 4.5%，
混进合计会把 hit@3 从 85% 拉到 45%。另删 8 条英文版深圳条例题（跨语言检索不是本评测
要测的东西）。两类合起来 304 条，2026-09 一次性删净。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..contracts import ParentChunk

__all__ = [
    "RE_LAW",
    "RE_ARTICLE",
    "CITE_WINDOW",
    "EvalCase",
    "BucketError",
    "LawResolver",
    "build_buckets",
    "check_buckets",
    "load_bucket",
    "display_path",
    "main",
]

RE_LAW = re.compile(r"《([^》]{2,40})》")
RE_ARTICLE = re.compile(r"第[零一二三四五六七八九十百千]+条")
CITE_WINDOW = 50

RETRIEVAL = "retrieval"
REFERENCE = "reference"
NOGOLD = "nogold"

BUCKET_PATHS: dict[str, Path] = {
    RETRIEVAL: config.EVAL_RETRIEVAL_PATH,
    REFERENCE: config.EVAL_REFERENCE_PATH,
    NOGOLD: config.EVAL_NOGOLD_PATH,
}

NO_CITATION = "答案里没引本库条号"


class BucketError(ValueError):
    """桶文件坏了，或者拿源语料当桶文件读。"""


@dataclass(frozen=True)
class EvalCase:
    """一条评测题：问题 + 期望命中的法条（parent_id）。"""

    question: str
    gold_ids: tuple[str, ...]
    gold_citations: tuple[str, ...]
    gold_laws: tuple[str, ...]


def display_path(path: Path) -> str:
    """能缩到仓库内就缩（`data/eval_retrieval.json`），否则原样给全路径。"""
    try:
        return path.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def _library():
    """切块产物 → `multihop.Library`（本条路径唯一的库视图）。

    **这一句只能在函数内 import**：`multihop` 在模块级就 `from .corpus import
    RE_ARTICLE, RE_LAW`，两边都提到模块级就是双向循环 —— 先 import 哪一边都炸在
    「半初始化的模块」上，两个方向都实测过。`load_bucket` 里那句 `HopError` 同理。
    """
    from .multihop import Library

    return Library.load()


def _law_maps(library) -> tuple[LawResolver, dict[tuple[str, str], ParentChunk]]:
    """`(法名, 条号)` 的两张表：解析答案里写的法名、换回条文本身。"""
    parents = list(library.parents.values())
    resolver = LawResolver(sorted({p.law_name for p in parents}))
    return resolver, {(p.law_name, p.article_no): p for p in parents}


def _group(raw: list, resolver: LawResolver, known: set[tuple[str, str]]) -> dict[str, list[list]]:
    """题面 → 每个变体引到的本库条号（顺序按它在语料里第一次出现）。"""
    groups: dict[str, list[list[tuple[str, str]]]] = {}
    for item in raw:
        question = (item.get("instruction") or "").strip()
        if not question:
            continue
        pairs = [
            (law, art)
            for law, art in _cited_articles(item.get("output") or "", resolver)
            if (law, art) in known
        ]
        groups.setdefault(question, []).append(list(dict.fromkeys(pairs)))
    return groups


def _classify(variants: list[list[tuple[str, str]]]) -> tuple[list[tuple[str, str]], str]:
    """一个题面的全部变体 → （gold 条号, 无 gold 时的成因）。"""
    with_gold = [pairs for pairs in variants if pairs]
    if not with_gold:
        return [], NO_CITATION
    gold = [pair for pair in with_gold[0] if all(pair in pairs for pairs in with_gold)]
    if gold:
        return gold, ""
    seen = ["、".join(f"《{law}》{art}" for law, art in pairs) for pairs in with_gold]
    return [], "两次生成引的条不一致：" + " vs ".join(seen)


def build_buckets(source_path: Path | None = None) -> dict[str, list[dict]]:
    """源语料 → 三份桶的内容。纯计算，不落盘（`check` 靠这一点重算比对）。"""
    library = _library()
    resolver, parent_of = _law_maps(library)
    raw = json.loads(Path(source_path or config.EVAL_CORPUS_PATH).read_text(encoding="utf-8"))
    groups = _group(raw, resolver, set(parent_of))

    buckets: dict[str, list[dict]] = {RETRIEVAL: [], REFERENCE: [], NOGOLD: []}
    for question, variants in groups.items():
        gold, why = _classify(variants)
        if not gold:
            buckets[NOGOLD].append({"question": question, "gold": [], "why": why})
            continue
        named = RE_ARTICLE.search(question) or RE_LAW.search(question)
        bucket = REFERENCE if named else RETRIEVAL
        keys = [f"{parent_of[pair].law_id}#{parent_of[pair].article_no}" for pair in gold]
        buckets[bucket].append({"question": question, "gold": list(dict.fromkeys(keys))})
    return buckets


def _dumps(payload: list[dict]) -> str:
    """桶文件的序列化。build 与 check 共用这一个函数，逐字节比才可能成立。

    不带时间戳：同一份源语料必须产出同一串字节，否则 git 里天天在改、
    `check` 也永远是红的。
    """
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _drift(path: Path, fresh: list[dict]) -> str:
    """不一致的第一处是什么 —— 只说「对不上」等于没说。"""
    if not path.exists():
        return f"{display_path(path)} 不存在"
    have = json.loads(path.read_text(encoding="utf-8"))
    if len(have) != len(fresh):
        return f"{display_path(path)} 有 {len(have)} 条，重算是 {len(fresh)} 条"
    if have != fresh:
        for index, (mine, theirs) in enumerate(zip(fresh, have), start=1):
            if mine != theirs:
                return (
                    f"{display_path(path)} 第 {index} 条起不同："
                    f"重算 {mine.get('question', '')[:24]}… ／ 文件 {theirs.get('question', '')[:24]}…"
                )
    return f"{display_path(path)} 内容一致但字节不同（多半是缩进或换行被改过）"


def check_buckets(source_path: Path | None = None) -> list[str]:
    """重算一遍与磁盘上的桶文件逐字节比对，返回不一致的描述（空 = 一致）。"""
    fresh = build_buckets(source_path)
    drift = []
    for name, path in BUCKET_PATHS.items():
        if path.exists() and path.read_text(encoding="utf-8") == _dumps(fresh[name]):
            continue
        drift.append(f"{name}：{_drift(path, fresh[name])}")
    return drift


def load_bucket(path: Path | None = None) -> list[EvalCase]:
    """桶文件 → 评测题。

    `gold` 里的 `law_id#条号` 在这里换回条文与 `parent_id` —— 指标比的是
    parent_id，而文件里存的是条号（见模块文档：条号才跟着法规走）。
    解析不到就是题集坏了，`Library.resolve` 直接抛，不静默降级成「无 gold」。

    下面那句 import 也只能在函数内（循环 import，理由见 `_library`）。
    """
    from .multihop import HopError

    path = Path(path or config.EVAL_RETRIEVAL_PATH)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise BucketError(f"{display_path(path)} 不是桶文件（顶层应是数组）")
    library = _library()

    cases: list[EvalCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or "question" not in item:
            raise BucketError(
                f"{display_path(path)} 第 {index} 条不像桶文件的记录（缺 question/gold）。"
                f"源语料 {display_path(config.EVAL_CORPUS_PATH)} 是桶文件的**输入**，不是桶文件 —— "
                f"先跑 python -m traffic_law_qa.eval.corpus build"
            )
        question = (item.get("question") or "").strip()
        if not question:
            raise BucketError(f"{display_path(path)} 第 {index} 条题面为空")
        try:
            parents = [library.resolve(key) for key in item.get("gold") or ()]
        except HopError as exc:
            raise BucketError(f"{display_path(path)} 第 {index} 条：{exc}") from None
        cases.append(
            EvalCase(
                question=question,
                gold_ids=tuple(p.parent_id for p in parents),
                gold_citations=tuple(f"《{p.law_name}》{p.article_no}" for p in parents),
                gold_laws=tuple(dict.fromkeys(p.law_name for p in parents)),
            )
        )
    return cases


USAGE = __doc__


def _parser() -> argparse.ArgumentParser:
    """只做校验的解析器（`--help` 与「不带参数」由 `main` 开头那个分支打印 `USAGE`，
    所以 `add_help=False`）。本模块没有默认动作 —— build 会写文件，必须点名要它。"""
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa.eval.corpus", add_help=False)
    parser.add_argument("command", choices=("build", "check"), help="build 重建三份桶文件 / check 只读比对")
    parser.add_argument("--data", default=None, help="换一份源语料（默认 data/eval_corpus.json）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args or not args:
        print(USAGE)
        return 0

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    source = Path(options.data) if options.data else None

    if options.command == "build":
        buckets = build_buckets(source)
        for name, path in BUCKET_PATHS.items():
            path.write_text(_dumps(buckets[name]), encoding="utf-8")
            print(f"[corpus] {name:<9} {len(buckets[name]):>4} 条 → {display_path(path)}")
        return 0

    drift = check_buckets(source)
    if not drift:
        print(f"[corpus] {len(BUCKET_PATHS)} 份桶文件与源语料逐字节一致")
        return 0
    print("[corpus] 桶文件与源语料不一致：")
    for line in drift:
        print(f"  {line}")
    print("  跑 python -m traffic_law_qa.eval.corpus build 重建")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
