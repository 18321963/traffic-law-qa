"""法名 / 条号 → 库内那一条。

分两半，共用同一套解析：

1. **落位**：`resolve_law_id`（法名 → law_id）、`parse_article_no`（条号 → int）、
   `build_article_index`（`(law_id, 条号) → 父块`）、`find_article`（从一句自然语言里
   抽出条号并定位到唯一一条）。
2. **取条**：`lookup_article` —— get_article 工具的执行体，落位失败时返回一句给模型
   看的中文说明而不是抛异常。

两个调用方各用一半：工具轮（`nodes.py`）走 `lookup_article`，评测的规则取条探针
（`eval/harness.py`）走 `find_article` —— 它们共用下面这套解析，所以拆成两个文件
只会把共用底座再切一刀。
"""

from __future__ import annotations

import re

from ...contracts import ParentChunk, RetrievalResult, RetrievedArticle
from ...kb.law_parser import cn_to_int


def resolve_law_id(
    law_name: str, parents: dict[str, ParentChunk]
) -> tuple[str | None, list[str]]:
    """法名 → law_id；返回 (law_id, 库内全部法名)。

    三级匹配，从严到宽，**每一级内部唯一才采纳**（多命中即歧义 → None，
    宁可让模型重说一次，也不要猜错法规）：

    1. 精确匹配全名
    2. **后缀匹配** —— 这条专治「道路交通安全法」这类通用简称
    3. 子串匹配 —— 兜住「深圳条例」这类更随意的说法

    第 2 级不能省。法规的通用简称就是全名去掉「中华人民共和国」前缀，
    而「中华人民共和国道路交通安全法」**同时**是「…安全法」和「…安全法实施条例」
    的（非后缀）子串 —— 只靠子串匹配，这部最常被引用的法反而会判成歧义。
    后缀命中说明匹配到了名字的**末尾**，而不是撞进了某个更长名字的中间，这才分得开。

    第二个返回值是**库内全部法名**：解析不到时调用方会把它写进工具结果，
    模型下一轮就能自己改对。这顺带白捡了「让 Agent 知道库边界 → 抑制库外法名幻觉」
    的能力，不用为此新增一个 list_laws 工具。

    **不引 eval.corpus.LawResolver**：`parents` 里本来就躺着全部法规的 law_name 与 law_id，
    引评测模块只会把 dev 工具拖进 agent 热路径。
    """
    table = {p.law_name: p.law_id for p in parents.values()}
    names = sorted(table)
    name = (law_name or "").strip()
    if not name:
        return None, names

    if name in table:
        return table[name], names

    suffixed = [n for n in names if n.endswith(name)]
    if len(suffixed) == 1:
        return table[suffixed[0]], names

    contained = [n for n in names if name in n or n in name]
    if len(contained) == 1:
        return table[contained[0]], names
    return None, names


RE_ARTICLE_CN = re.compile(r"第\s*([零一二三四五六七八九十百千]+)\s*条")
RE_ARTICLE_AR = re.compile(r"第\s*(\d{1,4})\s*条")
RE_LAW_TITLE = re.compile(r"《([^》]{2,80})》")

_LAW_PREFIX = "中华人民共和国"


def parse_article_no(raw: str) -> int | None:
    """条号 → int。接受「第九十条」「第90条」「90」。解析不出返回 None。

    中文数字那一支复用 `law_parser.cn_to_int`（它只吃纯中文数字，所以先剥「第」「条」）。
    不重写一份 —— 同一件事有两份实现，就迟早有两份不一致。
    """
    text = (raw or "").strip()
    if not text:
        return None
    text = text.removeprefix("第").removesuffix("条").strip()
    if text.isdigit():
        return int(text)
    try:
        return cn_to_int(text)
    except ValueError:
        return None


def build_article_index(
    parents: dict[str, ParentChunk],
) -> dict[tuple[str, int], ParentChunk]:
    """`(law_id, 条号) → ParentChunk`。

    **键必须是 `(law_id, 条号)`，不是 `article_index`。** 后者在
    kb/law_parser.py 里的定义是 `pending_index = len(articles) + 1`，是**位置序号**；
    本库里它恰好等于条号（508 条、每部法 1..N 连续，实测 0 例外），但那是语料巧合，
    不是契约 —— 条号一旦跳号（法规修订常见），两者立刻分叉。

    条号也不能单独当键：6 部法规各有「第一条」，实测「第一条」出现 6 次。

    重复条号取**先出现的那个**。按测试里锁住的不变量（条号在每部法规内唯一）不该发生，
    真发生了也不抛异常 —— 取条工具崩掉比取到一条可疑的条文更糟。
    """
    index: dict[tuple[str, int], ParentChunk] = {}
    for parent in parents.values():
        number = parse_article_no(parent.article_no)
        if number is None:
            continue
        index.setdefault((parent.law_id, number), parent)
    return index


def _law_names_in(text: str) -> list[str]:
    """题面里出现的法名，长的排前面。"""
    return [m.group(1).strip() for m in RE_LAW_TITLE.finditer(text)]


def find_article(
    text: str,
    *,
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
) -> ParentChunk | None:
    """从一句自然语言里抽出「第X条」并定位到**唯一**一条法条。

    命中不了就返回 None，判据从严 —— 因为调用方会据此**跳过检索直接取条**，
    取错比取不到糟得多：
    - 题面里没有条号 → None
    - 条号在所指法规里不存在（越界）→ None
    - 题面提到的法规不止一部 → None（配对关系是猜的；6 部法规各有「第一条」）
    - 说了法规名但解析不到库里 → None
    """
    match = RE_ARTICLE_CN.search(text) or RE_ARTICLE_AR.search(text)
    if match is None:
        return None
    number = parse_article_no(match.group(0))
    if number is None:
        return None

    bracketed = {
        law_id
        for name in _law_names_in(text)
        if (law_id := resolve_law_id(name, parents)[0]) is not None
    }
    if len(bracketed) == 1:
        return index.get((bracketed.pop(), number))
    if bracketed:
        return None

    spans: list[tuple[int, int, str]] = []
    for law_id, law_name in {(p.law_id, p.law_name) for p in parents.values()}:
        for needle in {law_name, law_name.removeprefix(_LAW_PREFIX)}:
            start = text.find(needle)
            while start >= 0:
                spans.append((start, start + len(needle), law_id))
                start = text.find(needle, start + 1)
    outer = [
        span
        for span in spans
        if not any(
            other[0] <= span[0] and span[1] <= other[1] and other[1] - other[0] > span[1] - span[0]
            for other in spans
        )
    ]
    law_ids = {law_id for _, _, law_id in outer}
    if len(law_ids) == 1:
        return index.get((law_ids.pop(), number))
    if law_ids:
        return None

    candidates = [chunk for (_, no), chunk in index.items() if no == number]
    return candidates[0] if len(candidates) == 1 else None


def _single(parent: ParentChunk, *, query: str) -> RetrievalResult:
    """一条法条包成一个 `RetrievalResult`。

    **这是精确取条能全链路零改动的关键**：包成与检索结果同形状之后，
    它直接进 `search_log`、被 `merge_retrievals` 合并、喂给既有的生成层 ——
    `to_dict()` / 回灌 / 合并 / 生成四段代码一行都不用改。
    `score=1.0` 只是个占位：合并走的是轮转交错（按排名，不按分数），这个值不会被比较。
    """
    return RetrievalResult(
        query=query,
        articles=(RetrievedArticle(article=parent, score=1.0),),
        used_vector=False,
        used_bm25=False,
        elapsed_ms=0.0,
        notes=("精确取条，未走向量/BM25 检索",),
    )


def _out_of_range(
    parents: dict[str, ParentChunk], law_id: str, number: int
) -> str:
    """条号越界时的说明 —— 把该法规的实际范围报出来，模型下一轮就不会再瞎猜。"""
    numbers = [
        n
        for p in parents.values()
        if p.law_id == law_id and (n := parse_article_no(p.article_no)) is not None
    ]
    law_name = next(p.law_name for p in parents.values() if p.law_id == law_id)
    if not numbers:
        return f"《{law_name}》里没有第 {number} 条。"
    return (
        f"《{law_name}》共 {len(numbers)} 条"
        f"（第 {min(numbers)} 条..第 {max(numbers)} 条），没有第 {number} 条。"
    )


def lookup_article(
    article_no: str,
    law_name: str | None,
    *,
    parents: dict[str, ParentChunk],
    index: dict[tuple[str, int], ParentChunk],
) -> tuple[RetrievalResult | None, str]:
    """执行一次精确取条。

    成功返回 `(结果, "")`；取不到返回 `(None, 给模型看的中文说明)`。

    **失败不抛异常**：条号写错是模型的输入问题，属于可恢复的对话，不是程序错误。
    把候选摆出来，下一轮它自己会改 —— 与 `resolve_law_id` 失败时回吐法名清单同一个思路。
    """
    number = parse_article_no(article_no)
    if number is None:
        return None, (
            f"无法识别条号「{article_no}」。请给出「第九十条」或「90」这样的形式。"
        )

    if law_name:
        law_id, known = resolve_law_id(law_name, parents)
        if law_id is None:
            return None, (
                f"无法确定法规「{law_name}」。库内只有这几部，请从中选一个"
                f"（或省略 law_name 检索全部）：{'；'.join(known)}"
            )
        parent = index.get((law_id, number))
        if parent is None:
            return None, _out_of_range(parents, law_id, number)
        return _single(parent, query=article_no), ""

    candidates = [chunk for (_, no), chunk in index.items() if no == number]
    if not candidates:
        return None, f"库里没有任何一部法规有第 {number} 条。"
    if len(candidates) > 1:
        names = "；".join(sorted(chunk.law_name for chunk in candidates))
        return None, (
            f"第 {number} 条在库内有 {len(candidates)} 部法规都有：{names}。"
            f"请指明是哪一部（用 law_name 参数）。"
        )
    return _single(candidates[0], query=article_no), ""
