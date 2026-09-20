"""Agent 工具层：RAG 与 Agent 循环之间的边界（阶段二）。

**这一层不认识 LangGraph。** 它只做两件事：

1. 把 `LegalRAG` 包成一个模型能调用的工具（`SEARCH_LAW_TOOL` + 校验 + 渲染）；
2. 把 N 次检索的结果合并回**一个** `RetrievalResult`，好让收尾节点原样喂给
   既有的 `AnswerGenerator.generate()` —— 生成层因此一行都不用改。

之所以刻意不依赖框架：工具的参数校验、法名解析、合并去重都能脱离图单测，
延续项目「契约是唯一真源」的一贯做法。将来若把 LangGraph 换掉，这一层不用动。

    search_law 的完整边界：

        LLM → 工具   arguments 字符串 {"query": str, "law_name"?: str, "top_k"?: int}
        工具 → LLM   一段中文文本（引用 + 相关度 + 摘要 + 相关条号 + 降级提示）
        工具 → 图    一行 RetrievalResult.to_dict()（指针，不含正文）
        图  → 生成层  合并后的一个 RetrievalResult（含正文，正文由 parents 回灌）
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import ParentChunk, RetrievalResult, RetrievedArticle
from ..kb.law_parser import cn_to_int

SEARCH_LAW_NAME = "search_law"

# 参数的上下界。Schema 里也写了，但那只对模型是「提示」——实测模型会给 999 这类值，
# 所以服务端必须自己再夹一次，不能指望 Schema 校验。
TOP_K_MIN = 1
TOP_K_MAX = 20

# 摘要里「帽子句」的长度上限：法条几乎都写成
# 「第X条　<主体规则/处罚>：下列…（一）…（二）…」，处罚写在帽子里、适用情形写在列举里。
CHAPEAU_CHARS = 40
# 拼接帽子句之后，留给最佳窗口的最小字数；小于它说明预算太小，退回只给窗口
MIN_BODY_CHARS = 40

SEARCH_LAW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_LAW_NAME,
        "description": (
            "在交通法规知识库（6 部法规、约 508 条）中检索法条。"
            "输入一句话或一组关键词，返回命中的法条（法规名+条号+相关度+原文摘要+该条提到的其他条号）。"
            "口语词会被自动对齐成法条用语（如 醉驾→醉酒驾驶）。"
            "需要多部法规才能回答的问题，请分多次检索，每次只问一件事。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    # 措辞必须与 AGENT_SYSTEM_PROMPT 第 2 条一致：这里曾经写的是
                    # 「尽量靠近法条用语，例如『醉酒驾驶 机动车 处罚』」，与系统提示词的
                    # 「第一次检索原样使用用户的问题，不要改写」正面冲突 —— 而 schema 挂在
                    # 模型真正填参数的那个字段上，于是它赢了：100 题里只有 2 题照搬原话。
                    "description": (
                        "检索词。**第一轮直接用用户的原话** —— 检索层会自动做口语对齐"
                        "（醉驾→醉酒驾驶）与混合召回，拆成关键词反而稀释信号。"
                        "续查时只写要补的那一块。"
                    ),
                },
                "law_name": {
                    "type": "string",
                    "description": (
                        "可选。只在某一部法规内检索，用于把结果限定到特定法规，"
                        "例如「深圳经济特区道路交通安全违法行为处罚条例」"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": TOP_K_MIN,
                    "maximum": TOP_K_MAX,
                    "description": "返回条数，默认 6",
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================ 参数解析
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

    **不引 eval.LawResolver**：`parents` 里本来就躺着全部法规的 law_name 与 law_id，
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


def parse_tool_arguments(
    raw: str, *, default_top_k: int
) -> tuple[str, str | None, int]:
    """解析模型给的 `arguments` 字符串 → (query, law_name, top_k)。

    模型给的东西什么都有可能：空串、非法 JSON、缺 query、top_k=999。
    这里只负责挡下**不能用**的（抛 ValueError，由调用方翻成一句中文回灌给模型），
    能用的就尽量救回来（top_k 夹取、law_name 原样带出交给 resolve_law_id）。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("arguments 为空")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"arguments 应是 JSON 对象，实际是 {type(data).__name__}")

    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError("缺少 query")

    raw_top_k = data.get("top_k", default_top_k)
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError):
        top_k = default_top_k
    top_k = max(TOP_K_MIN, min(TOP_K_MAX, top_k))

    law_name = str(data.get("law_name") or "").strip() or None
    return query, law_name, top_k


def tool_message(call_id: str, text: str) -> dict:
    """OpenAI 线上格式的 tool 消息。

    每个 tool_call 必须恰好配一条，否则下一次请求会被端点以 400 拒绝
    （assistant 的 tool_calls 必须紧跟对应的 tool 消息）。
    """
    return {"role": "tool", "tool_call_id": call_id, "content": text}


# ============================================================ 第二个工具：精确取条
GET_ARTICLE_NAME = "get_article"

GET_ARTICLE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": GET_ARTICLE_NAME,
        "description": (
            "按条号精确取一条法条的原文。**已经知道要查哪一条时用它，比检索更准** —— "
            "检索只保证「相关」，不保证「就是那一条」。"
            "当 search_law 结果里的「相关条」提示指向某条，而你还没看过它时，也用它去取。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "article_no": {
                    "type": "string",
                    "description": "条号，中英文数字都行，例如「第九十条」或「90」",
                },
                "law_name": {
                    "type": "string",
                    "description": (
                        "可选。法规名。省略时若该条号在全库唯一就直接取；"
                        "若多部法规都有这个条号，会返回候选列表让你指明"
                    ),
                },
            },
            "required": ["article_no"],
        },
    },
}

# 条号：中文数字与阿拉伯数字都收。库里的条文一律是中文数字（实测 508 条全是），
# 但真实用户会写「第90条」，收下来成本为零。
RE_ARTICLE_CN = re.compile(r"第\s*([零一二三四五六七八九十百千]+)\s*条")
RE_ARTICLE_AR = re.compile(r"第\s*(\d{1,4})\s*条")
# 书名号里的法名。上限给到 80 而不是 40：库里没有英文法名，但用户可能抄来一个很长的
# 别名，截断了反而解析不到 —— 能不能对上由 resolve_law_id 说了算，正则不必先卡一道。
RE_LAW_TITLE = re.compile(r"《([^》]{2,80})》")

# 「中华人民共和国」前缀：法规的通用简称就是全名去掉它
_LAW_PREFIX = "中华人民共和国"


def parse_article_no(raw: str) -> int | None:
    """条号 → int。接受「第九十条」「第90条」「90」。解析不出返回 None。

    中文数字那一支复用 `law_parser.cn_to_int`（它只吃纯中文数字，所以先剥「第」「条」）。
    不重写一份：那个函数已被 tests/test_law_parser.py 的参数化用例覆盖，
    连"第91条"/"91"/"abc" 这类非法输入的行为都锁死了。
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

    # 先看有没有指名道姓的法规（书名号）。题面里可能有好几个《》，其中引用了库外法规
    # （如《机动车管理办法》）是很常见的写法，不能因此放弃 —— 所以**只看能解析到的**：
    # 恰好解析到一部就用它；解析到两部及以上说明题面真的牵涉多部法规，
    # 而「第一个条号配第一部法规」只是猜测，退回检索。
    bracketed = {
        law_id
        for name in _law_names_in(text)
        if (law_id := resolve_law_id(name, parents)[0]) is not None
    }
    if len(bracketed) == 1:
        return index.get((bracketed.pop(), number))
    if bracketed:
        return None

    # 没写书名号，再试试有没有直接把法名写进句子的（含去掉「中华人民共和国」的简称）。
    # 真实用户就这个写法：「道路交通安全法实施条例第六十六条怎么规定的」。
    #
    # **长名必须吃掉短名。** 短名是长名的子串 ——「道路交通安全法」整段出现在
    # 「道路交通安全法实施条例」里 —— 所以「命中两个法名就判歧义」会把**唯一确定**的
    # 情形误判成歧义。实测这条（按当时的 208 行口径）让点名桶里 18 行白白退回检索，其中十几行题面
    # 只提到了「实施条例」这一部法规。
    spans: list[tuple[int, int, str]] = []  # (起点, 终点, law_id)
    for law_id, law_name in {(p.law_id, p.law_name) for p in parents.values()}:
        for needle in {law_name, law_name.removeprefix(_LAW_PREFIX)}:
            start = text.find(needle)
            while start >= 0:
                spans.append((start, start + len(needle), law_id))
                start = text.find(needle, start + 1)
    # 被更长的命中完整盖住的那个不算：同一处文字只指一部法规。
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
        return None  # 真的提到两部法规，不知道问的是哪一部

    # 完全没说法规：只有当这个条号在全库唯一时才敢取
    candidates = [chunk for (_, no), chunk in index.items() if no == number]
    return candidates[0] if len(candidates) == 1 else None


def parse_article_arguments(raw: str) -> tuple[str, str | None]:
    """解析模型给的 get_article `arguments` → (article_no, law_name)。

    与 `parse_tool_arguments` 同一套标准：挡下**不能用**的（抛 ValueError，
    由调用方翻成一句中文回灌给模型），能用的尽量救回来。
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("arguments 为空")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"arguments 不是合法 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"arguments 应是 JSON 对象，实际是 {type(data).__name__}")

    article_no = str(data.get("article_no") or "").strip()
    if not article_no:
        raise ValueError("缺少 article_no")
    return article_no, str(data.get("law_name") or "").strip() or None


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


# ============================================================ 摘要
def _bigrams(text: str) -> set[str]:
    """中文字符二元组。不做分词 —— 摘要是要「找得到那段话」，不是要语义理解，
    二元组对中文足够，且不需要引入分词依赖。"""
    cleaned = "".join(ch for ch in text if ch.strip())
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)} or {cleaned}


def _chapeau(text: str, limit: int) -> str:
    """条文开头的「帽子句」，尽量在冒号处收尾。

    「第十三条　驾驶机动车有下列行为之一的，处三百元罚款：」——处罚数额就在这里，
    而它后面才是那一长串「（一）…（七）…」的适用情形。截到冒号比分在
    「处三百元罚」这种地方干净。
    """
    head = text[:limit]
    cut = head.rfind("：")
    return head[: cut + 1] if cut >= 0 else head


def _snippet(text: str, query: str, width: int) -> str:
    """取最能体现「本文与查询相关」的那一段，而不是无脑取开头。

    **为什么不能固定截前 N 字**：法条大量是列举型（「下列…行为之一的，处…罚款：
    （一）…（二）…」）。开头 N 字永远是「下列」加头一两项，真正相关的那一项可能在
    第八项，**固定截断等于让模型看不见答案在哪**。实测的后果不是「模型答得差」，
    而是模型反复改写查询去找一个能确认相关性的说法，一路烧完预算才被迫收尾 ——
    一个摘要策略的缺陷，表现出来像是模型的规划能力差。

    做法是按查询二元组的命中密度滑窗，取密度最高的窗口。O(n)：
    先给每个位置打分，再前缀和求窗口内的总分。

    打分**不是每个命中记 1，而是记 1/该二元组在本文中出现的次数**。理由是
    「驾驶机动车」这类泛词在一条法条里反复出现，若按次数计，窗口会被拽向
    「和问题同属交通领域」的无关列举项。实测第十三条：查询里的「驾驶」出现 3 次，
    与它一同命中的窗口落在 (五)(六) 项上，而真正的答案 (七) 差几个字被切在窗口外 ——
    打分方式的一个小选择，决定了模型能不能看见答案。按词频倒数加权后，稀有词
    （移动电话 / 电子设备 / 妨碍安全驾驶）才会主导窗口位置。

    但**只给窗口还不够**。列举型法条里「处罚」和「情形」是天各一方的：
    处罚在帽子句（「…有下列行为之一的，处三百元罚款：」），情形在列举项里。
    实测第十三条，查询对齐后窗口正确落到了「（七）手动操作移动电话」——
    结果模型看见「有这一项」却看不见「罚多少」，审核节点照样判「缺罚款具体数额」。
    所以要**帽子句 + 最佳窗口两头都给**；窗口本来就在开头时合成一段，不重复也不拼接。

    一个二元组都命不中时退回开头（此时本文与查询确实没有字面重叠）。

    `query` 应当是**检索层实际拿去检索的那串词**（含口语对齐展开），不是用户原话 ——
    原话里的口语词在法条里压根不出现，窗口会因此停在开头。
    那串词由 `LegalRAG.expand` 给出（见 [qa/rag.py](../qa/rag.py)）。
    """
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text

    grams = _bigrams(query)
    if not grams or not query.strip():
        return text[:width] + "…"

    counts: dict[str, int] = {}
    for i in range(len(text) - 1):
        gram = text[i : i + 2]
        if gram in grams:
            counts[gram] = counts.get(gram, 0) + 1

    marks = [0.0] * len(text)
    for i in range(len(text) - 1):
        gram = text[i : i + 2]
        if gram in counts:
            marks[i] = 1.0 / counts[gram]
    prefix = [0.0] * (len(text) + 1)
    for i, mark in enumerate(marks):
        prefix[i + 1] = prefix[i] + mark

    if prefix[-1] == 0:  # 与查询毫无字面重叠
        return text[:width] + "…"

    best_start, best_score = 0, -1
    for start in range(0, len(text) - width + 1):
        score = prefix[start + width] - prefix[start]
        if score > best_score:
            best_start, best_score = start, score

    # 窗口覆盖的**字符**区间是 [best_start, best_start + width] —— 首尾都含。
    # 多出来的那个 1 不是笔误：marks[i] 是「以字符 i 开头的二元组」的分数，
    # 它同时属于字符 i 和 i+1。滑窗累加的是 marks[start..start+width-1]，
    # 于是最后一个二元组把字符 start+width 也带了进来。按 width 切片会**切掉
    # 最佳窗口的最后一个字** —— 实测「…应当停车让行」被切成「…应当停车让」，
    # 恰好弄丢查询词的收尾字。窗口长度与切片长度差 1 这种事不会报错，
    # 只会让模型看见半句话。
    window_end = best_start + width + 1

    if best_start <= CHAPEAU_CHARS:
        # 命中项就在开头附近：帽子句本来就在视野里，给一段连续文本即可，
        # 但要多给一截，保证整个最佳窗口都包进来。
        end = min(len(text), window_end)
        return f"{text[:end]}{'…' if end < len(text) else ''}"

    head = _chapeau(text, CHAPEAU_CHARS)
    body_len = width - len(head) - 1
    if body_len < MIN_BODY_CHARS:
        return f"…{text[best_start:window_end]}…"

    tail = "…" if best_start + body_len < len(text) else ""
    return f"{head}…{text[best_start : best_start + body_len]}{tail}"


# ============================================================ 渲染给模型看
def render_tool_result(
    result: RetrievalResult,
    *,
    index: int,
    seen: set[str] | None = None,
    snippet_chars: int = 120,
    match_text: str | None = None,
) -> str:
    """把检索结果渲染成给模型读的文本。

    **只给摘要，不给全文**（`snippet_chars`，默认 120 字）。规划轮是最贵的一轮 ——
    每一轮都要把整条消息历史重发一次，6 条法条全文（200~400 字/条）等于给后续每一轮
    都加上约 1000 token，而这些正文在规划阶段根本用不上：真正进生成提示词的是收尾节点
    从 `parents` 回灌的 `ParentChunk.text`。

    所以这里的取舍是：**够判断（是不是讲这件事）+ 能定位（是哪一条）**，
    判断不了的细节交给「相关条号」与法规名去兜。

    `seen` 是前几轮已经命中过的 parent_id；「新增 / 已知」的计数是模型判断
    「再检也检不出新东西了」的直接依据 —— 没有这个信号，模型只能靠预算耗尽才停。

    `match_text` 是**检索层实际用的检索词**（含口语对齐），由 `LegalRAG.expand` 给出。
    摘要窗口按它定位，不按 `result.query`（那是模型的原话）。详见 `_snippet`。
    """
    seen = seen or set()
    total = len(result.articles)
    fresh = sum(1 for a in result.articles if a.article.parent_id not in seen)
    lines = [
        f"检索#{index}「{result.query}」｜命中 {total} 条（新增 {fresh} / 已知 {total - fresh}）"
        f"｜向量={'开' if result.used_vector else '关'} "
        f"BM25={'开' if result.used_bm25 else '关'} ｜ {result.elapsed_ms:.0f}ms"
    ]
    if not total:
        lines.append("没有命中的法条。请换一组更接近法条原文的关键词，或补上具体违法情形与地点后重试。")
    elif not fresh:
        # 「新增 0」是模型判断「该停了」的直接依据，埋在标题行里会被忽略 —— 实测模型
        # 会一直换词重检同一个意思，把预算烧完才被迫收尾。所以单独挑明一行。
        lines.append("⚠ 本轮没有新增法条，与之前的检索重复。换个完全不同的角度，或直接结束检索。")

    for order, hit in enumerate(result.articles, start=1):
        lines.append(f"【{order}】{hit.citation} ｜ 相关度 {hit.score:.4f}")
        lines.append(f"    {_snippet(hit.article.text, match_text or result.query, snippet_chars)}")
        if hit.article.refs:
            lines.append(f"    相关条：{'、'.join(hit.article.refs)}")

    for note in result.notes:
        lines.append(f"提示：{note}")
    return "\n".join(lines)


# ============================================================ 合并回一个 RetrievalResult
def merge_retrievals(
    logs: list[dict],
    *,
    question: str,
    parents: dict[str, ParentChunk],
    max_evidence: int,
) -> RetrievalResult:
    """把 N 次检索的结果合并成一个 `RetrievalResult`。

    **这是收尾节点与生成层之间的唯一接口。** 有了它，`AnswerGenerator.generate()`
    可以原样接收 Agent 的证据 —— 提示词不是「保持一致」，而是「就是同一段代码」。

    合并规则：

    1. **去重**：同一条法条被多次命中只留一条（首次出现的那条）。
    2. **轮转交错排序**，不按分数排：query1 的 #1、query2 的 #1、query3 的 #1、
       query1 的 #2… 理由是**跨查询的 RRF 分数不可比** —— 不同查询的通道数可能不同，
       而且命中法名线索的那次检索整条被乘了 1.5（见 retriever 的 law_hint_boost）。
       按分数排会让「查询里带了法规名」的那次系统性压过其他次，而那未必是更相关的那次。
       轮转交错是纯排名操作，不需要跨查询比分数，且保证每次检索的头名都进得来。
    3. **截断**到 `max_evidence`（默认与 `top_k` 相同 —— 证据槽位和基线一样多，
       hit@k 才谈得上直接可比）。

    正文从 `parents` 回灌（`parent_id` → `ParentChunk`）。回灌不到就跳过并记一条 note，
    不静默丢依据 —— 那意味着 Milvus 里的块和本地块文件不同步，是需要被看见的异常。
    """
    ordered: list[RetrievedArticle] = []
    seen: set[str] = set()
    missing = 0

    # 轮转交错：外层是「轮次内排名」，内层是「第几次检索」
    depth = max((len(row.get("articles") or ()) for row in logs), default=0)
    for rank in range(depth):
        for row in logs:
            articles = row.get("articles") or ()
            if rank >= len(articles):
                continue
            item = articles[rank]
            parent_id = item.get("parent_id", "")
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)

            parent = parents.get(parent_id)
            if parent is None:
                missing += 1
                continue
            ordered.append(
                RetrievedArticle(
                    article=parent,
                    score=item.get("score", 0.0),
                    vector_rank=item.get("vector_rank"),
                    bm25_rank=item.get("bm25_rank"),
                    vector_score=item.get("vector_score"),
                    bm25_score=item.get("bm25_score"),
                    hit_chunks=tuple(item.get("hit_chunks") or ()),
                    law_hint=item.get("law_hint"),
                )
            )

    deduped = len(ordered)
    truncated = max(0, deduped - max_evidence)
    if max_evidence > 0:
        ordered = ordered[:max_evidence]

    notes: list[str] = []
    for number, row in enumerate(logs, start=1):
        for note in row.get("notes") or ():
            notes.append(f"检索#{number}：{note}")
    if len(logs) > 1:
        # 说「轮工具调用」而不是「次检索」：`get_article` 的产物同形状，但它没检索。
        notes.append(f"共 {len(logs)} 轮工具调用，合并去重后 {deduped} 条")
    if truncated:
        notes.append(f"证据按排名截断至 {max_evidence} 条（合并后共 {deduped} 条）")
    if missing:
        notes.append(f"有 {missing} 条命中无法回灌原文（索引与本地块文件不同步），已跳过")

    return RetrievalResult(
        query=question,
        articles=tuple(ordered),
        used_vector=any(bool(row.get("used_vector")) for row in logs),
        used_bm25=any(bool(row.get("used_bm25")) for row in logs),
        elapsed_ms=sum(float(row.get("elapsed_ms") or 0.0) for row in logs),
        notes=tuple(notes),
    )
