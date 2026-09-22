"""工具结果 → 给模型读的文本：摘要窗口 + 渲染。

两个工具共用这一层。这里的取舍（给摘要不给全文、帽子句与最佳窗口两头都给）
会直接影响模型能不能看见答案 —— 详见 `_snippet` 里记的那两次实测：
摘要策略的缺陷，表现出来像是模型的规划能力差。

**这一层的原则是「只印模型能据此动一下的东西」。** 检索层的内部量 —— 融合分、
通道开关、耗时、版本日期、法名线索加成 —— 模型看了也改不了下一次怎么查，而
`相关度` 尤其两头不讨好：命中的 RRF 分实测 888 条全落在 0.015~0.049，读起来像
「几乎不相关」，可它标的偏偏是排第一的那条（且与 `【N】` 完全重复 —— 命中按融合分
降序，序号已经是排名）；精确取条那条路印的又是 `_single()` 写死的占位 `1.0000`。

**这些量一个都没丢。** 它们逐字段都在 state 的 `search_log`（`RetrievalResult.to_dict()`）
里，人读的 `RetrievalResult.render()` 也照印 —— 这里做的只是不再往提示词里抄一遍。
按 172 次真工具结果逐字节比过（`git show HEAD:...tools.py` 的旧渲染对新渲染）：
整段文本 167,678 → 130,047 字，减 **22.4%**；同一句话重检的那一轮降得更多（1065 → 392 字），
因为已知条连正文都不用再印。
"""

from __future__ import annotations

from ...contracts import ParentChunk, RetrievalResult

CHAPEAU_CHARS = 40
MIN_BODY_CHARS = 40

_DROP_NOTES = ("法名线索：",)
"""不进提示词的 note 前缀。

`法名线索：X（命中法规的分数 ×1.5）` 讲的是检索层怎么调分，模型据此动不了任何事。
`口语对齐：A → B` 留着 —— 它说的是模型自己的问法被怎么理解，那是能据以改写的反馈。

**只在渲染这层过滤，不动 retriever 的产出**：同一批 note 还要进 `Answer.notes`
给人看（`--json`、轨迹、`RetrievalResult.render()`），在源头删掉等于把人的那份也删了。
"""


def _cite(article: ParentChunk) -> str:
    """`《法名》条号` —— 规划轮用的引用，不带版本日期。

    `RetrievedArticle.citation` 是 `article.citation + article.article_no`，
    即 `《法名》(版本)条号`。规划轮只需要知道「是哪条」；版本日期在这一轮没有用武之地
    （库内每部法只有一个版本，实测 508 条父块里的构成全一致），它留到答案的参考文献里
    再出现。
    """
    return f"《{article.law_name}》{article.article_no}"


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
    结果模型看见「有这一项」却看不见「罚多少」，那一轮照样被判成「缺罚款具体数额」。
    所以要**帽子句 + 最佳窗口两头都给**；窗口本来就在开头时合成一段，不重复也不拼接。

    一个二元组都命不中时退回开头（此时本文与查询确实没有字面重叠）。

    `query` 应当是**检索层实际拿去检索的那串词**（含口语对齐展开），不是用户原话 ——
    原话里的口语词在法条里压根不出现，窗口会因此停在开头。
    那串词由 `LegalRAG.expand` 给出（见 [qa/rag.py](../../qa/rag.py)）。
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

    if prefix[-1] == 0:
        return text[:width] + "…"

    best_start, best_score = 0, -1
    for start in range(0, len(text) - width + 1):
        score = prefix[start + width] - prefix[start]
        if score > best_score:
            best_start, best_score = start, score

    window_end = best_start + width + 1

    if best_start <= CHAPEAU_CHARS:
        end = min(len(text), window_end)
        return f"{text[:end]}{'…' if end < len(text) else ''}"

    head = _chapeau(text, CHAPEAU_CHARS)
    body_len = width - len(head) - 1
    if body_len < MIN_BODY_CHARS:
        return f"…{text[best_start:window_end]}…"

    tail = "…" if best_start + body_len < len(text) else ""
    return f"{head}…{text[best_start : best_start + body_len]}{tail}"


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

    **已知条只给引用行，不再重复正文。** 上一轮已经给过全文的条，那段话本来就躺在
    模型的上下文里，再印一遍是纯重复（实测 43 条多轮提问里 82 条命中是重复的，占正文
    字符数的 10.3%）。留引用行是因为「这条我看过了」这个信号本身有用 —— 头部那行
    只报总数，看不出是哪几条。

    `match_text` 是**检索层实际用的检索词**（含口语对齐），由 `LegalRAG.expand` 给出。
    摘要窗口按它定位，不按 `result.query`（那是模型的原话）。详见 `_snippet`。
    """
    seen = seen or set()
    total = len(result.articles)
    fresh = sum(1 for a in result.articles if a.article.parent_id not in seen)
    lines = [f"检索#{index}「{result.query}」｜命中 {total} 条（新增 {fresh} / 已知 {total - fresh}）"]
    if not total:
        lines.append("没有命中的法条。请换一组更接近法条原文的关键词，或补上具体违法情形与地点后重试。")
    elif not fresh:
        lines.append("⚠ 本轮没有新增法条，与之前的检索重复。换个完全不同的角度，或直接结束检索。")

    emitted = set(seen)
    for order, hit in enumerate(result.articles, start=1):
        known = hit.article.parent_id in emitted
        emitted.add(hit.article.parent_id)
        tail = "（上一轮已给过原文）" if known else ""
        lines.append(f"【{order}】{_cite(hit.article)}{tail}")
        if not known:
            lines.append(f"    {_snippet(hit.article.text, match_text or result.query, snippet_chars)}")
        if hit.article.refs:
            lines.append(f"    相关条：{'、'.join(hit.article.refs)}")

    for note in result.notes:
        if not note.startswith(_DROP_NOTES):
            lines.append(f"提示：{note}")
    return "\n".join(lines)
