"""chunk 层：条级父块 + 款级子块。

**这个文件的重头戏是「无孤立列举项」** —— README 把它当作头条量化成果
（「孤立子块从 248/867（29%）降至 0」），但此前没有任何自动化保护，
改动切分规则很容易悄悄退化回去。

其余断言分两类：父子块结构不变量，以及三条切分规则各自的行为。
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    EXPECTED_ARTICLE_COUNT,
    EXPECTED_CHUNK_COUNT,
    EXPECTED_LAW_COUNT,
)
from traffic_law_rag.chunker import RE_LIST_MARKER, LawChunker
from traffic_law_rag.contracts import Article, LawDocument


def _law_with(article: Article) -> LawDocument:
    """包一部单条法规，让切块规则的单测不必依赖真实知识库。"""
    return LawDocument(
        law_id="t",
        law_name="测试法",
        version="2026-01-01",
        source_file="t.docx",
        citation="《测试法》(2026-01-01)",
        articles=(article,),
    )


def _article(paragraphs: tuple[str, ...], *, chapter: str | None = None) -> Article:
    return Article(
        article_no="第一条",
        article_index=1,
        chapter=chapter,
        section=None,
        text="\n".join(paragraphs),
        paragraphs=paragraphs,
        refs=(),
    )

# ------------------------------------------------------------------ golden
def test_切块规模与README一致(chunk_set):
    assert len(chunk_set.parents) == EXPECTED_ARTICLE_COUNT  # 380
    assert len(chunk_set.chunks) == EXPECTED_CHUNK_COUNT  # 619
    assert chunk_set.stats["laws"] == EXPECTED_LAW_COUNT


def test_平均块长(chunk_set):
    """README 记录 79 字/块。块长失控会直接影响向量检索的信噪比。"""
    assert chunk_set.stats["avg_chunk_chars"] == pytest.approx(79.0, abs=0.5)


# ------------------------------------------------------------------ 头条声明
def test_没有孤立列举项(chunk_set):
    """子块正文不该以「（一）」「1.」「①」这类列举标记开头。

    列举项脱离引出它的那一款后既搜不到也读不懂（「（一）兜售物品、散发广告或者乞讨；」
    单独成块时 BM25 和向量都无从判断它在说什么）。
    实测这段逻辑未实现时这类子块占 29%。
    """
    orphans = [
        chunk.chunk_id
        for chunk in chunk_set.chunks
        if RE_LIST_MARKER.match(chunk.text.lstrip())
    ]
    assert not orphans, f"有 {len(orphans)} 个子块以列举标记开头，列举项合并规则失效：{orphans[:5]}"


def test_列举项确实并入了前一款():
    """直接构造一个列举项开头的条，确认它被并在引出句后面而不是单独成块。"""
    law = _law_with(
        _article(
            (
                "第一条　禁止下列行为：",
                "（一）兜售物品、散发广告或者乞讨；",
                "（二）其他行为。",
            )
        )
    )
    chunks = LawChunker().chunk(law).chunks

    assert len(chunks) == 1, "列举项应并入引出它的那一款，而不是各自成块"
    assert chunks[0].text.startswith("第一条　禁止下列行为：")
    assert "（一）兜售物品" in chunks[0].text
    assert "（二）其他行为" in chunks[0].text


def test_过短的款并入前一款():
    """短于 min_part_chars 的款同样并入 —— 例如只有一个「前款规定的除外。」的尾款。"""
    chunker = LawChunker(min_part_chars=15)
    assert chunker.split_parts(("这是足够长的一个款，超过十五个字。", "短款。")) == [
        "这是足够长的一个款，超过十五个字。\n短款。"
    ]


def test_首款过短时仍自成一个块不报错():
    """没有「前一款」可并时不能崩，也不能丢内容。"""
    assert LawChunker(min_part_chars=15).split_parts(("短。",)) == ["短。"]


def test_空段落被丢弃():
    assert LawChunker().split_parts(("正文。", "", "   ")) == ["正文。"]


def test_全部为空时退化为一个空块():
    """返回空列表会让下游 chunk_id 构造失去 part_index 基准，退化成一个空块更安全。"""
    assert LawChunker().split_parts(()) == [""]
    assert LawChunker().split_parts(("", "")) == [""]


def test_超长款按句末标点切分():
    """只在「句末标点距上一个切点已超过 max_part_chars」时切，不切在句子中间。"""
    sentence = "这是一个足够长的句子用来触发切分。"  # 17 字，含句号
    chunker = LawChunker(max_part_chars=20)

    parts = chunker.split_parts((sentence * 4,))

    assert len(parts) > 1
    assert "".join(parts) == sentence * 4, "切分不能丢字"
    for part in parts:
        assert part.endswith("。"), "必须切在句末，不能切在句子中间"
        # 允许最后一句超出阈值（阈值只在句末检查），但不该整体失控
        assert len(part) <= 20 + len(sentence)


def test_不超长的款保持原样():
    text = "一段不长的正文，不该被切开。"
    assert LawChunker(max_part_chars=500).split_parts((text,)) == [text]


# ------------------------------------------------------------------ 父子块结构
def test_每个子块都能回灌到父块(chunk_set):
    """父块回灌是这套设计的核心：命中子块后给 LLM 的是整条法条。"""
    parents = chunk_set.parent_map()
    for chunk in chunk_set.chunks:
        assert chunk.parent_id in parents, f"{chunk.chunk_id} 找不到父块 {chunk.parent_id}"


def test_每条法条至少产出一个子块(chunk_set):
    """380 条 → 619 块，平均一条 1.63 款。没有任何条文应该产出零个子块。"""
    from collections import Counter

    per_parent = Counter(chunk.parent_id for chunk in chunk_set.chunks)
    assert len(per_parent) == len(chunk_set.parents)
    assert min(per_parent.values()) >= 1


def test_子块的条号与父块一致(chunk_set):
    parents = chunk_set.parent_map()
    for chunk in chunk_set.chunks:
        parent = parents[chunk.parent_id]
        assert chunk.article_no == parent.article_no
        assert chunk.citation == parent.citation


def test_part_index_连续且part_total正确(chunk_set):
    from collections import defaultdict

    grouped: dict[str, list] = defaultdict(list)
    for chunk in chunk_set.chunks:
        grouped[chunk.parent_id].append(chunk)

    for parent_id, chunks in grouped.items():
        assert [c.part_index for c in chunks] == list(range(len(chunks))), parent_id
        assert all(c.part_total == len(chunks) for c in chunks), parent_id


def test_refs_从条文透传到父块与子块(laws, chunk_set):
    """refs 已解析但检索层尚未使用 —— agent 的 get_article 要用它，先确认透传没断。"""
    law = next(law for law in laws if any(a.refs for a in law.articles))
    with_refs = {a.article_no: a.refs for a in law.articles if a.refs}

    parents = chunk_set.parent_map()
    checked = 0
    for parent in chunk_set.parents:
        if parent.law_id == law.law_id and parent.article_no in with_refs:
            assert parents[parent.parent_id].refs == with_refs[parent.article_no]
            checked += 1
    assert checked, "没走到任何带引用的条文，这条断言是空的"


# ------------------------------------------------------------------ 命名规则
def test_parent_id_含law_id与版本(law_by_id):
    law = law_by_id["road_traffic_safety_law"]
    parent_id = LawChunker.parent_id_of(law, 91)

    assert parent_id == "road_traffic_safety_law@2021-04-29#a091"
    assert law.version in parent_id, "版本必须进 id，否则新旧版本会串号"


def test_parent_id_条序号补零到三位(chunk_set):
    for parent in chunk_set.parents:
        assert parent.parent_id.endswith(f"#a{parent.article_index:03d}")


def test_chunk_id_由parent_id派生(chunk_set):
    for chunk in chunk_set.chunks:
        assert chunk.chunk_id == f"{chunk.parent_id}#p{chunk.part_index:02d}"


# ------------------------------------------------------------------ embed_text
def test_embed_text_带法名条号前缀(chunk_set):
    """多部法规讲同一件事（道交法 vs 深圳处罚条例）时，前缀是区分它们的唯一线索。"""
    for chunk in chunk_set.chunks:
        assert chunk.embed_text.startswith(f"《{chunk.law_name}》")
        assert chunk.article_no in chunk.embed_text


def test_embed_text_首款去掉重复条号前缀():
    """首款正文自带「第X条」，前缀里已有一条，不剥掉会重复。"""
    law = _law_with(_article(("第一条　正文内容。",)))
    chunk = LawChunker(with_chapter_prefix=False).chunk(law).chunks[0]

    assert chunk.embed_text == "《测试法》第一条：正文内容。"
    assert chunk.embed_text.count("第一条") == 1


def test_embed_text_可关闭章节前缀():
    law = _law_with(_article(("第一条　正文内容。",), chapter="第一章 总则"))
    with_prefix = LawChunker(with_chapter_prefix=True).chunk(law).chunks[0]
    without = LawChunker(with_chapter_prefix=False).chunk(law).chunks[0]

    assert "第一章 总则" in with_prefix.embed_text
    assert "第一章 总则" not in without.embed_text


# ------------------------------------------------------------------ 落盘 / 读回
def test_落盘后读回的块与内存一致(chunk_stage):
    """ChunkStage 从磁盘上的结构层重跑一遍并落盘，读回后规模与字数自洽。"""
    chunk_stage.run()  # laws=None → 自动读 parsed/
    loaded = chunk_stage.load()

    assert len(loaded.parents) == EXPECTED_ARTICLE_COUNT
    assert len(loaded.chunks) == EXPECTED_CHUNK_COUNT
    assert loaded.stats["chars"] == sum(c.char_count for c in loaded.chunks)
    assert loaded.stats["laws"] == EXPECTED_LAW_COUNT
