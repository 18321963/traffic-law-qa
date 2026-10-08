from __future__ import annotations

from rag_contracts.domain.disk import Paragraph
from rag_service.indexing.parser import LawParser
from rag_service.indexing.sources import (
    PdfReader,
    _denoise_lines,
    _drop_frequent_lines,
    _to_blocks,
)


def test_denoise_drops_the_fixed_noise_shapes() -> None:
    raw = [
        "2026/9/19 08:49 机动车驾驶证申领和使用规定 _ 公安部 _ 中国政府网",
        "https://www.gov.cn/zhengce/2021-12/27/content_5712881.htm 1/29",
        "主办单位：国务院办公厅　运行维护单位：中国政府网运行中心",
        "网站标识码bm01000001　京ICP备05070218号京公网安备11010202000001号",
        "29 / 29",
        "42",
        "/",
        "____",
        "第一条 本规定自发布之日起施行。",
    ]

    assert _denoise_lines(raw) == ["第一条 本规定自发布之日起施行。"]


def test_denoise_normalizes_only_punct_adjacent_spaces() -> None:
    raw = ['所称“一日”、 “三日”，是指工作日。', "实施 条例"]

    assert _denoise_lines(raw) == ['所称“一日”、“三日”，是指工作日。', "实施 条例"]


def test_denoise_keeps_the_space_after_an_article_head() -> None:
    raw = ["第一条 为了规范机动车驾驶证申领和使用，制定本规定。"]

    assert _denoise_lines(raw) == raw


def test_frequent_lines_keep_a_punctuated_enumeration_at_the_threshold() -> None:
    lines = ["（一）机动车；", "（一）机动车；", "（一）机动车；", "只出现一次的行"]

    assert _drop_frequent_lines(lines) == ["（一）机动车；"] * 3 + ["只出现一次的行"]


def test_frequent_lines_keep_long_or_punctuated_repeats() -> None:
    long_line = "这一行故意写得非常非常非常非常非常非常非常非常长超过三十个字符"
    lines = [long_line] * 3 + ["标点，重复。"] * 3 + ["纯短重复"] * 3

    assert _drop_frequent_lines(lines) == [long_line] * 3 + ["标点，重复。"] * 3


def test_blocks_start_at_article_and_chapter_heads_and_join_folded_lines() -> None:
    lines = [
        "第一章 总则",
        "第一条 为了加强管理，制定本条例。",
        "第十八条、第十九条第一款规定的情形，",
        "应当从重处理。",
        "第二条 本条例适用于本市行政区域。",
        "附则",
    ]

    assert _to_blocks(lines) == [
        "第一章 总则",
        "第一条 为了加强管理，制定本条例。第十八条、第十九条第一款规定的情形，应当从重处理。",
        "第二条 本条例适用于本市行政区域。",
        "附则",
    ]


def test_a_folded_reference_never_becomes_a_phantom_article() -> None:
    lines = [
        "第一章 总则",
        "第二条 本条例适用于本市行政区域内的机动车。",
        "第十八条、第十九条第一款规定的情形，",
        "应当从重处理。",
        "第二十一条 本条排在折叠引用行之后，必须原样成条。",
    ]
    law = LawParser().parse(
        [Paragraph(index, block, None) for index, block in enumerate(_to_blocks(lines))],
        source_file="测试条例_20250101.pdf",
    )

    assert [article.article_no for article in law.articles] == ["第二条", "第二十一条"]
    assert "第十八条、第十九条第一款规定的情形，应当从重处理。" in law.articles[0].text
    assert law.articles[0].refs == ("第十八条", "第十九条")


def test_an_article_head_without_a_space_stays_inside_the_previous_block() -> None:
    lines = ["第一章 总则", "第一条为了加强管理，制定本条例。", "第二条 本条例自发布之日起施行。"]
    blocks = _to_blocks(lines)

    assert blocks == [
        "第一章 总则第一条为了加强管理，制定本条例。",
        "第二条 本条例自发布之日起施行。",
    ]
    law = LawParser().parse(
        [Paragraph(index, block, None) for index, block in enumerate(blocks)],
        source_file="测试条例_20250101.pdf",
    )
    assert [article.article_no for article in law.articles] == ["第二条"], (
        "宁丢勿幻影：条后没有空白的行不新起块，它并进上一块、不成为条"
    )


def test_the_reader_drops_a_plain_header_repeated_on_every_page(pdf_maker, tmp_path) -> None:
    header = "测试条例内部资料"
    pages = [
        [header, "第一条 正文甲。"],
        [header, "第二条 正文乙。"],
        [header, "第三条 正文丙。"],
    ]
    path = tmp_path / "测试条例.pdf"
    path.write_bytes(pdf_maker(pages))

    blocks = PdfReader().read(path)

    texts = [block.text for block in blocks]
    assert header not in texts, "页眉只按单页统计就永远够不着 >=3 次，这条锁的是跨全文档合并统计"
    assert texts == ["第一条 正文甲。", "第二条 正文乙。", "第三条 正文丙。"]


def test_the_reader_feeds_the_parser_a_clean_article_sequence(pdf_maker, tmp_path) -> None:
    header = "2026/9/19 08:49 测试条例 _ 公安部 _ 中国政府网"
    pages = [
        [header, "测试条例", "第一条 为了测试，制定本条例。"],
        [header, "第二条 本条例所称测试，是指自动化测试。"],
        [header, "第三条 本条例自2025年1月1日起施行。", "https://www.gov.cn/x 1/3"],
    ]
    path = tmp_path / "测试条例_20250101.pdf"
    path.write_bytes(pdf_maker(pages))

    law = LawParser().parse(PdfReader().read(path), source_file=path.name)

    assert [article.article_no for article in law.articles] == ["第一条", "第二条", "第三条"]
    assert law.law_name == "测试条例" and law.version == "2025-01-01"
    assert law.preamble == ("测试条例",)
    assert not law.leftovers
