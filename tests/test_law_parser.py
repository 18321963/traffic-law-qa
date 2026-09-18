"""parse 层：docx → 法 → 章 → 节 → 条。

这里的断言分两种：
- **golden**（4 部 / 380 条 / 逐部条数）：知识库的真实规模。改动解析逻辑若改变了它们，
  测试会红 —— 那正是提醒你确认改动是否有意为之。
- **不变量**（leftovers 恒空、条号唯一、章标题归一）：这些任何情况下都该成立。
"""

from __future__ import annotations

import pytest

from tests.conftest import EXPECTED_ARTICLE_COUNT, EXPECTED_ARTICLES_BY_LAW, EXPECTED_LAW_COUNT
from traffic_law_rag.contracts import Paragraph
from traffic_law_rag.docx_reader import DocxReader
from traffic_law_rag.law_parser import (
    LAW_ID_REGISTRY,
    LawLibrary,
    LawParser,
    cn_to_int,
    render_markdown,
)


# ------------------------------------------------------------------ golden
def test_知识库规模(laws):
    assert len(laws) == EXPECTED_LAW_COUNT
    assert sum(law.article_count for law in laws) == EXPECTED_ARTICLE_COUNT


def test_逐部条数与版本(law_by_id):
    got = {law_id: law.article_count for law_id, law in law_by_id.items()}
    assert got == EXPECTED_ARTICLES_BY_LAW


@pytest.mark.parametrize(
    ("law_id", "version"),
    [
        ("road_traffic_safety_law", "2021-04-29"),
        ("road_traffic_safety_regulation", "2017-10-07"),
        ("sz_icv_regulation", "2026-05-27"),
        ("sz_traffic_penalty_regulation", "2024-05-10"),
    ],
)
def test_版本号取自文件名(law_by_id, law_id, version):
    """版本从 docx 文件名的 _YYYYMMDD 解析 —— 这是时效性溯源的根基，不能退化。"""
    assert law_by_id[law_id].version == version


# ------------------------------------------------------------------ 不变量
def test_leftovers_恒为空(laws):
    """归不到任何条文的正文段落。非空说明解析规则漏了某种排版，是告警信号。"""
    offenders = {law.law_id: list(law.leftovers) for law in laws if law.leftovers}
    assert not offenders, f"这些法规有解析残留段落：{offenders}"


def test_条号在每部法规内唯一(laws):
    for law in laws:
        numbers = [a.article_no for a in law.articles]
        assert len(numbers) == len(set(numbers)), f"{law.law_id} 有重复条号"


def test_article_index_从1连续递增(laws):
    """article_index 是父块 id 的组成部分，跳号会让 parent_id 出现空洞。"""
    for law in laws:
        assert [a.article_index for a in law.articles] == list(range(1, law.article_count + 1))


def test_每部法规都有章标题(laws):
    for law in laws:
        assert law.chapters, f"{law.law_id} 没解析出任何章"


def test_章标题已去掉排版全角空格(laws):
    """docx 里是「第一章　总　　则」，比对时不能带排版空格。"""
    for law in laws:
        for heading in law.chapters:
            assert "　" not in heading.no
            assert "　" not in heading.title
            assert heading.display == f"{heading.no} {heading.title}"


def test_citation_格式(laws):
    law = next(law for law in laws if law.law_id == "road_traffic_safety_law")
    assert law.citation == "《中华人民共和国道路交通安全法》(2021-04-29)"


def test_leftovers_被计入解析产物(parsed_round_trip):
    """leftovers 要落盘 —— 否则「恒为空」这个不变量在磁盘上无从校验。"""
    assert "leftovers" in parsed_round_trip


@pytest.fixture(scope="module")
def parsed_round_trip(tmp_path_factory, laws):
    library = LawLibrary(parsed_dir=tmp_path_factory.mktemp("parsed"), text_dir=tmp_path_factory.mktemp("text"))
    library.save(laws[0], write_markdown=False)
    return library.load(laws[0].law_id).to_dict()


# ------------------------------------------------------------------ 落盘 / 读回
def test_结构层落盘后可原样读回(tmp_path, laws):
    parsed_dir, text_dir = tmp_path / "parsed", tmp_path / "text"
    library = LawLibrary(parsed_dir=parsed_dir, text_dir=text_dir)
    original = laws[0]

    library.save(original, write_markdown=True)
    revived = library.load(original.law_id)

    assert revived == original
    assert (text_dir / f"{original.law_id}.md").exists()  # 人读层 Markdown 一并产出


def test_manifest_缺失时不报错只是空清单(tmp_path):
    """manifest() 是「列法规目录」的数据源，库还没建时不该炸。"""
    manifest = LawLibrary(parsed_dir=tmp_path, text_dir=tmp_path).manifest()
    assert manifest["laws"] == []
    assert manifest["generated_at"] is None


def test_manifest_记录了sha1用于增量门控(tmp_path, docx_files):
    """manifest 的 sha1 是「docx 没变就跳过解析」的判据，必须真的落进去。"""
    law = LawParser().parse_docx(docx_files[0])
    entry = LawLibrary.entry_of(law, docx_files[0], "deadbeef")

    assert entry["sha1"] == "deadbeef"
    assert entry["file"] == docx_files[0].name
    assert entry["law_id"] == law.law_id


# ------------------------------------------------------------------ 从 docx 重解析
def test_从docx重解析出同样的条数(docx_files, law_by_id):
    """docx 是唯一真源 —— 磁盘上的 parsed/*.json 必须能由它重放出来。"""
    parser, reader = LawParser(), DocxReader()
    reparsed = [parser.parse_docx(path, reader) for path in docx_files]
    got = {law.law_id: law.article_count for law in reparsed}

    assert got == EXPECTED_ARTICLES_BY_LAW
    # 逐条正文也要一致，不只是条数对得上
    for law in reparsed:
        assert [a.text for a in law.articles] == [a.text for a in law_by_id[law.law_id].articles]


def test_law_id_登记表覆盖全部法规(laws):
    """新增法规忘记登记会退化成 law_<sha1前10位> 这种不可读的 id。"""
    assert set(LAW_ID_REGISTRY) == {law.law_name for law in laws}
    for law in laws:
        assert law.law_id == LAW_ID_REGISTRY[law.law_name]


def test_未登记的法名退化但不出错():
    law = LawParser().parse(
        [Paragraph(0, "第一条　测试用。")],
        source_file="某部没登记的法_20260101.docx",
    )
    assert law.law_id.startswith("law_")
    assert len(law.law_id) == len("law_") + 10


# ------------------------------------------------------------------ 条文结构
def test_条号前缀与原文一致(laws):
    """Article.text 含「第X条」前缀，article_no 与之一致 —— 引用输出的根基。"""
    for law in laws:
        for article in law.articles:
            assert article.text.startswith(article.article_no), (
                f"{law.law_id} {article.article_no} 的正文不以该条号开头"
            )


def test_第一款含条号后续款不含(laws):
    for law in laws:
        for article in law.articles:
            assert article.paragraphs[0].startswith(article.article_no)
            for part in article.paragraphs[1:]:
                assert not part.startswith(article.article_no)


def test_article_map_可按条号取值(law_by_id):
    law = law_by_id["road_traffic_safety_law"]
    article = law.article_map()["第一条"]
    assert article.article_index == 1
    assert "为了维护道路交通秩序" in article.text


def test_章标题冗余挂在条文上(laws):
    """Article.chapter 是「章 no + 空格 + title」的冗余字符串，非外键。"""
    law = next(law for law in laws if law.law_id == "road_traffic_safety_law")
    first = law.articles[0]
    assert first.chapter is not None
    assert any(first.chapter == heading.display for heading in law.chapters)


def test_refs_排除自引用(laws):
    """条文里提到自己条号不该算交叉引用。"""
    for law in laws:
        for article in law.articles:
            assert article.article_no not in article.refs


def test_refs_是该条正文里真实出现过的条号(laws):
    from traffic_law_rag.law_parser import RE_ARTICLE_REF

    for law in laws:
        for article in law.articles:
            in_text = {m.group(0) for m in RE_ARTICLE_REF.finditer(article.text)}
            assert set(article.refs) <= in_text


# ------------------------------------------------------------------ cn_to_int
@pytest.mark.parametrize(
    ("cn", "expected"),
    [("一", 1), ("十", 10), ("十一", 11), ("二十", 20), ("二十一", 21),
     ("九十一", 91), ("一百", 100), ("一百二十四", 124), ("零", 0)],
)
def test_中文数字转阿拉伯数字(cn, expected):
    assert cn_to_int(cn) == expected


@pytest.mark.parametrize("bad", ["", "第91条", "91", "abc", "第九十一条"])
def test_非法输入抛ValueError(bad):
    """cn_to_int 只吃纯中文数字：「第X条」要由调用方先剥掉「第」「条」再传。

    这也是它全仓零调用的原因之一 —— 直接拿条号原文喂它会抛。
    """
    with pytest.raises(ValueError):
        cn_to_int(bad)


def test_空输入的错误信息可读():
    with pytest.raises(ValueError, match="空的中文数字"):
        cn_to_int("")


# ------------------------------------------------------------------ Markdown 人读层
def test_render_markdown_含法名条号与正文(laws):
    md = render_markdown(laws[0])
    assert f"# {laws[0].law_name}" in md
    assert laws[0].articles[0].article_no in md
