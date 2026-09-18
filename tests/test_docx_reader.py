"""read 层：docx → 段落列表。

这一层是整条管道的根基：「第X条必须独占一段」决定了后面所有的解析假设。
所以断言的核心是**段落边界与顺序**，而不是文本内容本身。

读的是仓库里的真实 docx（唯一真源），不是构造的假文件 ——
假文件测不出真实排版里的坑（嵌套段落、样式、空段）。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from traffic_law_rag.docx_reader import DocxReader

# 真实知识库的段落规模（golden）。keep_empty=True 的数量另记，
# 差值就是被丢弃的空段 —— 空段计数突变通常意味着排版或解析器行为变了。
EXPECTED_PARAGRAPHS = {
    "中华人民共和国道路交通安全法_20210429.docx": 310,
    "中华人民共和国道路交通安全法实施条例_20171007.docx": 320,
    "深圳经济特区智能网联汽车管理条例_20260527.docx": 125,
    "深圳经济特区道路交通安全违法行为处罚条例_20240510.docx": 232,
}
EXPECTED_TOTAL = 987


def test_逐份docx段落数(docx_files):
    reader = DocxReader()
    got = {path.name: len(reader.read(path)) for path in docx_files}

    assert got == EXPECTED_PARAGRAPHS
    assert sum(got.values()) == EXPECTED_TOTAL


def test_段落index从0连续递增(docx_files):
    """index 是段落顺序的唯一凭据，解析层靠它定位正文起点。"""
    for path in docx_files:
        paragraphs = DocxReader().read(path)
        assert [p.index for p in paragraphs] == list(range(len(paragraphs)))


def test_段落文本已去掉首尾空白(docx_files):
    for path in docx_files:
        for para in DocxReader().read(path):
            assert para.text == para.text.strip()
            assert para.text, "空段应被丢弃（keep_empty=False）"


def test_默认丢弃空段落(docx_files):
    """空段会在解析阶段变成噪声，默认不留。"""
    for path in docx_files:
        kept = DocxReader(keep_empty=True).read(path)
        dropped = DocxReader(keep_empty=False).read(path)
        assert len(kept) > len(dropped), f"{path.name} 本来就没有空段？请确认语料"


def test_keep_empty保留空段且index仍连续(docx_files):
    paragraphs = DocxReader(keep_empty=True).read(docx_files[0])

    assert len(paragraphs) > EXPECTED_PARAGRAPHS[docx_files[0].name]
    assert [p.index for p in paragraphs] == list(range(len(paragraphs)))


def test_样式字段在真实语料上确实出现(docx_files):
    """至少有一份 docx 带 pStyle（实测是实施条例的 '14'）——
    全为 None 说明样式提取坏了，而解析层依赖样式判断标题。"""
    styles = {
        para.style
        for path in docx_files
        for para in DocxReader().read(path)
        if para.style
    }
    assert styles, "没有任何段落带样式，StyleName 提取可能已失效"


def test_read_text_拍平成纯文本(docx_files):
    reader = DocxReader()
    paragraphs = reader.read(docx_files[0])
    text = reader.read_text(docx_files[0])

    assert text.startswith(paragraphs[0].text)
    assert text.count("\n") == len(paragraphs) - 1


def test_read_many_按文件名索引(docx_files):
    got = DocxReader().read_many(docx_files)

    assert set(got) == {path.name for path in docx_files}
    assert len(got[docx_files[0].name]) == EXPECTED_PARAGRAPHS[docx_files[0].name]


def test_以条号开头的段落数等于该法条数(docx_files):
    """「第X条独占一段」是整条管道的根基假设，这里直接把它量出来。

    以「第X条」开头的段落数应当**恰好等于**该部法规的条文数：多一段说明有条文
    被拆到了两段（解析层会多切出一条），少一段说明两条被并在了一段（解析层会漏条）。
    这是在 docx 层对上游解析结果做的一次独立交叉验证 —— 不经过 law_parser。
    """
    from tests.conftest import EXPECTED_ARTICLES_BY_LAW
    from traffic_law_rag.law_parser import LAW_ID_REGISTRY, RE_ARTICLE

    got = {}
    for path in docx_files:
        law_name = path.stem.rsplit("_", 1)[0]  # 文件名去掉 _YYYYMMDD
        law_id = LAW_ID_REGISTRY[law_name]
        got[law_id] = sum(1 for p in DocxReader().read(path) if RE_ARTICLE.match(p.text))

    assert got == EXPECTED_ARTICLES_BY_LAW


def test_段落内部没有换行或制表符(docx_files):
    """w:br / w:tab 会被还原成 \\n 与 \\t，而解析层是**逐段**判断条号的。

    段落里若混进换行，第二个「第X条」就藏在同一段里 —— 那一段会被整体当成
    一条法条，直到其后的条号才重新对齐，中间整段条文会被吞掉。
    实测全库均为 0；这条断言是为了在换语料时立刻发现这个前提不再成立。
    """
    offenders = [
        (path.name, para.index, para.text[:40])
        for path in docx_files
        for para in DocxReader().read(path)
        if "\n" in para.text or "\t" in para.text
    ]
    assert not offenders, f"段落内出现换行/制表符，逐段解析的前提被打破：{offenders[:3]}"


# ------------------------------------------------------------------ 异常路径
def test_文件不存在时报错可读(tmp_path):
    with pytest.raises(FileNotFoundError, match="找不到 docx 文件"):
        DocxReader().read(tmp_path / "没有这个.docx")


def test_不是docx时报错可读(tmp_path):
    fake = tmp_path / "假的.docx"
    fake.write_bytes(b"not a zip at all")

    with pytest.raises((ValueError, zipfile.BadZipFile)):
        DocxReader().read(fake)


def test_是zip但不是docx时报错可读(tmp_path):
    """能解开但缺 word/document.xml —— 报错要指明缺什么，不能是裸 KeyError。"""
    fake = tmp_path / "空的.docx"
    with zipfile.ZipFile(fake, "w") as zf:
        zf.writestr("hello.txt", "not a docx")

    with pytest.raises(ValueError, match="不是有效的 docx"):
        DocxReader().read(fake)


def test_异常消息里带路径(tmp_path):
    missing = tmp_path / "缺.docx"
    with pytest.raises(FileNotFoundError) as excinfo:
        DocxReader().read(missing)
    assert str(missing) in str(excinfo.value)


def test_接受字符串路径(docx_files):
    """Path 和 str 都该接受 —— CLI 传进来的是 str。"""
    assert DocxReader().read(str(docx_files[0])) == DocxReader().read(Path(docx_files[0]))
