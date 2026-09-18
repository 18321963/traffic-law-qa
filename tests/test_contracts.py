"""契约层的序列化往返。

`contracts.py` 自称「磁盘格式的唯一真源」—— 那么它的 `to_dict`/`from_dict`
必须两两配平，否则改了字段就会在某个角落静默丢数据。
这里把每个**会落盘**的契约都跑一遍往返。

注意分两类：
- 落盘对象（Heading/Article/LawDocument/ParentChunk/Chunk/ChunkSet/IndexStats）
  有 `from_dict`，做严格往返断言；
- 只是往外传的对象（RetrievalResult/Answer/RetrievedArticle 等）只有 `to_dict`，
  做「键齐全 + JSON 可序列化」断言，不假装能读回来。
"""

from __future__ import annotations

import json

import pytest

from traffic_law_rag.contracts import (
    Answer,
    Article,
    Chunk,
    ChunkSet,
    Evidence,
    Heading,
    IndexStats,
    LawDocument,
    Paragraph,
    ParentChunk,
    PipelineReport,
    Query,
    RetrievalResult,
    RetrievedArticle,
    RewrittenQuery,
    StageReport,
)

# ------------------------------------------------------------------ 样例构造
HEADING = Heading(no="第一章", title="总则")

ARTICLE = Article(
    article_no="第九十一条",
    article_index=91,
    chapter="第七章 法律责任",
    section=None,
    text="第九十一条　饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证。\n醉酒驾驶机动车的，依法追究刑事责任。",
    paragraphs=(
        "第九十一条　饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证。",
        "醉酒驾驶机动车的，依法追究刑事责任。",
    ),
    refs=("第九十条",),
)

PARENT = ParentChunk(
    parent_id="road_traffic_safety_law@2021-04-29#a091",
    law_id="road_traffic_safety_law",
    law_name="中华人民共和国道路交通安全法",
    version="2021-04-29",
    citation="《中华人民共和国道路交通安全法》(2021-04-29)",
    article_no="第九十一条",
    article_index=91,
    chapter="第七章 法律责任",
    section=None,
    text=ARTICLE.text,
    refs=("第九十条",),
)

CHUNK = Chunk(
    chunk_id="road_traffic_safety_law@2021-04-29#a091#p00",
    parent_id=PARENT.parent_id,
    law_id=PARENT.law_id,
    law_name=PARENT.law_name,
    version=PARENT.version,
    citation=PARENT.citation,
    article_no="第九十一条",
    article_index=91,
    part_index=0,
    part_total=2,
    text=ARTICLE.paragraphs[0],
    embed_text=f"《{PARENT.law_name}》{ARTICLE.chapter} 第九十一条：饮酒后驾驶机动车的…",
    chapter="第七章 法律责任",
    section=None,
    refs=("第九十条",),
)

LAW = LawDocument(
    law_id=PARENT.law_id,
    law_name=PARENT.law_name,
    version=PARENT.version,
    source_file="中华人民共和国道路交通安全法_20210429.docx",
    citation=PARENT.citation,
    articles=(ARTICLE,),
    chapters=(HEADING,),
    sections=(),
    toc=("第一章　总　　则",),
    preamble=("中华人民共和国道路交通安全法",),
    leftovers=(),
)

INDEX_STATS = IndexStats(
    collection="traffic_law",
    uri="http://localhost:19530",
    rows=619,
    dense_rows=619,
    sparse_rows=619,
    embedding_model="text-embedding-v4",
    embedding_dim=1024,
    vector_enabled=True,
    built_at="2026-09-18T10:00:00",
    elapsed_ms=12345.6,
    schema={"fields": ["chunk_id", "vector"]},
)


# ------------------------------------------------------------------ 落盘对象：严格往返
@pytest.mark.parametrize(
    "obj",
    [HEADING, ARTICLE, PARENT, CHUNK, LAW, INDEX_STATS],
    ids=lambda o: type(o).__name__,
)
def test_落盘契约_json_往返一致(obj):
    """to_dict → json → from_dict 必须还原出相等的对象。"""
    revived = type(obj).from_dict(json.loads(json.dumps(obj.to_dict(), ensure_ascii=False)))
    assert revived == obj


def test_ChunkSet_落盘往返(tmp_path):
    """ChunkSet 走 jsonl 落盘再读回，父块/子块逐字段一致。"""
    original = ChunkSet(parents=(PARENT,), chunks=(CHUNK,), stats={"whatever": 1})
    chunks_path, parents_path = tmp_path / "chunks.jsonl", tmp_path / "parents.jsonl"
    original.write(chunks_path, parents_path)

    revived = ChunkSet.read(chunks_path, parents_path)
    assert revived.parents == original.parents
    assert revived.chunks == original.chunks
    # read() 会按实际内容重算 stats，而不是沿用写入时的那份
    assert revived.stats["parents"] == 1
    assert revived.stats["chunks"] == 1
    assert revived.stats["laws"] == 1


def test_jsonl_每行一条且中文不转义(tmp_path):
    """落盘格式是「一行一 JSON」且 ensure_ascii=False —— 人工排查时要能直接读。"""
    ChunkSet(parents=(PARENT,), chunks=(CHUNK,)).write(
        tmp_path / "chunks.jsonl", tmp_path / "parents.jsonl"
    )
    lines = (tmp_path / "chunks.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert "第九十一条" in lines[0]  # 未转义成 \uXXXX


def test_ChunkSet_read_缺文件时给出可操作的中文提示(tmp_path):
    with pytest.raises(FileNotFoundError, match="请先运行切块阶段"):
        ChunkSet.read(tmp_path / "不存在.jsonl", tmp_path / "也不存在.jsonl")


# ------------------------------------------------------------------ 兼容性
def test_IndexStats_忽略旧版遗留字段():
    """Chroma 时代写的快照里有 chunk_count，读历史文件不该直接崩。"""
    legacy = INDEX_STATS.to_dict()
    legacy["chunk_count"] = 867  # 已被 Milvus 版淘汰的字段
    legacy["some_future_field"] = "ignored"

    stats = IndexStats.from_dict(legacy)

    assert stats.rows == 619
    assert not hasattr(stats, "chunk_count")


def test_IndexStats_纯BM25快照的零值可读():
    """未配 embedding key 时 dense_rows=0、embedding_dim=None，必须能落盘再读回。"""
    bm25_only = IndexStats(
        collection="traffic_law",
        uri="http://localhost:19530",
        rows=619,
        dense_rows=0,
        sparse_rows=619,
        embedding_model=None,
        embedding_dim=None,
        vector_enabled=False,
        built_at="2026-09-18T10:00:00",
        elapsed_ms=1.0,
    )
    assert IndexStats.from_dict(bm25_only.to_dict()) == bm25_only


# ------------------------------------------------------------------ 派生属性
def test_Article_派生计数不入构造参数():
    """paragraph_count / char_count 是算出来的，from_dict 要忽略盘上的值并重算。"""
    data = ARTICLE.to_dict()
    data["char_count"] = 99999  # 伪造一个错误值
    assert Article.from_dict(data).char_count == len(ARTICLE.text)


def test_LawDocument_派生计数重算():
    data = LAW.to_dict()
    data["article_count"] = 99999
    law = LawDocument.from_dict(data)
    assert law.article_count == 1
    assert law.article_map() == {"第九十一条": ARTICLE}


def test_Heading_display_去掉排版全角空格():
    assert Heading(no="第一章", title="总则").display == "第一章 总则"


def test_RetrievedArticle_citation_拼法名版本条号():
    hit = RetrievedArticle(article=PARENT, score=0.5)
    assert hit.citation == "《中华人民共和国道路交通安全法》(2021-04-29)第九十一条"


# ------------------------------------------------------------------ 单向对象：键齐全 + 可序列化
def test_只读契约都能_json_序列化():
    """这些对象要经 CLI --json / FastAPI 返回，必须能 json.dumps。"""
    retrieval = RetrievalResult(
        query="醉驾怎么处罚",
        articles=(RetrievedArticle(article=PARENT, score=0.5, bm25_rank=1),),
        used_vector=True,
        used_bm25=True,
        elapsed_ms=12.3,
        notes=("口语对齐：醉驾 → 醉酒驾驶",),
    )
    answer = Answer(
        question="醉驾怎么处罚",
        text="依据【依据1】，醉酒驾驶机动车的…",
        evidences=(
            Evidence(
                label="【依据1】",
                citation=PARENT.citation + PARENT.article_no,
                text=PARENT.text,
                score=0.5,
                article=PARENT,
            ),
        ),
        model="qwen-plus",
        elapsed_ms=100.0,
        usage={"total_tokens": 42},
        retrieval=retrieval,
    )
    for obj in (retrieval, answer, RewrittenQuery(original="a", expanded="b")):
        payload = json.loads(json.dumps(obj.to_dict(), ensure_ascii=False))
        assert isinstance(payload, dict)


def test_PipelineReport_按阶段渲染并汇总():
    """PipelineReport 是给人看的（无 to_dict），渲染要含每阶段状态与总计。"""
    report = PipelineReport(
        stages=(
            StageReport("parse", "4 个 docx", "4 部 / 380 条", True, 12.0, detail="全部重新解析"),
            StageReport(
                "index", "619 子块", "619 行", True, 3000.0, detail="向量化跳过", skipped=True
            ),
        ),
        law_count=4,
        article_count=380,
        chunk_count=619,
        index=INDEX_STATS,
    )
    rendered = report.render()

    assert "[  ok] parse" in rendered
    assert "[skip] index" in rendered  # 跳过而不是失败
    assert "法规 4 部 / 条文 380 条 / 索引块 619 个" in rendered
    assert report.elapsed_ms == 3012.0


def test_Answer_render_含正文与参考文献():
    answer = Answer(
        question="醉驾怎么处罚",
        text="醉酒驾驶机动车的，依法追究刑事责任。",
        evidences=(
            Evidence("【依据1】", PARENT.citation + PARENT.article_no, PARENT.text, 0.5, PARENT),
        ),
        model="qwen-plus",
        elapsed_ms=1.0,
        notes=("检索为空，未调用 LLM",),
    )
    rendered = answer.render()

    assert "醉酒驾驶机动车的" in rendered
    assert "参考文献：" in rendered
    assert "【依据1】" in rendered
    assert "注：检索为空" in rendered


def test_RetrievalResult_render_显示双通道排名():
    result = RetrievalResult(
        query="醉驾",
        articles=(RetrievedArticle(article=PARENT, score=0.03, vector_rank=3, bm25_rank=1),),
        used_vector=True,
        used_bm25=True,
        elapsed_ms=5.0,
    )
    rendered = result.render()

    assert "向量#3 BM25#1" in rendered
    assert PARENT.article_no in rendered


# ------------------------------------------------------------------ 检索请求
def test_Query_normalized_收拢非法参数():
    """top_k 传 0 或负数要兜到 1；candidates 不能小于 top_k。"""
    query = Query(text="  醉驾  ", top_k=0, candidates=0).normalized()
    assert query.text == "醉驾"
    assert query.top_k == 1
    assert query.candidates >= query.top_k


def test_RewrittenQuery_changed_只看有没有扩写():
    assert not RewrittenQuery(original="醉驾", expanded="醉驾").changed
    assert RewrittenQuery(original="醉驾", expanded="醉驾 醉酒驾驶", expansions=("醉酒驾驶",)).changed


def test_Paragraph_默认不带样式():
    para = Paragraph(index=0, text="第一条　为了…")
    assert para.style is None
