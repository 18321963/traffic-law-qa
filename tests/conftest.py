"""共享 fixture。

**所有测试必须离线可跑**：不连 Milvus、不发网络请求。
需要真实端点的用例标记 `@pytest.mark.integration`，默认被 pyproject 的 addopts 跳过。

数据来源刻意选 `法规知识库/parsed/`（已入库）与 `法规知识库/docx/`（唯一真源），
而不是 `法规知识库/chunks/` —— 后者被 .gitignore 忽略，CI 上不存在；
而且让测试**真跑一遍切块逻辑**，比断言一份陈旧产物有意义得多。
"""

from __future__ import annotations

import pytest

from traffic_law_rag import config
from traffic_law_rag.contracts import ChunkSet, LawDocument
from traffic_law_rag.kb.chunker import ChunkStage, LawChunker
from traffic_law_rag.kb.law_parser import LawLibrary

# 知识库的真实规模（golden）。改动解析/切块逻辑若改变了它们，
# 这些测试会红 —— 那正是提醒你确认改动是否有意为之。
EXPECTED_LAW_COUNT = 6
EXPECTED_ARTICLE_COUNT = 508
EXPECTED_CHUNK_COUNT = 812
EXPECTED_ARTICLES_BY_LAW = {
    "road_traffic_safety_law": 124,
    "road_traffic_safety_regulation": 115,
    "road_transport_regulation": 82,
    "traffic_insurance_regulation": 46,
    "sz_icv_regulation": 64,
    "sz_traffic_penalty_regulation": 77,
}


@pytest.fixture(scope="session")
def docx_files() -> list:
    """知识库里的 docx 源文件（按文件名排序，顺序稳定）。"""
    files = sorted(config.DOCX_DIR.glob("*.docx"))
    if not files:
        pytest.skip(f"知识库为空：{config.DOCX_DIR} 下没有 docx")
    return files


@pytest.fixture(scope="session")
def laws() -> list[LawDocument]:
    """结构层产物（法 → 章 → 节 → 条）。约 7ms。"""
    loaded = LawLibrary().load_all()
    if not loaded:
        pytest.skip("结构层产物缺失，请先运行 python -m traffic_law_rag.kb.law_parser")
    return loaded


@pytest.fixture(scope="session")
def chunk_set(laws: list[LawDocument]) -> ChunkSet:
    """现场跑一遍切块（不读 chunks/*.jsonl），约 50ms。"""
    return LawChunker().chunk_all(laws)


@pytest.fixture(scope="session")
def chunk_stage(tmp_path_factory) -> ChunkStage:
    """把切块产物落到临时目录的 ChunkStage，用于测落盘/读回。"""
    tmp = tmp_path_factory.mktemp("chunks")
    return ChunkStage(
        chunks_path=tmp / "chunks.jsonl",
        parents_path=tmp / "parents.jsonl",
        verbose=False,
    )


@pytest.fixture(scope="session")
def law_by_id(laws: list[LawDocument]) -> dict[str, LawDocument]:
    return {law.law_id: law for law in laws}
