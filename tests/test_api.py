"""`api._stale_reason` 的离线用例 —— 只测「要不要重建索引」这一个判断。

判错的两个方向代价不对称，所以两个方向都要钉：

- **漏判**：索引快照记着旧向量模型，但这个函数说「不用重建」。于是查询用新模型
  编码、库里躺着旧模型的向量，召回质量**静默**劣化 —— 不报错、不失败，只是变差。
  换模型时尤其危险：text-embedding-v4 与 bge-large-zh-v1.5 都是 1024 维，
  行数和 vector_enabled 都不变，前面所有检查都会放行。
- **误判**：模型明明一致却说要重建。表现是每问一句都先全库重建一遍。

`法规知识库/chunks/` 被 .gitignore 忽略、CI 上不存在，而本函数要数父块行数，
所以下面的 fixture 把这两个路径指到临时文件。
"""

from __future__ import annotations

import pytest

from traffic_law_rag import api, config
from traffic_law_rag.contracts import IndexStats
from traffic_law_rag.kb.indexer import EmbeddingClient
from traffic_law_rag.kb.law_parser import LawLibrary


class _假Store:
    """只提供 `_stale_reason` 真正用到的两样东西；不连 Milvus。"""

    collection = "traffic_law"

    def __init__(self, rows: int) -> None:
        self._rows = rows

    def has_collection(self) -> bool:
        return True

    def count(self) -> int:
        return self._rows


@pytest.fixture
def 落盘产物(tmp_path, monkeypatch):
    """把 chunks/parents 指到临时文件，父块行数取自清单 —— 不硬编码 508。

    行数必须与清单条文数**相等**，否则先撞上「父块数不一致」那条，
    被测的向量模型分支根本走不到。
    """
    articles = sum(int(item["articles"]) for item in LawLibrary().manifest()["laws"])
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("", encoding="utf-8")
    parents = tmp_path / "parents.jsonl"
    parents.write_text("{}\n" * articles, encoding="utf-8")
    monkeypatch.setattr(config, "CHUNKS_PATH", chunks)
    monkeypatch.setattr(config, "PARENTS_PATH", parents)


def _快照(rows: int, model: str) -> IndexStats:
    """一份「已经建好稠密索引」的快照。"""
    return IndexStats(
        collection="traffic_law",
        uri="http://localhost:19530",
        rows=rows,
        dense_rows=rows,
        sparse_rows=rows,
        embedding_model=model,
        embedding_dim=1024,
        vector_enabled=True,
        built_at="2026-09-19T00:00:00+00:00",
        elapsed_ms=1.0,
    )


def test_换向量模型会被判为过期(落盘产物, chunk_set):
    """维度撞车也拦得住 —— 这条是切本地 bge 时补上的。"""
    rows = len(chunk_set.chunks)
    stats = _快照(rows, "text-embedding-v4@https://dashscope.aliyuncs.com/compatible-mode/v1")

    reason = api._stale_reason(_假Store(rows), stats, want_dense=True)

    assert reason is not None, "换了向量模型却没报过期，旧向量会被静默继续用"
    assert "text-embedding-v4" in reason, "原因里要写清楚是哪个旧模型，否则看不懂为什么要重建"


def test_向量模型一致时不重建(落盘产物, chunk_set):
    """反方向。拿**当前配置**的 model_label 去比，所以以后再换模型这条也不会腐坏。"""
    rows = len(chunk_set.chunks)
    stats = _快照(rows, EmbeddingClient().model_label)

    assert api._stale_reason(_假Store(rows), stats, want_dense=True) is None
