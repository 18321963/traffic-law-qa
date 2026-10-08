from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from rag_contracts import config
from rag_contracts.domain.disk import Chunk, ChunkSet, IndexStats, ParentChunk
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.reports import WEIGHTS_MISMATCH, WEIGHTS_OK, WEIGHTS_UNKNOWN
from rag_service import container
from rag_service.adapters.embedding import LocalEmbedder
from rag_service.adapters.local_model import weights_fingerprint
from rag_service.indexing.indexer import Indexer
from rag_service.indexing.status import CorpusStatus

STAMP = 1_700_000_000_000_000_000
TEXT = "饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。"


class _Embedder:

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    @property
    def model_label(self) -> str:
        return "内存嵌入"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


class _Indexer:

    def __init__(self, stats: IndexStats | None) -> None:
        self.stats = stats

    def load_stats(self) -> IndexStats | None:
        return self.stats


class _Store:

    uri = "memory://演练"
    collection = "内存集合"

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def ping(self) -> str:
        return "memory-1.0"

    def recreate(self, *, dim: int | None = None) -> dict:
        return {"dim": dim}

    def insert(self, rows: list[dict], *, batch_size: int = 200) -> int:
        self.rows.extend(rows)
        return len(rows)

    def count(self) -> int:
        return len(self.rows)


class _Library:

    def manifest(self) -> dict:
        return {"laws": []}


class _Pipeline:

    def __init__(self, indexer: _Indexer, fingerprint: str) -> None:
        self._indexer = indexer
        self._fingerprint = fingerprint
        self.calls: list[bool] = []

    def build(self, *, force: bool = False, with_vector: bool = True) -> None:
        self.calls.append(force)
        if self._indexer.stats is not None:
            self._indexer.stats = replace(
                self._indexer.stats, embedding_fingerprint=self._fingerprint
            )


def _stats(fingerprint: str | None, *, vector: bool = True) -> IndexStats:
    return IndexStats(
        collection="内存集合",
        uri="memory://演练",
        rows=1,
        dense_rows=1,
        sparse_rows=1,
        embedding_model="bge-m3",
        embedding_dim=2,
        vector_enabled=vector,
        built_at="2026-09-28T00:00:00+00:00",
        elapsed_ms=1.0,
        embedding_fingerprint=fingerprint,
    )


def _status(indexer: _Indexer, fingerprint: str) -> CorpusStatus:
    return CorpusStatus(
        store=_Store(),
        indexer=indexer,
        embedder=_Embedder(fingerprint),
        library=_Library(),
    )


def _stamp(path: Path, moment: int = STAMP) -> None:
    os.utime(path, ns=(moment, moment))


def _chunk_set() -> ChunkSet:
    parent = ParentChunk(
        parent_id="road#第九十一条",
        law_id="road",
        law_name="中华人民共和国道路交通安全法",
        version="2021",
        citation="《中华人民共和国道路交通安全法》",
        article_no="第九十一条",
        article_index=91,
        chapter=None,
        section=None,
        text=TEXT,
    )
    chunk = Chunk(
        chunk_id="c1",
        parent_id=parent.parent_id,
        law_id=parent.law_id,
        law_name=parent.law_name,
        version=parent.version,
        citation=parent.citation,
        article_no=parent.article_no,
        article_index=parent.article_index,
        part_index=0,
        part_total=1,
        text=TEXT,
        embed_text=TEXT,
    )
    return ChunkSet(parents=(parent,), chunks=(chunk,))


def test_a_rebuild_writes_the_fingerprint_of_the_weights_it_embedded_with(tmp_path: Path) -> None:
    indexer = Indexer(
        store=_Store(),
        embedder=_Embedder("当前的"),
        meta_path=tmp_path / "index_meta.json",
        verbose=False,
    )

    stats = indexer.build(_chunk_set(), with_vector=True)

    assert stats.embedding_fingerprint == "当前的"
    written = indexer.load_stats()
    assert written is not None and written.embedding_fingerprint == "当前的"
    assert _status(_Indexer(written), "当前的").weights() == WEIGHTS_OK
    assert _status(_Indexer(written), "换过的").weights() == WEIGHTS_MISMATCH


def test_a_snapshot_written_before_the_fingerprint_existed_still_loads(tmp_path: Path) -> None:
    meta = tmp_path / "index_meta.json"
    meta.write_text(
        json.dumps(
            {
                "stats": {
                    "collection": "traffic_law",
                    "uri": "http://localhost:19530",
                    "rows": 812,
                    "dense_rows": 812,
                    "sparse_rows": 812,
                    "embedding_model": "bge-m3",
                    "embedding_dim": 1024,
                    "vector_enabled": True,
                    "built_at": "2026-09-20T00:00:00+00:00",
                    "elapsed_ms": 1.0,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    stats = Indexer(
        store=_Store(), embedder=_Embedder("当前的"), meta_path=meta, verbose=False
    ).load_stats()

    assert stats is not None and stats.embedding_fingerprint is None
    assert _status(_Indexer(stats), "当前的").weights() == WEIGHTS_UNKNOWN
    assert _status(_Indexer(stats), "当前的").blocked_reason() is None


def test_the_fingerprint_follows_the_file_list_not_the_bytes(tmp_path: Path) -> None:
    weights = tmp_path / "bge-m3"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    (weights / "model.bin").write_bytes(b"0" * 32)
    _stamp(weights / "config.json")
    _stamp(weights / "model.bin")

    first = weights_fingerprint(str(weights))
    assert first and first == weights_fingerprint(str(weights))
    assert weights_fingerprint(str(tmp_path / "还没下")) == ""

    (weights / "model.bin").write_bytes(b"0" * 33)
    _stamp(weights / "model.bin")
    assert weights_fingerprint(str(weights)) != first, "换了权重文件还不翻新，指纹就白算了"

    twin = tmp_path / "另一份" / "bge-m3"
    twin.mkdir(parents=True)
    (twin / "config.json").write_text("{}", encoding="utf-8")
    (twin / "model.bin").write_bytes(b"9" * 33)
    _stamp(twin / "config.json")
    _stamp(twin / "model.bin")
    assert weights_fingerprint(str(twin)) == weights_fingerprint(str(weights)), (
        "清单、大小、mtime 摆成一样、只有字节不同：这就是「防误换不防篡改」的边界，"
        "要内容级校验得另做离线脚本"
    )


def test_the_embedder_hands_out_the_fingerprint_of_its_weights_dir(tmp_path: Path) -> None:
    weights = tmp_path / "bge-m3"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")

    assert LocalEmbedder(config.EmbedConfig(model=str(weights))).fingerprint == (
        weights_fingerprint(str(weights))
    )
    assert LocalEmbedder(config.EmbedConfig(model="BAAI/bge-m3")).fingerprint == ""


def test_a_snapshot_without_a_fingerprint_is_unknown_and_does_not_block() -> None:
    assert _status(_Indexer(_stats(None)), "当前").weights() == WEIGHTS_UNKNOWN
    assert _status(_Indexer(_stats(None)), "当前").blocked_reason() is None
    assert _status(_Indexer(_stats("旧", vector=False)), "当前").weights() == WEIGHTS_UNKNOWN
    assert _status(_Indexer(_stats("旧")), "").weights() == WEIGHTS_UNKNOWN


def test_a_matching_fingerprint_passes_and_a_mismatch_names_both() -> None:
    assert _status(_Indexer(_stats("同一个")), "同一个").weights() == WEIGHTS_OK
    assert _status(_Indexer(_stats("同一个")), "同一个").blocked_reason() is None

    stale = _status(_Indexer(_stats("旧的")), "当前的")
    assert stale.weights() == WEIGHTS_MISMATCH
    reason = stale.blocked_reason()
    assert reason is not None and "旧的" in reason and "当前的" in reason


def _wire(monkeypatch, fingerprint: str, recorded: str | None = "旧的") -> _Pipeline:
    indexer = _Indexer(_stats(recorded))
    pipeline = _Pipeline(indexer, fingerprint)
    monkeypatch.setattr(config, "source_files", lambda *args, **kwargs: [Path("道交法.docx")])
    monkeypatch.setattr(container, "build_status", lambda **kwargs: _status(indexer, fingerprint))
    monkeypatch.setattr(container, "build_pipeline", lambda **kwargs: pipeline)
    monkeypatch.delenv("RAG_ALLOW_REBUILD", raising=False)
    return pipeline


def test_a_mismatched_fingerprint_stops_the_boot_until_someone_asks_for_a_rebuild(
    monkeypatch,
) -> None:
    pipeline = _wire(monkeypatch, "当前的")

    with pytest.raises(QaError) as caught:
        container.readiness()
    assert "旧的" in str(caught.value) and "当前的" in str(caught.value)
    assert pipeline.calls == [], "指纹不匹配时不许自己悄悄重建，要人显式开口"

    state = container.readiness(rebuild=True)
    assert state.action == "rebuild" and state.reason == "指定了 rebuild=True"
    assert pipeline.calls == [True]
    assert state.weights == WEIGHTS_OK, "重建把当前指纹写进了快照，下一次启动就不该再拦"


def test_the_env_flag_lifts_the_block_by_rebuilding_with_the_current_fingerprint(
    monkeypatch,
) -> None:
    pipeline = _wire(monkeypatch, "当前的")
    monkeypatch.setenv("RAG_ALLOW_REBUILD", "1")

    state = container.readiness()

    assert state.action == "rebuild" and state.reason == "RAG_ALLOW_REBUILD=1"
    assert pipeline.calls == [False], "环境级放行是「重建一次」，不是带着旧索引硬跑"
    assert state.weights == WEIGHTS_OK


def test_a_missing_fingerprint_keeps_serving_and_reports_degraded(monkeypatch) -> None:
    pipeline = _wire(monkeypatch, "当前的", None)
    monkeypatch.setattr(CorpusStatus, "stale_reason", lambda self, **kwargs: None)

    state = container.readiness()

    assert state.action == "reuse" and state.weights == WEIGHTS_UNKNOWN
    assert pipeline.calls == [], "指纹没记录只是没法核对，不是重建的理由"
