from __future__ import annotations

from rag_contracts import config
from rag_contracts.domain.errors import QaError
from rag_contracts.domain.reports import WEIGHTS_UNKNOWN, ReadyState
from rag_contracts.ports import IndexBuilder, IndexStatus

__all__ = ["ReadyState", "ensure_ready"]

_weights_announced = False


def _announce_weights(weights: str) -> None:
    global _weights_announced
    if _weights_announced:
        return
    _weights_announced = True
    if weights == WEIGHTS_UNKNOWN:
        print("[qa] 权重指纹没核对上（快照未记录，或权重目录读不到）：照常服务，/health 会标 degraded")


def ensure_ready(*, with_vector: bool = True, rebuild: bool = False) -> ReadyState:
    from ..container import build_pipeline, build_status

    if not config.source_files():
        raise QaError(
            f"知识库为空：{config.SOURCE_DIR} 与 {config.PDF_DIR} 下没有 {config.SOURCE_SUFFIXES} 文件"
        )

    status: IndexStatus = build_status()
    version = status.ping()
    stats = status.snapshot()
    build_dense = config.embed_config().ready and (
        with_vector or bool(stats is not None and stats.vector_enabled)
    )

    block = status.blocked_reason()
    lifted = block is not None and config.allow_rebuild()
    if lifted:
        print("[qa] RAG_ALLOW_REBUILD=1：权重对不上也放行，按当前配置重建")
    elif block is not None and not rebuild:
        raise QaError(block)

    if rebuild:
        reason = "指定了 rebuild=True"
    elif lifted:
        reason = "RAG_ALLOW_REBUILD=1"
    else:
        reason = status.stale_reason(want_dense=build_dense)
    if reason is not None:
        print(f"[qa] 索引需要重建（{reason}），开始建库；首次约 30~60 秒…")
        builder: IndexBuilder = build_pipeline()
        builder.build(force=rebuild, with_vector=build_dense)
        stats = status.snapshot()
        if stats is None:
            raise QaError(f"建库未写出索引快照：{config.INDEX_META_PATH}")

    weights = status.weights()
    _announce_weights(weights)

    laws, articles = status.counts()
    return ReadyState(
        action="rebuild" if reason is not None else "reuse",
        reason=reason or "docx、本地产物与集合三者一致",
        rows=stats.rows,
        laws=laws,
        articles=articles,
        parents=articles,
        dense=stats.vector_enabled,
        milvus=version,
        weights=weights,
    )
