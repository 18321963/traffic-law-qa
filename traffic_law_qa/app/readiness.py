from __future__ import annotations

from .. import config
from ..contracts.errors import QaError
from ..contracts.reports import ReadyState
from ..ports import IndexBuilder, IndexStatus

__all__ = ["ReadyState", "ensure_ready"]


def ensure_ready(*, with_vector: bool = True, rebuild: bool = False) -> ReadyState:
    from ..container import build_pipeline, build_status

    if not config.source_files():
        raise QaError(f"知识库为空：{config.SOURCE_DIR} 下没有 {config.SOURCE_SUFFIXES} 文件")

    status: IndexStatus = build_status()
    version = status.ping()
    stats = status.snapshot()
    build_dense = config.embed_config().ready and (
        with_vector or bool(stats is not None and stats.vector_enabled)
    )

    reason = "指定了 rebuild=True" if rebuild else status.stale_reason(want_dense=build_dense)
    if reason is not None:
        print(f"[qa] 索引需要重建（{reason}），开始建库；首次约 30~60 秒…")
        builder: IndexBuilder = build_pipeline()
        builder.build(force=rebuild, with_vector=build_dense)
        stats = status.snapshot()
        if stats is None:
            raise QaError(f"建库未写出索引快照：{config.INDEX_META_PATH}")

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
    )
