from __future__ import annotations

from dataclasses import dataclass

from .disk import IndexStats

__all__ = [
    "StageReport",
    "PipelineReport",
    "CorpusStats",
    "ReadyState",
    "channel_label",
    "channel_state",
]


@dataclass(frozen=True)
class StageReport:

    name: str
    input_desc: str
    output_desc: str
    ok: bool
    elapsed_ms: float
    detail: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class PipelineReport:

    stages: tuple[StageReport, ...]
    law_count: int
    article_count: int
    chunk_count: int
    index: IndexStats | None = None

    @property
    def elapsed_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.stages)

    def render(self) -> str:
        width = max(len(s.name) for s in self.stages) if self.stages else 10
        lines = []
        for stage in self.stages:
            mark = "skip" if stage.skipped else ("ok" if stage.ok else "FAIL")
            lines.append(
                f"  [{mark:>4}] {stage.name:<{width}}  {stage.input_desc} → {stage.output_desc}"
                f"  {stage.elapsed_ms:>8.1f}ms  {stage.detail}"
            )
        lines.append(
            f"  法规 {self.law_count} 部 / 条文 {self.article_count} 条 / 索引块 {self.chunk_count} 个"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class CorpusStats:

    articles: int
    chunks: int
    dense: bool | None
    collection: str


@dataclass(frozen=True)
class ReadyState:

    action: str
    reason: str
    rows: int
    laws: int
    articles: int
    parents: int
    dense: bool
    milvus: str

    def describe(self) -> str:
        verb = "复用已有索引" if self.action == "reuse" else "已重建索引"
        channels = channel_label(self.dense)
        return (
            f"{verb}：{self.laws} 部法规 / {self.articles} 条 / {self.parents} 父块，"
            f"集合 {self.rows} 行（{channels}）｜{self.reason}｜Milvus {self.milvus}"
        )


def channel_label(dense: bool) -> str:
    return "稠密+BM25" if dense else "纯 BM25"


def channel_state(dense: bool | None) -> str:
    if dense is None:
        return "未知（Milvus 未连接）"
    return "已启用" if dense else "未启用（仅 BM25）"
