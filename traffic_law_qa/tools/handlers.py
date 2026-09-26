from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import config
from ..contracts.disk import ParentChunk
from ..contracts.retrieval import MaterialPassage, RetrievalResult, WebFinding
from ..infra.websearch import search_web as fetch_web
from ..observability.tracer import Tracer
from ..ports import RagService
from ..search.articles import lookup_article, resolve_law_id
from ..search.materials import search_materials as score_materials
from .arguments import (
    parse_article_arguments,
    parse_material_arguments,
    parse_tool_arguments,
    parse_web_arguments,
)
from .render import render_tool_result
from .schemas import (
    GET_ARTICLE_NAME,
    MATERIAL_TOP_K_DEFAULT,
    SEARCH_LAW_NAME,
    SEARCH_MATERIALS_NAME,
    WEB_SEARCH_NAME,
)

__all__ = ["ToolCall", "ToolEnv", "ToolOutcome", "HANDLERS", "retrieval_digest"]

NO_FRESH_EVIDENCE = (
    "⚠ 本轮一条新证据都没取到（命中的条上一轮都已经给过你了）。同一个方向换词换不出新东西 ——"
    "要么改问你真正缺的那个要素，要么就此停下作答。\n"
)

NO_FRESH_MATERIAL = (
    "⚠ 这些段上一轮已经给过你了，材料里没有别的相关内容。就此停下作答，或换个说法再试一次。\n"
)

_BAD_ARGUMENTS = "参数不合法：{}。请修正后重试。"


@dataclass(frozen=True)
class ToolEnv:
    rag: RagService
    index: dict[tuple[str, int], ParentChunk]
    cfg: config.AgentConfig
    observer: Tracer
    materials: Callable[[], tuple[MaterialPassage, ...]]


@dataclass(frozen=True)
class ToolCall:
    arguments: str
    retrieval_no: int
    web_no: int
    seen: frozenset[str]
    seen_labels: frozenset[str]
    region_scope: tuple[str, ...]
    default_top_k: int


@dataclass(frozen=True)
class ToolOutcome:
    text: str
    retrieval: RetrievalResult | None = None
    findings: tuple[WebFinding, ...] = ()
    passages: tuple[MaterialPassage, ...] = ()


def retrieval_digest(result: RetrievalResult) -> dict:
    return {
        "命中": len(result.articles),
        "向量": result.used_vector,
        "BM25": result.used_bm25,
        "耗时ms": round(result.elapsed_ms, 1),
        "头三条": result.citations()[:3],
    }


def search_law(env: ToolEnv, call: ToolCall) -> ToolOutcome:
    try:
        query, law_name, top_k = parse_tool_arguments(
            call.arguments, default_top_k=call.default_top_k
        )
    except ValueError as exc:
        return ToolOutcome(_BAD_ARGUMENTS.format(exc))

    kwargs: dict[str, Any] = {}
    if law_name:
        law_id, known_names = resolve_law_id(law_name, env.rag.parents)
        if law_id is None:
            return ToolOutcome(
                f"无法确定法规「{law_name}」。库内只有这几部，请从中选一个"
                f"（或省略 law_name 检索全部）：{'；'.join(known_names)}"
            )
        kwargs["law_filter"] = (law_id,)
    elif call.region_scope:
        kwargs["law_filter"] = call.region_scope

    try:
        with env.observer.observation(
            "检索",
            "retriever",
            input={
                "query": query,
                "top_k": top_k,
                "law_filter": kwargs.get("law_filter"),
            },
        ) as span:
            result = env.rag.search(query, top_k=top_k, **kwargs)
            match_text = env.rag.expand(query)
            span.update(output=retrieval_digest(result))
    except Exception as exc:  # noqa: BLE001
        return ToolOutcome(f"检索失败：{exc}。可以换一组关键词再试，或用更通用的说法。")

    text = render_tool_result(
        result,
        index=call.retrieval_no,
        seen=call.seen,
        snippet_chars=env.cfg.snippet_chars,
        match_text=match_text,
    )
    if not {hit.article.parent_id for hit in result.articles} - call.seen:
        text = NO_FRESH_EVIDENCE + text
    return ToolOutcome(text, retrieval=result)


def get_article(env: ToolEnv, call: ToolCall) -> ToolOutcome:
    try:
        article_no, law_name = parse_article_arguments(call.arguments)
    except ValueError as exc:
        return ToolOutcome(_BAD_ARGUMENTS.format(exc))

    with env.observer.observation(
        "精确取条",
        "retriever",
        input={"article_no": article_no, "law_name": law_name},
    ) as span:
        result, error = lookup_article(
            article_no, law_name, parents=env.rag.parents, index=env.index
        )
        if result is not None:
            span.update(output=retrieval_digest(result))
    if result is None:
        return ToolOutcome(error)
    return ToolOutcome(
        render_tool_result(
            result,
            index=call.retrieval_no,
            seen=call.seen,
            snippet_chars=env.cfg.article_chars,
            match_text="",
        ),
        retrieval=result,
    )


def web_search(env: ToolEnv, call: ToolCall) -> ToolOutcome:
    try:
        query, count = parse_web_arguments(
            call.arguments, default_count=config.bocha_config().count
        )
    except ValueError as exc:
        return ToolOutcome(_BAD_ARGUMENTS.format(exc))

    with env.observer.observation(
        "联网检索",
        "retriever",
        input={"query": query, "count": count},
    ) as span:
        findings, text = fetch_web(query, start=call.web_no, count=count)
        span.update(output={"命中": len(findings), "头三条": [w.title for w in findings[:3]]})
    return ToolOutcome(text, findings=tuple(findings))


def search_materials(env: ToolEnv, call: ToolCall) -> ToolOutcome:
    try:
        query, top_k = parse_material_arguments(
            call.arguments, default_top_k=MATERIAL_TOP_K_DEFAULT
        )
    except ValueError as exc:
        return ToolOutcome(_BAD_ARGUMENTS.format(exc))

    with env.observer.observation(
        "材料检索", "retriever", input={"query": query, "top_k": top_k}
    ) as span:
        hits, text = score_materials(query, env.materials(), top_k=top_k)
        span.update(
            output={
                "命中": len(hits),
                "材料份数": len({hit.doc_id for hit in hits}),
                "段": [hit.citation for hit in hits[:3]],
            }
        )
    fresh = tuple(hit for hit in hits if hit.label not in call.seen_labels)
    if hits and not fresh:
        text = NO_FRESH_MATERIAL + text
    return ToolOutcome(text, passages=fresh)


HANDLERS: dict[str, Callable[[ToolEnv, ToolCall], ToolOutcome]] = {
    SEARCH_LAW_NAME: search_law,
    GET_ARTICLE_NAME: get_article,
    WEB_SEARCH_NAME: web_search,
    SEARCH_MATERIALS_NAME: search_materials,
}
