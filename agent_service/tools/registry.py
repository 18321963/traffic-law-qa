from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .handlers import (
    ToolCall,
    ToolEnv,
    ToolOutcome,
    get_article,
    search_law,
    search_materials,
    web_search,
)
from .schemas import (
    GET_ARTICLE_NAME,
    GET_ARTICLE_TOOL,
    SEARCH_LAW_NAME,
    SEARCH_LAW_TOOL,
    SEARCH_MATERIALS_NAME,
    SEARCH_MATERIALS_TOOL,
    WEB_SEARCH_NAME,
    WEB_SEARCH_TOOL,
)

__all__ = ["ToolSpec", "TOOL_REGISTRY", "HANDLERS", "TOOLS"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict[str, Any]
    handler: Callable[[ToolEnv, ToolCall], ToolOutcome]
    llm_visible: bool = True


TOOL_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(SEARCH_LAW_NAME, SEARCH_LAW_TOOL, search_law),
    ToolSpec(GET_ARTICLE_NAME, GET_ARTICLE_TOOL, get_article),
    ToolSpec(SEARCH_MATERIALS_NAME, SEARCH_MATERIALS_TOOL, search_materials),
    ToolSpec(WEB_SEARCH_NAME, WEB_SEARCH_TOOL, web_search, llm_visible=False),
)

HANDLERS: dict[str, Callable[[ToolEnv, ToolCall], ToolOutcome]] = {
    spec.name: spec.handler for spec in TOOL_REGISTRY
}

TOOLS: list[dict[str, Any]] = [spec.schema for spec in TOOL_REGISTRY if spec.llm_visible]
