from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from api_contracts.client import RagClient
from rag_contracts.domain.errors import QaError

__all__ = ["TOOLS", "ToolSpec", "call_tool", "tool_names"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[RagClient, dict[str, Any]], dict[str, Any]]


def _search(client: RagClient, args: dict[str, Any]) -> dict:
    return client.qa(
        args["question"],
        mode="search",
        top_k=args.get("top_k"),
        law_filter=args.get("law_filter") or (),
        debug=bool(args.get("debug")),
    )


def _ask(client: RagClient, args: dict[str, Any]) -> dict:
    return client.qa(
        args["question"],
        mode="ask",
        top_k=args.get("top_k"),
        law_filter=args.get("law_filter") or (),
    )


def _article(client: RagClient, args: dict[str, Any]) -> dict:
    return client.article(
        article_no=args.get("article_no"),
        law_name=args.get("law_name"),
        text=args.get("text"),
    )


def _laws(client: RagClient, args: dict[str, Any]) -> dict:
    return client.request("GET", "/laws")


def _materials(client: RagClient, args: dict[str, Any]) -> dict:
    return client.materials(args["query"], args.get("doc_ids") or (), top_k=args.get("top_k"))


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="rag_search",
        description=(
            "在交通法规知识库里检索法条，返回命中的条文（法规名、条号、原文、相关条提示）与检索统计。"
            "只查不答，不花大模型调用。要拿候选条号、核实某条规定、收集作答依据时用它；"
            "已经知道是第几条就直接用 rag_get_article。"
        ),
        schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "检索词。第一轮直接用用户原话 —— 检索层会做口语对齐（醉驾→醉酒驾驶）"
                        "与混合召回，拆成关键词反而稀释信号。续查时只写要补的那一块。"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "返回条数，默认 6。",
                },
                "law_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "只在给定的 law_id 里检索（law_id 从 rag_laws 拿）。不给就全库检索。",
                },
                "debug": {
                    "type": "boolean",
                    "description": (
                        "true 时每条多带通道诊断（向量名次、BM25 名次、原始分）。"
                        "排查「为什么没召回」时才开。"
                    ),
                },
            },
            "required": ["question"],
        },
        handler=_search,
    ),
    ToolSpec(
        name="rag_ask",
        description=(
            "把整个问题交给服务端：它自己检索并生成一篇带引用标记的答案。要花一次大模型调用。"
            "你要自己控制检索过程时用 rag_search，只想直接拿结论时用它。"
        ),
        schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的问题。原话直接给，别先改成关键词。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "检索条数，默认 6。",
                },
                "law_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "只在给定的 law_id 里检索（law_id 从 rag_laws 拿）。不给就全库检索。",
                },
            },
            "required": ["question"],
        },
        handler=_ask,
    ),
    ToolSpec(
        name="rag_get_article",
        description=(
            "按条号精确取一条法条原文。知道要哪一条时用它，比检索准。"
            "同一个条号在多部法规里都有时，会返回候选清单，配上 law_name 再问一次即可；"
            "查不到不报错，返回 found=false 与原因说明。"
        ),
        schema={
            "type": "object",
            "properties": {
                "article_no": {
                    "type": "string",
                    "description": "条号，中英文数字都行，例如「第九十条」或「90」。",
                },
                "law_name": {
                    "type": "string",
                    "description": "可选。法规名，例如「中华人民共和国道路交通安全法」；多部法规都有这个条号时必须给。",
                },
                "text": {
                    "type": "string",
                    "description": "也可以给一句话、条号写在里面（与 article_no 二选一，给了 article_no 就按它走）。",
                },
            },
            "required": ["article_no"],
        },
        handler=_article,
    ),
    ToolSpec(
        name="rag_laws",
        description=(
            "列出知识库里全部法规（law_id、名称、版本、条数）。"
            "要按法规过滤（rag_search 的 law_filter）之前，先来这里取 law_id。"
        ),
        schema={"type": "object", "properties": {}},
        handler=_laws,
    ),
    ToolSpec(
        name="rag_search_materials",
        description=(
            "在用户本次上传的材料（会话材料）里检索，返回命中的段落与拼好的文本。"
            "它只查材料、不查法条 —— 查法条用 rag_search。"
        ),
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索词。用材料里真出现过的说法，别用同义词改写。",
                },
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要检索的材料 doc_id 列表（上传材料时返回）。给不全就等于没材料可查。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "返回段数，默认 5。",
                },
            },
            "required": ["query", "doc_ids"],
        },
        handler=_materials,
    ),
)

_BY_NAME = {spec.name: spec for spec in TOOLS}


def tool_names() -> list[str]:
    return [spec.name for spec in TOOLS]


def call_tool(client: RagClient, name: str, arguments: dict[str, Any] | None = None) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise QaError(f"没有这个工具：{name}（可用：{'、'.join(tool_names())}）")
    try:
        return spec.handler(client, arguments or {})
    except KeyError as exc:
        raise QaError(f"{name} 缺必填参数：{exc.args[0]}") from None
