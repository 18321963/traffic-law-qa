from __future__ import annotations

import httpx

from .. import config
from ..contracts import WebFinding

__all__ = ["search_web"]

MAX_TITLE_CHARS = 80
MAX_SNIPPET_CHARS = 300

NO_KEY_NOTE = (
    "联网检索不可用（未配置 BOCHA_API_KEY）—— 本次只能按库内法条作答，"
    "不要再调用 web_search。"
)
HTTP_FAILED_NOTE = (
    "联网检索服务返回 {status}，本次没取到结果 —— 先按库内法条作答，"
    "并在答案里说明「是否有最新规定」未能确认。"
)
TIMEOUT_NOTE = (
    "联网检索超时（{timeout:g} 秒），本次没取到结果 —— 先按库内法条作答，"
    "不要重复调用 web_search。"
)
EMPTY_NOTE = (
    "联网检索没有命中任何网页。换一组更具体的说法再试一次（补上地区、年份、文件名）；"
    "再空就说明没有公开的新规定，就此停下作答。"
)
"""四种回执文案。分开写是因为模型**据此能做不同的事**：无 key 不必再试、超时不要重试、
服务端错先按库内作答、真空结果才值得换个说法再搜一次。四种都不抛异常 —— 工具回执必须
永远是一句中文，抛出去就成了整轮中断。"""


def _clip(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width] + "…"


def _pages(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    pages = (data.get("webPages") or {}).get("value")
    return [page for page in pages or [] if isinstance(page, dict)]


def _render(query: str, findings: list[WebFinding], start: int) -> str:
    if not findings:
        return f"网搜#{start}「{query}」｜命中 0 条\n{EMPTY_NOTE}"
    return "\n".join(
        [f"网搜#{start}「{query}」｜命中 {len(findings)} 条"] + [w.render() for w in findings]
    )


def search_web(
    query: str,
    *,
    start: int = 1,
    count: int | None = None,
    cfg: config.BochaConfig | None = None,
) -> tuple[list[WebFinding], str]:
    cfg = cfg or config.bocha_config()
    if not cfg.ready:
        return [], NO_KEY_NOTE

    try:
        with httpx.Client(timeout=cfg.timeout) as client:
            response = client.post(
                f"{cfg.base_url}/web-search",
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "freshness": cfg.freshness,
                    "summary": cfg.summary,
                    "count": count or cfg.count,
                },
            )
    except httpx.TimeoutException:
        return [], TIMEOUT_NOTE.format(timeout=cfg.timeout)
    except httpx.HTTPError as exc:
        return [], HTTP_FAILED_NOTE.format(status=type(exc).__name__)

    if response.status_code != 200:
        return [], HTTP_FAILED_NOTE.format(status=response.status_code)

    try:
        pages = _pages(response.json())
    except ValueError:
        return [], HTTP_FAILED_NOTE.format(status="非 JSON")

    findings: list[WebFinding] = []
    for page in pages:
        title = _clip(str(page.get("name") or ""), MAX_TITLE_CHARS)
        url = str(page.get("url") or "").strip()
        if not (title or url):
            continue
        findings.append(
            WebFinding(
                label=f"[时效{start + len(findings)}]",
                title=title,
                url=url,
                snippet=_clip(str(page.get("snippet") or ""), MAX_SNIPPET_CHARS),
                site=_clip(str(page.get("siteName") or ""), MAX_TITLE_CHARS),
                published=str(page.get("datePublished") or "")[:10],
            )
        )

    text = _render(query, findings, start)
    if findings:
        text += "\n（联网检索，非本库法条 —— 只能标 [时效N]，标成 [依据N] 会让整篇答案作废）"
    return findings, text
