from __future__ import annotations

import json

import httpx
import pytest

from traffic_law_qa.infra.websearch import search_web

QUERY = "深圳 电动自行车 最新规定"

BOCHA_JSON = {
    "data": {
        "webPages": {
            "value": [
                {
                    "name": "深圳电动自行车管理条例 2026 年修订",
                    "url": "https://example.test/sz-ebike",
                    "snippet": "自 2026 年 1 月 1 日起，深圳经济特区电动自行车未登记上路的，处警告或者五十元罚款。",
                    "datePublished": "2026-01-01T00:00:00+08:00",
                    "siteName": "深圳市政府",
                },
                {
                    "name": "电动自行车新规解读",
                    "url": "https://example.test/read",
                    "snippet": "解读文章。",
                    "datePublished": "2026-02-03T00:00:00+08:00",
                    "siteName": "示例网",
                },
            ]
        }
    }
}


def _install_bocha(
    monkeypatch, *, key: str = "stub-key", payload=None, status=200, error=None
) -> list[dict]:
    monkeypatch.setenv("BOCHA_API_KEY", key)
    sent: list[dict] = []
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("Authorization"),
                "body": json.loads(request.content),
            }
        )
        if error is not None:
            raise error
        return httpx.Response(status, json=payload)

    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)
    return sent


def test_request_body_speaks_the_bocha_contract(monkeypatch) -> None:
    sent = _install_bocha(monkeypatch, payload=BOCHA_JSON)

    search_web(QUERY)

    assert len(sent) == 1
    assert sent[0]["url"].endswith("/web-search")
    assert sent[0]["auth"] == "Bearer stub-key"
    assert sent[0]["body"] == {
        "query": QUERY,
        "freshness": "noLimit",
        "summary": True,
        "count": 5,
    }


def test_count_argument_overrides_the_configured_default(monkeypatch) -> None:
    sent = _install_bocha(monkeypatch, payload=BOCHA_JSON)

    search_web(QUERY, count=8)

    assert sent[0]["body"]["count"] == 8


def test_findings_are_labelled_timeliness_not_evidence(monkeypatch) -> None:
    _install_bocha(monkeypatch, payload=BOCHA_JSON)

    findings, text = search_web(QUERY, start=3)

    assert [w.label for w in findings] == ["[时效3]", "[时效4]"]
    assert findings[0].title == "深圳电动自行车管理条例 2026 年修订"
    assert findings[0].url == "https://example.test/sz-ebike"
    assert findings[0].site == "深圳市政府"
    assert findings[0].published == "2026-01-01"
    assert text.startswith(f"网搜#3「{QUERY}」｜命中 2 条")
    assert "[时效3]" in text and "[时效4]" in text
    assert "非本库法条" in text


def test_empty_result_is_a_receipt_not_a_failure(monkeypatch) -> None:
    _install_bocha(monkeypatch, payload={"data": {"webPages": {"value": []}}})

    findings, text = search_web(QUERY)

    assert findings == []
    assert text.startswith(f"网搜#1「{QUERY}」｜命中 0 条")
    assert "没有命中任何网页" in text


def test_missing_key_never_calls_out(monkeypatch) -> None:
    sent = _install_bocha(monkeypatch, key="", payload=BOCHA_JSON)

    findings, text = search_web(QUERY)

    assert sent == []
    assert findings == []
    assert "联网检索不可用" in text


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"status": 500, "payload": {"error": "boom"}}, "联网检索服务返回 500"),
        ({"status": 200, "payload": None}, "联网检索服务返回 非 JSON"),
        ({"error": httpx.ConnectTimeout("打不通")}, "联网检索超时"),
    ],
    ids=["http-500", "non-json", "timeout"],
)
def test_transport_failures_return_a_note_instead_of_raising(monkeypatch, kwargs, expected) -> None:
    _install_bocha(monkeypatch, **kwargs)

    findings, text = search_web(QUERY)

    assert findings == []
    assert text.startswith(expected)
    assert "本次没取到结果" in text
