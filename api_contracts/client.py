from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import httpx

from rag_contracts import config
from rag_contracts.domain.answer import Answer, Question
from rag_contracts.domain.errors import QaError, QaTimeout
from rag_contracts.domain.laws import LawInfo
from rag_contracts.domain.retrieval import (
    MATERIAL_TOP_K_DEFAULT,
    TOP_K_MAX,
    TOP_K_MIN,
    MaterialPassage,
    RetrievalResult,
    WebFinding,
)

__all__ = ["HttpTransport", "RagClient", "to_passages", "to_retrieval"]

DEFAULT_TIMEOUT = 60.0
MAX_DETAIL_CHARS = 500


class HttpTransport(Protocol):

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...


def _clamp_top_k(value: int | None) -> int:
    default = config.retrieve_config().top_k
    return min(TOP_K_MAX, max(TOP_K_MIN, default if value is None else value))


def to_retrieval(payload: dict) -> RetrievalResult:
    body = payload.get("retrieval")
    return RetrievalResult.from_dict(body if isinstance(body, dict) else payload)


def to_passages(payload: dict) -> tuple[MaterialPassage, ...]:
    return tuple(MaterialPassage.from_dict(item) for item in payload.get("passages") or ())


class RagClient:

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: HttpTransport | None = None,
        top_k: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self.top_k = _clamp_top_k(top_k)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RagClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        kwargs: dict = {"json": body, "params": params}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            effective = self.timeout if timeout is None else timeout
            raise QaTimeout(f"{method} {path} 超时（{effective:g} 秒）：{exc}") from None
        except httpx.HTTPError as exc:
            raise QaError(f"{method} {path} 连不上 {self.base_url}：{exc}") from None
        if resp.status_code >= 400:
            raise QaError(f"{method} {path} 返回 {resp.status_code}：{self._detail(resp)}")
        try:
            return resp.json()
        except ValueError:
            raise QaError(
                f"{method} {path} 返回的不是 JSON：{resp.text[:MAX_DETAIL_CHARS]}"
            ) from None

    @staticmethod
    def _detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:MAX_DETAIL_CHARS]
        detail = body.get("detail") if isinstance(body, dict) else body
        if isinstance(detail, str):
            return detail
        return json.dumps(detail, ensure_ascii=False)[:MAX_DETAIL_CHARS]

    def _relay(
        self,
        method: str,
        path: str,
        *,
        files: dict | None = None,
        data: dict | None = None,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict]:
        kwargs: dict = {"params": params}
        if files is not None:
            kwargs["files"] = files
        if data is not None:
            kwargs["data"] = data
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            effective = self.timeout if timeout is None else timeout
            raise QaTimeout(f"{method} {path} 超时（{effective:g} 秒）：{exc}") from None
        except httpx.HTTPError as exc:
            raise QaError(f"{method} {path} 连不上 {self.base_url}：{exc}") from None
        try:
            payload = resp.json()
        except ValueError:
            raise QaError(
                f"{method} {path} 返回的不是 JSON：{resp.text[:MAX_DETAIL_CHARS]}"
            ) from None
        return resp.status_code, payload

    def health(self) -> dict:
        return self.request("GET", "/health")

    def qa(
        self,
        question: str,
        *,
        mode: str = "search",
        top_k: int | None = None,
        pool: int | None = None,
        debug: bool = False,
        law_filter: Sequence[str] = (),
        timeout: float | None = None,
    ) -> dict:
        body: dict = {"question": question, "mode": mode}
        if top_k is not None:
            body["top_k"] = top_k
        if pool is not None:
            body["pool"] = pool
        if debug:
            body["debug"] = True
        if law_filter:
            body["law_filter"] = list(law_filter)
        return self.request("POST", "/qa", body=body, timeout=timeout)

    def article(
        self,
        *,
        article_no: str | None = None,
        law_name: str | None = None,
        text: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        body: dict = {}
        if article_no:
            body["article_no"] = article_no
        if law_name:
            body["law_name"] = law_name
        if text:
            body["text"] = text
        return self.request("POST", "/articles/lookup", body=body, timeout=timeout)

    def materials(
        self,
        query: str,
        doc_ids: Sequence[str] = (),
        *,
        top_k: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        body: dict = {"query": query, "doc_ids": list(doc_ids)}
        if top_k is not None:
            body["top_k"] = top_k
        return self.request("POST", "/materials/search", body=body, timeout=timeout)

    def upload_document(
        self,
        filename: str,
        content: bytes,
        *,
        mode: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict]:
        data = {"mode": mode} if mode else None
        return self._relay(
            "POST", "/documents", files={"file": (filename, content)}, data=data, timeout=timeout
        )

    def list_documents(
        self, *, mode: str | None = None, timeout: float | None = None
    ) -> tuple[int, dict]:
        params = {"mode": mode} if mode else None
        return self._relay("GET", "/documents", params=params, timeout=timeout)

    def delete_document(self, doc_id: str, *, timeout: float | None = None) -> tuple[int, dict]:
        return self._relay("DELETE", f"/documents/{doc_id}", timeout=timeout)

    def laws(self) -> tuple[LawInfo, ...]:
        payload = self.request("GET", "/laws")
        return tuple(LawInfo.from_dict(item) for item in payload.get("laws") or ())

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        law_filter: Sequence[str] = (),
        candidates: int | None = None,
        timeout: float | None = None,
    ) -> RetrievalResult:
        payload = self.qa(
            question,
            mode="search",
            top_k=top_k or self.top_k,
            pool=candidates,
            debug=channel_debug,
            law_filter=law_filter,
            timeout=timeout,
        )
        return to_retrieval(payload)

    def answer(
        self,
        question: str | Question,
        retrieval: RetrievalResult,
        *,
        timeliness: Sequence[WebFinding] = (),
        materials: Sequence[MaterialPassage] = (),
        timeout: float | None = None,
    ) -> Answer:
        asked = question if isinstance(question, Question) else Question(text=question)
        body = {
            "question": {
                "text": asked.text,
                "history": [[role, content] for role, content in asked.history],
            },
            "retrieval": retrieval.to_dict(),
            "timeliness": [item.to_dict() for item in timeliness],
            "materials": [item.to_dict() for item in materials],
        }
        return Answer.from_dict(self.request("POST", "/answer", body=body, timeout=timeout))

    def answer_stream(
        self,
        question: str | Question,
        retrieval: RetrievalResult,
        *,
        timeliness: Sequence[WebFinding] = (),
        materials: Sequence[MaterialPassage] = (),
    ) -> Iterator[tuple[str, Any]]:
        asked = question if isinstance(question, Question) else Question(text=question)
        body = {
            "question": {
                "text": asked.text,
                "history": [[role, content] for role, content in asked.history],
            },
            "retrieval": retrieval.to_dict(),
            "timeliness": [item.to_dict() for item in timeliness],
            "materials": [item.to_dict() for item in materials],
        }
        try:
            with self._client.stream("POST", "/answer/stream", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise QaError(
                        f"POST /answer/stream 返回 {resp.status_code}：{self._detail(resp)}"
                    )
                yield from self._answer_frames(resp)
        except httpx.HTTPError as exc:
            raise QaError(f"POST /answer/stream 连不上 {self.base_url}：{exc}") from None

    def _answer_frames(self, resp: httpx.Response) -> Iterator[tuple[str, Any]]:
        event = ""
        done = False
        for raw in resp.iter_lines():
            line = raw.strip()
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                payload = json.loads(line.removeprefix("data:").strip())
                if event == "delta":
                    yield "delta", str(payload.get("text") or "")
                elif event == "done":
                    done = True
                    yield "answer", Answer.from_dict(payload)
                elif event == "error":
                    raise QaError(str(payload.get("message") or ""))
        if not done:
            raise QaError("POST /answer/stream 的流结束了，却没有 done 帧（断流）")

    def ask(
        self,
        question: str | Question,
        top_k: int | None = None,
        *,
        channel_debug: bool = False,
        timeout: float | None = None,
    ) -> Answer:
        asked = question if isinstance(question, Question) else Question(text=question)
        retrieval = self.search(
            asked.text, top_k or asked.top_k, channel_debug=channel_debug, timeout=timeout
        )
        return self.answer(asked, retrieval, timeout=timeout)

    def get_article(
        self,
        article_no: str,
        law_name: str | None = None,
        *,
        timeout: float | None = None,
    ) -> tuple[RetrievalResult | None, str]:
        payload = self.article(article_no=article_no, law_name=law_name, timeout=timeout)
        note = str(payload.get("note") or "")
        if not payload.get("found"):
            return None, note
        return to_retrieval(payload), note

    def search_materials(
        self,
        query: str,
        doc_ids: Sequence[str] = (),
        *,
        top_k: int = MATERIAL_TOP_K_DEFAULT,
        timeout: float | None = None,
    ) -> tuple[tuple[MaterialPassage, ...], str]:
        payload = self.materials(query, doc_ids, top_k=top_k, timeout=timeout)
        return to_passages(payload), str(payload.get("text") or "")
