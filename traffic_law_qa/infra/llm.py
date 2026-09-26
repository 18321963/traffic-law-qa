from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from .. import config
from ..observability.tracer import Tracer
from ..ports import LLM

__all__ = ["OpenAILLM"]

_STREAM_OPTIONS = {"include_usage": True}
"""流式请求要 usage 块 —— 不加的话 `stream` 里那三个 token 计数永远缺席，是静默少字段。"""


def _arguments(call: Any) -> str:
    raw = getattr(call.function, "arguments", None)
    try:
        json.loads(raw)
    except (TypeError, ValueError):
        return "{}"
    return raw


def _usage_of(raw: Any) -> dict[str, int]:
    if raw is None:
        return {}
    return {
        "prompt_tokens": raw.prompt_tokens,
        "completion_tokens": raw.completion_tokens,
        "total_tokens": raw.total_tokens,
    }


def _normalize(response: Any) -> tuple[dict, dict]:
    message = response.choices[0].message
    reply: dict[str, Any] = {"role": "assistant", "content": message.content or ""}

    calls = getattr(message, "tool_calls", None) or []
    if calls:
        reply["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": _arguments(call),
                },
            }
            for call in calls
        ]

    return reply, _usage_of(getattr(response, "usage", None))


class OpenAILLM(LLM):
    def __init__(
        self,
        cfg: config.LLMConfig | None = None,
        *,
        retries: int = 2,
        timeout: float = 30.0,
        max_tokens: int = 1024,
        observer: Tracer | None = None,
        client: Any = None,
    ) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._observer = observer or Tracer()
        self._client = client

    @property
    def available(self) -> bool:
        return self.cfg.ready

    @property
    def model_name(self) -> str:
        return self.cfg.model

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        name: str = "llm.chat",
    ) -> tuple[dict, dict]:
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            extra["tool_choice"] = "auto"

        effective = self.cfg.temperature if temperature is None else temperature
        last_error: Exception | None = None
        with self._observer.observation(
            name,
            "generation",
            model=self.cfg.model,
            input=messages,
            model_parameters={"temperature": effective},
        ) as span:
            for attempt in range(1, self.retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,
                        temperature=effective,
                        timeout=self.timeout,
                        max_tokens=self.max_tokens,
                        **extra,
                    )
                    reply, usage = _normalize(response)
                    span.update(output=reply, usage_details=usage, metadata={"attempts": attempt})
                    return reply, usage
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < self.retries:
                        time.sleep(1.5 * attempt)
            span.update(level="ERROR", status_message=str(last_error))
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error

    def stream(
        self, messages: list[dict], *, temperature: float | None = None
    ) -> Iterator[tuple[str, Any]]:
        effective = self.cfg.temperature if temperature is None else temperature
        try:
            chunks = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=effective,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options=_STREAM_OPTIONS,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"调用 {self.cfg.model} 失败：{exc}") from exc

        for chunk in chunks:
            choices = getattr(chunk, "choices", None) or []
            delta = choices[0].delta.content if choices else None
            if delta:
                yield "delta", delta
            usage = _usage_of(getattr(chunk, "usage", None))
            if usage:
                yield "usage", usage
