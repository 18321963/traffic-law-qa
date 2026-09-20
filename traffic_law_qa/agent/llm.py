"""带工具调用能力的 LLM 客户端（规划轮与审核轮专用）。

与 `AnswerGenerator` 分开，是因为两者的**层契约**不同：生成层的契约是
`Question + RetrievalResult → Answer`，不该被工具调用污染；这里只负责
「给定消息历史，返回下一条助手消息」。

**这个模块不依赖 langgraph** —— 它只用 `openai` + `config`。所以它可以被顶层 import，
不需要 `agent.graph` 那样的延迟导入（`tests/test_multihop.py` 钉着这条边界）。
"""

from __future__ import annotations

import time
from typing import Any

from .. import config

__all__ = ["ToolCallingLLM"]


def _normalize(response: Any) -> tuple[dict, dict]:
    """SDK 返回值 → (线上格式的助手消息, usage)。"""
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
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]

    usage: dict[str, int] = {}
    raw = getattr(response, "usage", None)
    if raw is not None:
        usage = {
            "prompt_tokens": raw.prompt_tokens,
            "completion_tokens": raw.completion_tokens,
            "total_tokens": raw.total_tokens,
        }
    return reply, usage


class ToolCallingLLM:
    def __init__(self, cfg: config.LLMConfig | None = None, *, retries: int = 2) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self._client = None

    @property
    def available(self) -> bool:
        return self.cfg.ready

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
    ) -> tuple[dict, dict]:
        """调一次模型，返回 (助手消息, usage)。重试策略与 generator 一致。"""
        from openai import OpenAI  # 延迟导入

        if self._client is None:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = tools
            # 刻意用 "auto" 而不是 "required"：一是不是所有兼容端点都支持 required
            # （百炼就不支持），二是本设计本来就需要模型能自主停下来。
            extra["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature if temperature is None else temperature,
                    **extra,
                )
                return _normalize(response)
            except Exception as exc:  # noqa: BLE001 - 统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error
