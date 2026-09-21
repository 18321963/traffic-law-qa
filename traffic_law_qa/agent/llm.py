"""带工具调用能力的 LLM 客户端（规划轮与审核轮专用）。

与 `AnswerGenerator` 分开，是因为两者的**层契约**不同：生成层的契约是
`Question + RetrievalResult → Answer`，不该被工具调用污染；这里只负责
「给定消息历史，返回下一条助手消息」。

**这个模块不依赖 langgraph** —— 它只用 `openai` + `config` + 零依赖的 `obs`。所以它可以被
顶层 import，不需要 `agent.graph` 那样的延迟导入（`tests/test_multihop.py` 钉着这条边界）。
"""

from __future__ import annotations

import json
import time
from typing import Any

from .. import config
from ..obs import Tracer

__all__ = ["ToolCallingLLM"]


def _arguments(call: Any) -> str:
    """工具参数必须是**合法 JSON 字符串** —— 不是就把这一条降级成 `{}`。

    这是外部服务脏数据进系统的接缝。2026-09-20 实测 `qwen-turbo` 回过非 JSON 的
    `arguments`，服务端自己拒收：

        400 InternalError.Algo.InvalidParameter:
        The "function.arguments" parameter of the code model must be in JSON format.

    脏值一旦写进 `history` 就赖着不走了 —— 之后**每一次**请求都会带上它，整条题
    在重试耗尽后抛 RuntimeError 中断。宁可当空参数：`parse_tool_arguments` 会抛
    ValueError，节点把它转成「参数不合法，请修正后重试」的观察结果
    （`nodes.py:109`），模型下一轮自己会改。一次退化成空参，好过整条题报废。
    """
    raw = getattr(call.function, "arguments", None)
    try:
        json.loads(raw)
    except (TypeError, ValueError):
        return "{}"
    return raw


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
                    "arguments": _arguments(call),
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
    def __init__(
        self,
        cfg: config.LLMConfig | None = None,
        *,
        retries: int = 2,
        observer: Tracer | None = None,
    ) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self._observer = observer or Tracer()
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
        name: str = "llm.chat",
    ) -> tuple[dict, dict]:
        """调一次模型，返回 (助手消息, usage)。重试策略与 generator 一致。

        `name` 只是**观测里那一格的名字**：三个节点调的是同一个方法，云端得能一眼分开
        是入口判定、规划轮还是审核轮。默认值对得起不观测时的情形。

        整个重试循环包在一个 generation 观察里：失败的尝试没有单独的时间线，
        只把次数与最后一次的错误记进去 —— 这个模块要的是「这次调用了多久、花了多少 token」，
        不是一份失败史。
        """
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

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
                    response = self._client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,
                        temperature=effective,
                        **extra,
                    )
                    reply, usage = _normalize(response)
                    span.update(output=reply, usage_details=usage, metadata={"attempts": attempt})
                    return reply, usage
                except Exception as exc:  # noqa: BLE001 - 统一重试
                    last_error = exc
                    if attempt < self.retries:
                        time.sleep(1.5 * attempt)
            span.update(level="ERROR", status_message=str(last_error))
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error
