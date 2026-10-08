from __future__ import annotations

from types import SimpleNamespace

from rag_contracts import config
from rag_contracts.llm import OpenAILLM
from rag_contracts.ports import TRUNCATED_FINISH_REASON

CFG = config.LLMConfig(base_url="http://stub", api_key="stub", model="stub-model")


class _Completions:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        return iter(item) if kwargs.get("stream") else item


class _Client:
    def __init__(self, script: list) -> None:
        self.completions = _Completions(script)
        self.chat = SimpleNamespace(completions=self.completions)


def _response(content: str, finish: str | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2, total_tokens=13),
    )


def _chunk(content: str | None, finish: str | None) -> SimpleNamespace:
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _usage_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


def test_chat_hands_back_the_finish_reason() -> None:
    client = _Client([_response("半句话", TRUNCATED_FINISH_REASON)])
    reply, usage, finish = OpenAILLM(CFG, client=client).chat([{"role": "user", "content": "问"}])
    assert reply == {"role": "assistant", "content": "半句话"}
    assert usage == {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13}
    assert finish == TRUNCATED_FINISH_REASON


def test_chat_hands_back_a_normal_stop() -> None:
    client = _Client([_response("完整一句话。", "stop")])
    _reply, _usage, finish = OpenAILLM(CFG, client=client).chat([{"role": "user", "content": "问"}])
    assert finish == "stop"


def test_stream_reports_the_finish_reason_and_survives_the_usage_only_tail() -> None:
    client = _Client([[_chunk("甲", None), _chunk(None, TRUNCATED_FINISH_REASON), _usage_chunk()]])
    events = list(OpenAILLM(CFG, client=client).stream([{"role": "user", "content": "问"}]))
    assert events == [
        ("delta", "甲"),
        ("finish_reason", TRUNCATED_FINISH_REASON),
        ("usage", {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
    ]
    assert client.completions.calls[0]["stream_options"] == {"include_usage": True}


def test_stream_stays_silent_about_a_missing_finish_reason() -> None:
    client = _Client([[_chunk("甲", None), _chunk("乙", None)]])
    events = list(OpenAILLM(CFG, client=client).stream([{"role": "user", "content": "问"}]))
    assert events == [("delta", "甲"), ("delta", "乙")]
