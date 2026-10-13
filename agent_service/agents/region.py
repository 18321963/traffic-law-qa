from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence

from rag_contracts.domain.laws import LawInfo
from rag_contracts.domain.retrieval import REGION_UNKNOWN
from rag_contracts.ports import LLM

from ..prompts import (
    INTAKE_SYSTEM_PROMPT,
    REGION_SIGNAL_INHERIT,
    REGION_SIGNAL_INTENT,
    REGION_SIGNAL_NEW_TOPIC,
    REGION_SYSTEM_PROMPT,
)
from .state import AgentState

__all__ = [
    "REGION_UNKNOWN",
    "HISTORY_ANSWER_CHARS",
    "INTAKE_INDEX_SNIPPET",
    "INTAKE_INDEX_LINES",
    "law_scope",
    "national_law_ids",
    "make_region_node",
]

HISTORY_ANSWER_CHARS = 300
INTAKE_INDEX_SNIPPET = 80
INTAKE_INDEX_LINES = 10

INTAKE_INDEX_NOTE = "（仅列最近 {lines} 个话题；更早话题未列出、无法回访，指代不到时按 new 处理）"

_NOT_A_PLACE = frozenset({"?", "？", "全国", "空串"})
_SWITCH_VALUES = frozenset({"continue", "resume", "new", "unsure"})
_INTENT_VALUES = frozenset({"followup", "challenge", "compare", "quote", "other"})
_INHERIT_INTENTS = frozenset({"followup", "challenge", "compare", "quote"})
_INTENT_LABELS = {
    "followup": "追问",
    "challenge": "质疑",
    "compare": "对比",
    "quote": "要求引用",
    "other": "其他",
}
_TURN_RE = re.compile(r"^t(\d+)$")


def national_law_ids(laws: Sequence[LawInfo]) -> tuple[str, ...]:
    return tuple(sorted({law.law_id for law in laws if not law.local}))


def law_scope(region: str, laws: Sequence[LawInfo]) -> tuple[str, ...]:
    if not region or region == REGION_UNKNOWN:
        return ()
    if region in {law.law_name for law in laws}:
        return ()
    local = tuple(sorted({law.law_id for law in laws if region in law.law_name}))
    if not local:
        return ()
    return tuple(sorted(set(national_law_ids(laws)) | set(local)))


def _parse_region(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            region = str(data.get("region") or "").strip() or REGION_UNKNOWN
            place = str(data.get("place") or "").strip()
            if region in _NOT_A_PLACE:
                region = REGION_UNKNOWN
            if place in _NOT_A_PLACE:
                place = ""
            return region, place
    return REGION_UNKNOWN, ""


def _clip_answer(text: str) -> str:
    if len(text) <= HISTORY_ANSWER_CHARS:
        return text
    return text[:HISTORY_ANSWER_CHARS] + "…"


def _pairs_to_entries(pairs: Iterable[Sequence[str]]) -> list[dict]:
    entries: list[dict] = []
    pending: str | None = None
    for role, content in pairs:
        if role == "user":
            pending = content
        elif role == "assistant" and pending is not None:
            entries.append({"question": pending, "answer": content})
            pending = None
    return entries


def _region_tag(entry: dict) -> str:
    region = str(entry.get("region") or "").strip()
    if not region or region == REGION_UNKNOWN:
        return ""
    if region == "national":
        return "（本轮按全国法作答）"
    return f"（本轮按「{region}」的规定作答）"


def _parse_intake(text: str) -> dict:
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    switch = str(data.get("switch") or "").strip().lower()
    intent = str(data.get("intent") or "").strip().lower()
    return {
        "switch": switch if switch in _SWITCH_VALUES else "unsure",
        "intent": intent if intent in _INTENT_VALUES else "other",
        "query": str(data.get("query") or "").strip(),
        "topic": str(data.get("topic_id") or "").strip(),
    }


def _entries_of(conversation: Iterable[dict], fallback: Iterable[Sequence[str]]) -> list[dict]:
    entries = [entry for entry in conversation if isinstance(entry, dict)]
    if not entries:
        entries = _pairs_to_entries(fallback)
    return entries


def _entry_topic(entry: dict) -> str:
    return str(entry.get("topic_id") or "")


def _entry_turn(entry: dict) -> int:
    try:
        return int(entry.get("turn") or 0)
    except (TypeError, ValueError):
        return 0


def _render_window(entries: list[dict], turns: int) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    for entry in entries[-turns:] if turns > 0 else []:
        history.append(("user", str(entry.get("question") or "")))
        history.append(
            ("assistant", _region_tag(entry) + _clip_answer(str(entry.get("answer") or "")))
        )
    return history


def _build_history(
    conversation: Iterable[dict],
    fallback: Iterable[Sequence[str]],
    turns: int,
) -> list[tuple[str, str]]:
    return _render_window(_entries_of(conversation, fallback), turns)


def _topic_view(
    entries: list[dict], active: str, turns: int
) -> tuple[list[tuple[str, str]], list[str], set[str]]:
    if not active:
        return _render_window(entries, turns), [], set()
    pool = [entry for entry in entries if _entry_topic(entry) in ("", active)]
    window = pool[-turns:] if turns > 0 else []
    in_window = {id(entry) for entry in window}
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        topic = _entry_topic(entry)
        if topic and topic != active:
            groups.setdefault(topic, []).append(entry)
    rows: list[tuple[int, str, str]] = []
    for topic, group in groups.items():
        last = max(_entry_turn(entry) for entry in group)
        head = str(group[0].get("question") or "")[:INTAKE_INDEX_SNIPPET]
        rows.append((last, topic, f"[{topic}] {head}（共 {len(group)} 轮，最近第 {last} 轮）"))
    rows.sort(key=lambda row: row[0], reverse=True)
    shown = rows[:INTAKE_INDEX_LINES]
    older = sum(
        1 for entry in entries if not _entry_topic(entry) and id(entry) not in in_window
    )
    lines: list[str] = []
    if shown or older:
        lines.append(INTAKE_INDEX_NOTE.format(lines=INTAKE_INDEX_LINES))
    lines.extend(text for _last, _topic, text in shown)
    if older:
        lines.append(f"[?] 更早 {older} 轮")
    valid = {active} | {topic for _last, topic, _text in shown}
    return _render_window(pool, turns), lines, valid


def _intake_prompt(
    history: list[tuple[str, str]], index_lines: list[str], active: str, question: str
) -> list[dict]:
    prompt: list[dict] = [{"role": "system", "content": INTAKE_SYSTEM_PROMPT}]
    prompt.extend({"role": role, "content": content} for role, content in history)
    blocks: list[str] = []
    if index_lines:
        blocks.append("较早的其他话题（只有索引，没有原文）：\n" + "\n".join(index_lines))
    blocks.append(f"当前活跃话题：{active}。" if active else "当前没有活跃话题。")
    blocks.append(f"本轮用户问题：{question}")
    prompt.append({"role": "user", "content": "\n\n".join(blocks)})
    return prompt


def _fold_topic(parsed: dict, active: str, turn_no: int, valid: set[str]) -> str:
    switch = parsed.get("switch") or "unsure"
    if switch == "resume" and parsed.get("topic") in valid:
        return str(parsed["topic"])
    if switch == "new":
        return f"t{turn_no}"
    return active or f"t{turn_no}"


def _region_signals(parsed: dict) -> list[str]:
    if not parsed:
        return []
    intent = str(parsed.get("intent") or "other")
    signals = [REGION_SIGNAL_INTENT.format(intent=_INTENT_LABELS.get(intent, "其他"))]
    if parsed.get("switch") == "new":
        signals.append(REGION_SIGNAL_NEW_TOPIC)
    elif intent in _INHERIT_INTENTS:
        signals.append(REGION_SIGNAL_INHERIT)
    return signals


def _next_turn_no(state: AgentState) -> int:
    try:
        current = int(state.get("turn_no") or 0)
    except (TypeError, ValueError):
        current = 0
    seen = 0
    for entry in state.get("conversation") or ():
        if not isinstance(entry, dict):
            continue
        match = _TURN_RE.match(_entry_topic(entry))
        if match:
            seen = max(seen, int(match.group(1)))
        seen = max(seen, _entry_turn(entry))
    return max(current + 1, seen + 1)


def make_region_node(
    llm: LLM,
    laws: Sequence[LawInfo],
    *,
    history_turns: int = 3,
    intake_llm: LLM | None = None,
):
    from langgraph.types import Overwrite

    prompt_head = REGION_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {law.law_name}" for law in laws),
    }

    def region_node(state: AgentState) -> dict:
        turn_no = _next_turn_no(state)
        active = str(state.get("topic_id") or "")
        conversation = state.get("conversation") or ()
        convo = [entry for entry in conversation if isinstance(entry, dict)]
        entries = convo or _pairs_to_entries(state.get("history") or ())

        parsed: dict = {}
        intake_usage: dict = {}
        exposed: set[str] = set()
        if intake_llm is not None and intake_llm.available and convo:
            pre_view, pre_index, exposed = _topic_view(entries, active, history_turns)
            try:
                reply, intake_usage, _finish = intake_llm.chat(
                    _intake_prompt(pre_view, pre_index, active, state["question"]),
                    temperature=0.0,
                    name="llm.intake",
                )
                parsed = _parse_intake(reply.get("content") or "")
            except Exception:  # noqa: BLE001
                parsed = {}
                intake_usage = {}

        if parsed:
            topic_id = _fold_topic(parsed, active, turn_no, exposed)
            history = _topic_view(entries, topic_id, history_turns)[0]
        else:
            topic_id = active or f"t{turn_no}"
            history = _render_window(entries, history_turns)

        reset: dict = {
            "messages": Overwrite([]),
            "search_log": Overwrite([]),
            "external": Overwrite([]),
            "materials": Overwrite([]),
            "steps": Overwrite(0),
            "usage": Overwrite([]),
            "usage_extra": Overwrite([]),
            "truncated": Overwrite(0),
            "call_keys": Overwrite([]),
            "answer": None,
            "region_note": "",
            "history": history,
            "query_rewrite": str(parsed.get("query") or ""),
            "topic_switch": str(parsed.get("switch") or ""),
            "intent": str(parsed.get("intent") or ""),
            "topic_id": topic_id,
            "turn_no": turn_no,
        }
        carried = Overwrite([intake_usage] if intake_usage else [])
        if not llm.available:
            return {
                **reset,
                "region": REGION_UNKNOWN,
                "region_scope": (),
                "place": "",
                "usage_extra": carried,
            }

        prompt: list[dict] = [{"role": "system", "content": prompt_head}]
        prompt.extend({"role": role, "content": content} for role, content in history)
        user_text = reset["query_rewrite"] or state["question"]
        signals = _region_signals(parsed)
        if signals:
            user_text = "\n\n".join([user_text, *signals])
        prompt.append({"role": "user", "content": user_text})

        reply, usage, _finish = llm.chat(prompt, temperature=0.0, name="llm.region")
        region, place = _parse_region(reply.get("content") or "")
        if region in {law.law_name for law in laws}:
            region = REGION_UNKNOWN
        return {
            **reset,
            "region": region,
            "region_scope": law_scope(region, laws),
            "place": place,
            "usage_extra": Overwrite([u for u in (intake_usage, usage) if u]),
        }

    return region_node
