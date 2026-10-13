from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

from rag_contracts.domain.laws import LawInfo
from rag_contracts.domain.retrieval import REGION_UNKNOWN
from rag_contracts.ports import LLM

from ..prompts import INTAKE_OUTPUT_GUARD, INTAKE_SYSTEM_PROMPT
from .state import AgentState

__all__ = [
    "REGION_UNKNOWN",
    "HISTORY_ANSWER_CHARS",
    "INTAKE_INDEX_SNIPPET",
    "INTAKE_INDEX_LINES",
    "SUMMARY_MAX_CHARS",
    "TOPIC_SUMMARIES_MAX",
    "SLOT_RESTATEMENT_RATIO",
    "SLOT_RESTATEMENT_MIN_CHARS",
    "law_scope",
    "national_law_ids",
    "make_region_node",
]

HISTORY_ANSWER_CHARS = 300
INTAKE_INDEX_SNIPPET = 80
INTAKE_INDEX_LINES = 10
SUMMARY_MAX_CHARS = 60
"""话题摘要的字数上限：入口产出时截断一次，索引行渲染时再兜底截一次。"""

TOPIC_SUMMARIES_MAX = INTAKE_INDEX_LINES + 1
"""摘要的保留面＝可渲染面，+1 是推导不是缓冲：消费者只有两处 —— 其他话题的索引行
（最多 INTAKE_INDEX_LINES 个非活跃话题）＋ 活跃话题的「既有摘要」行 1 个槽位。
回访前提是话题在可见窗内（_fold_topic 只认 valid），该槽位恒可渲染、必须保留；
取 INTAKE_INDEX_LINES 会恰丢它（回访中再离开时丢了增量基底），取 2× 则留一半永远
渲染不到的死重。淘汰按各话题最近轮号、tie-break 取话题号小者。"""

SLOT_RESTATEMENT_RATIO = 0.75
"""槽位复述闸的相似度门槛（归一化去空白标点后的 SequenceMatcher ratio）：只在模型已报
new 时做一次保守的二次确认 —— 同事项、仅槽位（地区 / 车型）换写的重述按环节一规则 3
应判 continue。合成观测角（深圳上牌 → 广州上牌）≈0.87、真换事项对照 ≈0.4、换车型最险
≈0.71 不触发；宁可漏降级（维持模型原判），不可误降级。"""
SLOT_RESTATEMENT_MIN_CHARS = 8
"""头文本归一化后的最短长度：短头（如「那记分呢」）的相似度不稳，不参与闸判。"""

INTAKE_INDEX_NOTE = "（仅列最近 {lines} 个话题；更早话题未列出、无法回访，指代不到时按 new 处理）"

_NOT_A_PLACE = frozenset({"?", "？", "全国", "空串"})
_SWITCH_VALUES = frozenset({"continue", "resume", "new", "unsure"})
_INTENT_VALUES = frozenset({"followup", "challenge", "compare", "quote", "other"})
_TURN_RE = re.compile(r"^t(\d+)$")
_NOISE_RE = re.compile(r"[\W_]+")


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
        "summary": str(data.get("summary") or "").strip(),
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


def _topic_head(entries: list[dict], topic: str) -> str:
    for entry in entries:
        if _entry_topic(entry) == topic:
            return str(entry.get("question") or "")
    return ""


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
    entries: list[dict], active: str, turns: int, summaries: dict[str, str]
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
        note = str(summaries.get(topic) or "").strip()
        if note:
            body = note[:SUMMARY_MAX_CHARS]
        else:
            body = str(group[0].get("question") or "")[:INTAKE_INDEX_SNIPPET]
        rows.append((last, topic, f"[{topic}] {body}（共 {len(group)} 轮，最近第 {last} 轮）"))
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
    system: str,
    history: list[tuple[str, str]],
    index_lines: list[str],
    active: str,
    question: str,
    active_summary: str = "",
) -> list[dict]:
    prompt: list[dict] = [{"role": "system", "content": system}]
    prompt.extend({"role": role, "content": content} for role, content in history)
    blocks: list[str] = []
    if index_lines:
        blocks.append("较早的其他话题（只有索引，没有原文）：\n" + "\n".join(index_lines))
    blocks.append(f"当前活跃话题：{active}。" if active else "当前没有活跃话题。")
    if active and active_summary:
        blocks.append(f"当前活跃话题既有摘要：{active_summary}（离开本话题时以它为底更新）")
    blocks.append(f"本轮用户问题：{question}")
    blocks.append(INTAKE_OUTPUT_GUARD)
    prompt.append({"role": "user", "content": "\n\n".join(blocks)})
    return prompt


def _slot_only_restatement(head: str, query: str) -> bool:
    core = _NOISE_RE.sub("", head)
    ask = _NOISE_RE.sub("", query)
    if len(core) < SLOT_RESTATEMENT_MIN_CHARS:
        return False
    return SequenceMatcher(None, core, ask).ratio() >= SLOT_RESTATEMENT_RATIO


def _fold_topic(parsed: dict, active: str, turn_no: int, valid: set[str]) -> str:
    switch = parsed.get("switch") or "unsure"
    if switch == "resume" and parsed.get("topic") in valid:
        return str(parsed["topic"])
    if switch == "new":
        return f"t{turn_no}"
    return active or f"t{turn_no}"


def _recent_topics(entries: Iterable[dict], limit: int) -> set[str]:
    last: dict[str, int] = {}
    for entry in entries:
        topic = _entry_topic(entry)
        if topic:
            last[topic] = max(last.get(topic, 0), _entry_turn(entry))
    ordered = sorted(last.items(), key=lambda item: (-item[1], item[0]))
    return {topic for topic, _turn in ordered[:limit]}


def _summaries_after_write(
    entries: list[dict], summaries: dict[str, str], active: str, topic_id: str, parsed: dict
) -> dict[str, str] | None:
    if not parsed or not active or topic_id == active:
        return None
    text = str(parsed.get("summary") or "").strip()
    if not text or not any(_entry_topic(entry) == active for entry in entries):
        return None
    updated = dict(summaries)
    updated[active] = text[:SUMMARY_MAX_CHARS]
    keep = _recent_topics(entries, TOPIC_SUMMARIES_MAX)
    return {topic: value for topic, value in updated.items() if topic in keep}


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
):
    from langgraph.types import Overwrite

    prompt_head = INTAKE_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {law.law_name}" for law in laws),
    }

    def region_node(state: AgentState) -> dict:
        turn_no = _next_turn_no(state)
        active = str(state.get("topic_id") or "")
        conversation = state.get("conversation") or ()
        convo = [entry for entry in conversation if isinstance(entry, dict)]
        entries = convo or _pairs_to_entries(state.get("history") or ())
        raw_summaries = state.get("topic_summaries")
        summaries = dict(raw_summaries) if isinstance(raw_summaries, dict) else {}

        parsed: dict = {}
        usage: dict = {}
        exposed: set[str] = set()
        region, place = REGION_UNKNOWN, ""
        if llm.available:
            pre_view, pre_index, exposed = _topic_view(entries, active, history_turns, summaries)
            try:
                reply, usage, _finish = llm.chat(
                    _intake_prompt(
                        prompt_head,
                        pre_view,
                        pre_index,
                        active,
                        state["question"],
                        summaries.get(active, ""),
                    ),
                    temperature=0.0,
                    name="llm.intake",
                )
                content = reply.get("content") or ""
                parsed = _parse_intake(content)
                region, place = _parse_region(content)
            except Exception:  # noqa: BLE001
                parsed = {}
                region, place = REGION_UNKNOWN, ""
                usage = {}

        switch_raw = str(parsed.get("switch") or "")
        if parsed:
            if switch_raw == "new":
                head = str(summaries.get(active) or "").strip() or _topic_head(entries, active)
                if (
                    active
                    and head
                    and parsed.get("query")
                    and _slot_only_restatement(head, str(parsed["query"]))
                ):
                    parsed["switch"] = "continue"
            topic_id = _fold_topic(parsed, active, turn_no, exposed)
            history = _topic_view(entries, topic_id, history_turns, summaries)[0]
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
            "switch_raw": switch_raw,
            "intent": str(parsed.get("intent") or ""),
            "topic_id": topic_id,
            "turn_no": turn_no,
        }
        written = _summaries_after_write(entries, summaries, active, topic_id, parsed)
        if written is not None:
            reset["topic_summaries"] = written
        if region in {law.law_name for law in laws}:
            region = REGION_UNKNOWN
        return {
            **reset,
            "region": region,
            "region_scope": law_scope(region, laws),
            "place": place,
            "usage_extra": Overwrite([usage] if usage else []),
        }

    return region_node
