from __future__ import annotations

import json
import re
from dataclasses import replace

from .. import config
from ..contracts.answer import Evidence, Review
from ..generation.generator import REVIEW_DOWNGRADE_ANSWER
from ..ports import LLM
from ..prompts import REVIEW_SYSTEM_PROMPT
from .state import AgentState

__all__ = ["cited_labels", "downgrade_text", "parse_judgments", "make_review_node"]

CITE_RE = re.compile(r"[\[【]依据\s*(\d+)[\]】]")
"""答案里的引用标注。**半角 `[依据3]` 与全角 `【依据3】` 都认** —— 实测 120 条真答案里有 2 条
在同一条答案里两种混用；只认一种会把那几条当成「没引这一条」，分母随之偏小、分数虚高。"""


def cited_labels(text: str) -> list[int]:
    return list(dict.fromkeys(int(n) for n in CITE_RE.findall(text or "")))


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


def parse_judgments(text: str) -> dict[int, bool | None]:
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
    items = data.get("judgments")
    if not isinstance(items, list):
        return {}
    out: dict[int, bool | None] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        number = item.get("n")
        if isinstance(number, bool):
            continue
        try:
            out[int(number)] = _as_bool(item.get("supported"))
        except (TypeError, ValueError):
            continue
    return out


def _labels_text(numbers) -> str:
    return "、".join(f"依据{n}" for n in numbers)


def downgrade_text(review: Review, evidences: tuple[Evidence, ...]) -> str:
    lines = [
        REVIEW_DOWNGRADE_ANSWER,
        "",
        f"复核结果：支撑 {review.supported}/{review.total}（阈值 {review.threshold:g}），需人工复审。",
    ]
    if evidences:
        lines += ["", "候选法条（未经复核确认，仅供人工核对）："]
        lines += [f"  {e.label} {e.citation}" for e in evidences]
    return "\n".join(lines)


def _build_payload(question: str, answer_text: str, listed: list[Evidence]) -> list[dict]:
    blocks = "\n\n".join(e.render() for e in listed)
    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\n答案：\n{answer_text}\n\n"
                f"答案引用到的法条（共 {len(listed)} 条）：\n{blocks}"
            ),
        },
    ]


def make_review_node(llm: LLM, cfg: config.AgentConfig):

    def review_node(state: AgentState) -> dict:
        answer = state.get("answer")
        if answer is None or cfg.review_min_score < 0:
            return {}

        threshold = cfg.review_min_score
        labels = cited_labels(answer.text)
        if not labels:
            return {
                "answer": replace(
                    answer,
                    notes=answer.notes
                    + ("复核未打分：答案里没有 [依据N]（拒答与未配置模型那两句常量文案走这条）",),
                )
            }

        evidences = answer.evidences
        total = len(labels)
        by_label = {}
        for evidence in evidences:
            numbers = cited_labels(evidence.label)
            if numbers:
                by_label[numbers[0]] = evidence
        over = [n for n in labels if n not in by_label]
        listed = [by_label[n] for n in labels if n in by_label]
        over_set = set(over)

        judgments: dict[int, bool | None] = {}
        if listed:
            if not llm.available:
                return {
                    "answer": replace(
                        answer,
                        notes=answer.notes + ("复核未完成：未配置复核模型，本次未拦截",),
                    )
                }
            try:
                reply, _usage = llm.chat(
                    _build_payload(answer.question, answer.text, listed),
                    temperature=0.0,
                    name="llm.review",
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"{type(exc).__name__}: {exc}"[:120]
                return {
                    "answer": replace(
                        answer,
                        notes=answer.notes + (f"复核未完成：{reason}，本次未拦截",),
                    )
                }
            judgments = parse_judgments(reply.get("content") or "")
            if not judgments:
                return {
                    "answer": replace(
                        answer,
                        notes=answer.notes + ("复核未完成：判据解析失败，本次未拦截",),
                    )
                }

        supported: list[int] = []
        unsupported: list[int] = []
        missing: list[int] = []
        for n in labels:
            verdict = None if n in over_set else judgments.get(n)
            if verdict is True:
                supported.append(n)
            else:
                unsupported.append(n)
                if verdict is None and n not in over_set:
                    missing.append(n)

        score = round(len(supported) / total, 3)
        passed = not (threshold > 0 and score < threshold)
        passed_note = (
            f"复核未通过：支撑 {len(supported)}/{total} < 阈值 {threshold:g}"
            " —— 已降级为「依据不足」，需人工复审"
            if not passed
            else (
                f"复核通过：支撑 {len(supported)}/{total} ≥ 阈值 {threshold:g}"
                if threshold > 0
                else f"复核打分：支撑 {len(supported)}/{total}（阈值 0，只打分不拦截）"
            )
        )
        notes = [passed_note]
        if over:
            notes.append(
                f"复核：编号 {_labels_text(over)} 不在本次给的依据里（共 {len(evidences)} 条），按不支撑计"
            )
        if missing:
            notes.append(f"复核：编号 {_labels_text(missing)} 未给判据，按不支撑计")

        review = Review(
            score=score,
            threshold=threshold,
            total=total,
            supported=len(supported),
            unsupported=tuple(f"依据{n}" for n in unsupported),
            original_text=answer.text,
            model=llm.model_name,
            passed=passed,
        )
        return {
            "answer": replace(
                answer,
                text=answer.text if passed else downgrade_text(review, evidences),
                notes=answer.notes + tuple(notes),
                review=review,
            )
        }

    return review_node
