from __future__ import annotations

import time
from typing import Any

from .. import config
from ..contracts.answer import Answer, Evidence, Question
from ..contracts.retrieval import MaterialPassage, RetrievalResult, WebFinding
from ..ports import LLM
from ..prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE, MATERIAL_HEADER, TIMELINESS_HEADER

EMPTY_RETRIEVAL_ANSWER = (
    "现有法规库中未检索到与问题相关的条文，无法给出有依据的回答。"
    "建议补充更具体的违法情形、地点，或确认是否属于本知识库覆盖的 6 部法规范围。"
)
UNAVAILABLE_ANSWER = "（未配置大模型，下面只给出召回的法条）"
REVIEW_DOWNGRADE_ANSWER = (
    "本次回答未通过依据复核，暂不给出结论 —— 现有依据不足以支撑它，需人工复审。"
)
"""复核不通过时的替换文案，与上面两句同属「不直接给结论」的文案族。

**放在这里而不是 `agents/review.py`**：这一族一共四句（检索为空 / 未配置模型 / 模型说依据不足 /
复核不通过），改口径时要能一次看见全部。生产者（agent 的复核节点）在别处，但它写出来的
是给人看的话，话归这里管。数字与候选法条清单由 `review.downgrade_text` 拼在后面。"""


def _external_block(
    timeliness: tuple[WebFinding, ...], materials: tuple[MaterialPassage, ...]
) -> str:
    blocks: list[str] = []
    if timeliness:
        blocks.append(TIMELINESS_HEADER + "\n" + "\n\n".join(w.render() for w in timeliness))
    if materials:
        blocks.append(MATERIAL_HEADER + "\n" + "\n\n".join(m.render() for m in materials))
    if not blocks:
        return ""
    return "\n" + "\n\n".join(blocks) + "\n"


class AnswerGenerator:

    layer = "generate"
    input_desc = "Question + RetrievalResult"
    output_desc = "Answer"

    def __init__(
        self,
        cfg: config.LLMConfig | None = None,
        *,
        retries: int = 2,
        show_top: int = 0,
        client: Any = None,
        timeout: float = 30.0,
        max_tokens: int = 1024,
        llm: LLM | None = None,
    ) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self.show_top = show_top
        self._client = client
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            from ..container import build_llm

            self._llm = build_llm(
                self.cfg,
                retries=self.retries,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
                client=self._client,
            )
        return self._llm

    @property
    def model_name(self) -> str:
        return self.cfg.model

    @property
    def available(self) -> bool:
        return self.cfg.ready

    @property
    def unavailable_reason(self) -> str:
        return "" if self.available else "未配置 LLM_API_KEY，无法生成答案（可用 search 查看检索结果）"

    def build_evidence(self, retrieval: RetrievalResult, top_n: int = 0) -> list[Evidence]:
        limit = top_n or self.show_top or len(retrieval.articles)
        evidences: list[Evidence] = []
        for index, hit in enumerate(retrieval.articles[:limit], start=1):
            evidences.append(
                Evidence(
                    label=f"【依据{index}】",
                    citation=hit.citation,
                    text=hit.article.text,
                    score=hit.score,
                    article=hit.article,
                )
            )
        return evidences

    def build_prompt(
        self,
        question: Question,
        evidences: list[Evidence],
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> str:
        blocks = "\n\n".join(e.render() for e in evidences)
        return ANSWER_USER_TEMPLATE.format(
            question=question.text,
            count=len(evidences),
            evidences=blocks,
            external=_external_block(timeliness, materials),
        )

    def generate(
        self,
        question: Question,
        retrieval: RetrievalResult,
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> Answer:
        evidences = self.build_evidence(retrieval)
        started = time.perf_counter()
        notes = list(retrieval.notes)

        if retrieval.is_empty:
            return Answer(
                question=question.text,
                text=EMPTY_RETRIEVAL_ANSWER,
                evidences=(),
                model="(skip)",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                retrieval=retrieval,
                notes=tuple(notes + ["检索结果为空，未调用 LLM"]),
                timeliness=timeliness,
                materials=materials,
            )

        if not self.available:
            return Answer(
                question=question.text,
                text=UNAVAILABLE_ANSWER,
                evidences=tuple(evidences),
                model="(unavailable)",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                retrieval=retrieval,
                notes=tuple(notes + [self.unavailable_reason]),
                timeliness=timeliness,
                materials=materials,
            )

        content, usage = self._call_llm(
            question, evidences, timeliness=timeliness, materials=materials
        )
        return Answer(
            question=question.text,
            text=content,
            evidences=tuple(evidences),
            model=self.cfg.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
            retrieval=retrieval,
            notes=tuple(notes),
            timeliness=timeliness,
            materials=materials,
        )

    def stream(self, question: Question, retrieval: RetrievalResult):
        evidences = self.build_evidence(retrieval)

        if retrieval.is_empty or not self.available:
            text = EMPTY_RETRIEVAL_ANSWER if retrieval.is_empty else UNAVAILABLE_ANSWER
            yield "delta", text
            yield "usage", {}
            return

        yield from self.llm.stream(
            self._messages(question, evidences), temperature=self.cfg.temperature
        )

    def _messages(
        self,
        question: Question,
        evidences: list[Evidence],
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> list[Any]:
        messages: list[Any] = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
        messages.extend({"role": role, "content": content} for role, content in question.history)
        messages.append(
            {
                "role": "user",
                "content": self.build_prompt(question, evidences, timeliness, materials),
            }
        )
        return messages

    def _call_llm(
        self,
        question: Question,
        evidences: list[Evidence],
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> tuple[str, dict]:
        reply, usage = self.llm.chat(
            self._messages(question, evidences, timeliness=timeliness, materials=materials),
            temperature=self.cfg.temperature,
            name="answer.generate",
        )
        return (reply.get("content") or "").strip(), usage
