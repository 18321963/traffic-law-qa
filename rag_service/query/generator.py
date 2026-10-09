from __future__ import annotations

import time
from typing import Any

from rag_contracts import config
from rag_contracts.domain.answer import Answer, Evidence, Question
from rag_contracts.domain.retrieval import MaterialPassage, RetrievalResult, WebFinding
from rag_contracts.ports import LLM, TRUNCATED_FINISH_REASON

from ..prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE, MATERIAL_HEADER, TIMELINESS_HEADER

EMPTY_RETRIEVAL_ANSWER = (
    "现有法规库中未检索到与问题相关的条文，无法给出有依据的回答。"
    "建议补充更具体的违法情形、地点，或确认是否属于本知识库覆盖的法规范围。"
)
UNAVAILABLE_ANSWER = "（未配置大模型，下面只给出召回的法条）"


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
            from rag_contracts.llm import build_llm

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

    def _assemble(
        self,
        question: Question,
        retrieval: RetrievalResult,
        *,
        text: str,
        model: str,
        evidences: list[Evidence],
        notes: list[str],
        started: float,
        usage: dict[str, Any] | None = None,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> Answer:
        return Answer(
            question=question.text,
            text=text,
            evidences=tuple(evidences),
            model=model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            usage=usage or {},
            retrieval=retrieval,
            notes=tuple(notes),
            timeliness=timeliness,
            materials=materials,
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

        if retrieval.is_empty and not timeliness and not materials:
            return self._assemble(
                question,
                retrieval,
                text=EMPTY_RETRIEVAL_ANSWER,
                model="(skip)",
                evidences=[],
                notes=notes + ["检索结果为空，未调用 LLM"],
                started=started,
                timeliness=timeliness,
                materials=materials,
            )

        if not self.available:
            return self._assemble(
                question,
                retrieval,
                text=UNAVAILABLE_ANSWER,
                model="(unavailable)",
                evidences=evidences,
                notes=notes + [self.unavailable_reason],
                started=started,
                timeliness=timeliness,
                materials=materials,
            )

        content, usage, finish_reason = self._call_llm(
            question, evidences, timeliness=timeliness, materials=materials
        )
        if finish_reason == TRUNCATED_FINISH_REASON:
            notes.append("模型输出被 max_tokens 截断，答案可能不完整")
        return self._assemble(
            question,
            retrieval,
            text=content,
            model=self.cfg.model,
            evidences=evidences,
            notes=notes,
            started=started,
            usage=usage,
            timeliness=timeliness,
            materials=materials,
        )

    def stream(
        self,
        question: Question,
        retrieval: RetrievalResult,
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ):
        started = time.perf_counter()
        evidences = self.build_evidence(retrieval)
        notes = list(retrieval.notes)

        if retrieval.is_empty and not timeliness and not materials:
            yield "delta", EMPTY_RETRIEVAL_ANSWER
            yield "usage", {}
            yield "answer", self._assemble(
                question,
                retrieval,
                text=EMPTY_RETRIEVAL_ANSWER,
                model="(skip)",
                evidences=[],
                notes=notes + ["检索结果为空，未调用 LLM"],
                started=started,
                timeliness=timeliness,
                materials=materials,
            )
            return

        if not self.available:
            yield "delta", UNAVAILABLE_ANSWER
            yield "usage", {}
            yield "answer", self._assemble(
                question,
                retrieval,
                text=UNAVAILABLE_ANSWER,
                model="(unavailable)",
                evidences=evidences,
                notes=notes + [self.unavailable_reason],
                started=started,
                timeliness=timeliness,
                materials=materials,
            )
            return

        parts: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        for kind, payload in self.llm.stream(
            self._messages(question, evidences, timeliness=timeliness, materials=materials),
            temperature=self.cfg.temperature,
        ):
            if kind == "delta":
                parts.append(payload)
            elif kind == "usage":
                usage = payload
            elif kind == "finish_reason":
                finish_reason = payload
            yield kind, payload
        if finish_reason == TRUNCATED_FINISH_REASON:
            notes.append("模型输出被 max_tokens 截断，答案可能不完整")
        yield "answer", self._assemble(
            question,
            retrieval,
            text="".join(parts).strip(),
            model=self.cfg.model,
            evidences=evidences,
            notes=notes,
            started=started,
            usage=usage,
            timeliness=timeliness,
            materials=materials,
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
    ) -> tuple[str, dict, str | None]:
        reply, usage, finish_reason = self.llm.chat(
            self._messages(question, evidences, timeliness=timeliness, materials=materials),
            temperature=self.cfg.temperature,
            name="answer.generate",
        )
        return (reply.get("content") or "").strip(), usage, finish_reason
