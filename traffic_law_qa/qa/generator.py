from __future__ import annotations

import time
from typing import Any

from .. import config
from ..contracts import (
    Answer,
    Evidence,
    MaterialPassage,
    Question,
    RetrievalResult,
    WebFinding,
)

SYSTEM_PROMPT = """你是面向驾驶员、驾校学员与交管客服场景的交通法规问答助手。

工作规则：
1. 只依据「依据」中给出的法条回答，禁止使用未提供的知识，禁止推测与补全。
2. 每条结论后必须用 [依据N] 标注来源，N 对应依据编号；没有依据支撑的话不要写。
3. 依据不足时，直接说明"现有法规库中未收录相关依据"，并指出还缺哪类规定。
   **不要写出任何库外法规、规章、司法解释的具体名称**（包括上位法、其他省市条例），
   只能描述“需要哪类规定” —— 列出听起来合理的库外法名就是幻觉。
4. 涉及深圳经济特区法规时，必须说明其适用范围仅限深圳经济特区。
5. 引用法条要写全「法规名称 + 条号」，例如《中华人民共和国道路交通安全法》第九十一条。
6. 先给结论，再给依据与说明；输出简洁的 Markdown，不要复述整条法条原文。
7. 依据里没提到的事（例如具体金额、记分、后续流程）一律不补充。
8. 「时效提示」与「本次会话材料」两块**不属于依据**：引用它们时只能标 [时效N] / [材料N]，
   **绝不允许标成 [依据N]** —— 依据编号对不上会让整篇答案被判为无依据而作废。"""

USER_TEMPLATE = """问题：{question}

依据（共 {count} 条）：
{evidences}
{external}
请依据上述条文回答问题，并在每条结论后标注 [依据N]。"""

TIMELINESS_HEADER = "时效提示（联网检索，非本库法条 —— 引用时标 [时效N]，不要标 [依据N]）："
MATERIAL_HEADER = "本次会话材料（未入知识库，仅供参照 —— 引用时标 [材料N]，不要标 [依据N]）："
"""库里没收录的东西（网搜、本次上传）放这两块，与「依据」块并列但**不算依据**。

`{external}` 在 `USER_TEMPLATE` 里独占一行，由 `_external_block` 拼装：为空串时那一行塌成
原有的空行，提示词与加这两块之前**逐字节相同**；非空时前后各补一个换行，与依据块分开。

标题自身就写死「不要标 [依据N]」这条 —— 复核闸 `review.CITE_RE` 只认 `[依据N]`，模型一旦
把网搜内容标成依据编号，编号对不上就会被 `over` 判不支撑、整篇降级。"""

_STREAM_OPTIONS = {"include_usage": True}
"""流式请求要 usage 块。理由见 `_stream_llm` —— 不加就是静默少一个字段。"""

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
    ) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self.show_top = show_top
        self._client = client
        self.timeout = timeout
        self.max_tokens = max_tokens

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
        return USER_TEMPLATE.format(
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

        for chunk in self._stream_llm(question, evidences):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield "delta", delta
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
                yield "usage", {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }

    def _stream_llm(self, question: Question, evidences: list[Evidence]):
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

        messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": role, "content": content} for role, content in question.history)
        messages.append({"role": "user", "content": self.build_prompt(question, evidences)})

        try:
            return self._client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options=_STREAM_OPTIONS,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"调用 {self.cfg.model} 失败：{exc}") from exc

    def _call_llm(
        self,
        question: Question,
        evidences: list[Evidence],
        *,
        timeliness: tuple[WebFinding, ...] = (),
        materials: tuple[MaterialPassage, ...] = (),
    ) -> tuple[str, dict]:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

        messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": role, "content": content} for role, content in question.history)
        messages.append(
            {
                "role": "user",
                "content": self.build_prompt(question, evidences, timeliness, materials),
            }
        )

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    timeout=self.timeout,
                    max_tokens=self.max_tokens,
                )
                usage = {}
                if response.usage is not None:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                return (response.choices[0].message.content or "").strip(), usage
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error
