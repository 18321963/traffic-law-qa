"""Layer 6 · generate：Question + RetrievalResult → Answer（强制引用溯源）。

    AnswerGenerator.generate : Question + RetrievalResult → Answer

纯 RAG 的成败在这里：模型只能看检索到的法条，且必须逐条标注 [依据N]。
检索为空时直接拒答（不调用 LLM），避免幻觉。
"""

from __future__ import annotations

import time
from typing import Any

from . import config
from .contracts import Answer, Evidence, Question, RetrievalResult

SYSTEM_PROMPT = """你是面向驾驶员、驾校学员与交管客服场景的交通法规问答助手。

工作规则：
1. 只依据「依据」中给出的法条回答，禁止使用未提供的知识，禁止推测与补全。
2. 每条结论后必须用 [依据N] 标注来源，N 对应依据编号；没有依据支撑的话不要写。
3. 依据不足时，直接说明"现有法规库中未收录相关依据"，并指出还缺哪部法规或哪类条文。
4. 涉及深圳经济特区法规时，必须说明其适用范围仅限深圳经济特区。
5. 引用法条要写全「法规名称 + 条号」，例如《中华人民共和国道路交通安全法》第九十一条。
6. 先给结论，再给依据与说明；输出简洁的 Markdown，不要复述整条法条原文。"""

USER_TEMPLATE = """问题：{question}

依据（共 {count} 条）：
{evidences}

请依据上述条文回答问题，并在每条结论后标注 [依据N]。"""


class AnswerGenerator:
    """答案生成器：把检索结果变成带引用的答案。

    Input : Question + RetrievalResult
    Output: Answer
    """

    layer = "generate"
    input_desc = "Question + RetrievalResult"
    output_desc = "Answer"

    def __init__(self, cfg: config.LLMConfig | None = None, *, retries: int = 2, show_top: int = 0) -> None:
        self.cfg = cfg or config.llm_config()
        self.retries = retries
        self.show_top = show_top  # 0 表示全部依据都给 LLM
        self._client = None

    @property
    def available(self) -> bool:
        return self.cfg.ready

    @property
    def unavailable_reason(self) -> str:
        return "" if self.available else "未配置 LLM_API_KEY，无法生成答案（可用 search 查看检索结果）"

    # -------------------------------------------------------------- 证据组织
    def build_evidence(self, retrieval: RetrievalResult, top_n: int = 0) -> list[Evidence]:
        """把召回的法条编成【依据N】。"""
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

    def build_prompt(self, question: Question, evidences: list[Evidence]) -> str:
        blocks = "\n\n".join(e.render() for e in evidences)
        return USER_TEMPLATE.format(question=question.text, count=len(evidences), evidences=blocks)

    # -------------------------------------------------------------- 主接口
    def generate(self, question: Question, retrieval: RetrievalResult) -> Answer:
        evidences = self.build_evidence(retrieval)
        started = time.perf_counter()
        notes = list(retrieval.notes)

        if retrieval.is_empty:
            return Answer(
                question=question.text,
                text="现有法规库中未检索到与问题相关的条文，无法给出有依据的回答。"
                "建议补充更具体的违法情形、地点，或确认是否属于本知识库覆盖的 4 部法规范围。",
                evidences=(),
                model="(skip)",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                retrieval=retrieval,
                notes=tuple(notes + ["检索结果为空，未调用 LLM"]),
            )

        if not self.available:
            return Answer(
                question=question.text,
                text="（未配置大模型，下面只给出召回的法条）",
                evidences=tuple(evidences),
                model="(unavailable)",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                retrieval=retrieval,
                notes=tuple(notes + [self.unavailable_reason]),
            )

        content, usage = self._call_llm(question, evidences)
        return Answer(
            question=question.text,
            text=content,
            evidences=tuple(evidences),
            model=self.cfg.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
            retrieval=retrieval,
            notes=tuple(notes),
        )

    # -------------------------------------------------------------- 调用
    def _call_llm(self, question: Question, evidences: list[Evidence]) -> tuple[str, dict]:
        from openai import OpenAI  # 延迟导入

        if self._client is None:
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

        messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": role, "content": content} for role, content in question.history)
        messages.append({"role": "user", "content": self.build_prompt(question, evidences)})

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                )
                usage = {}
                if response.usage is not None:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                return (response.choices[0].message.content or "").strip(), usage
            except Exception as exc:  # noqa: BLE001 - 统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error
