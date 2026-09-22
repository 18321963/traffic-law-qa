"""Layer 6 · generate：Question + RetrievalResult → Answer（强制引用溯源）。

    AnswerGenerator.generate : Question + RetrievalResult → Answer

纯 RAG 的成败在这里：模型只能看检索到的法条，且必须逐条标注 [依据N]。
检索为空时直接拒答（不调用 LLM），避免幻觉。
"""

from __future__ import annotations

import time
from typing import Any

from .. import config
from ..contracts import Answer, Evidence, Question, RetrievalResult

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
7. 依据里没提到的事（例如具体金额、记分、后续流程）一律不补充。"""

USER_TEMPLATE = """问题：{question}

依据（共 {count} 条）：
{evidences}

请依据上述条文回答问题，并在每条结论后标注 [依据N]。"""

_STREAM_OPTIONS = {"include_usage": True}
"""流式请求要 usage 块。理由见 `_stream_llm` —— 不加就是静默少一个字段。"""

EMPTY_RETRIEVAL_ANSWER = (
    "现有法规库中未检索到与问题相关的条文，无法给出有依据的回答。"
    "建议补充更具体的违法情形、地点，或确认是否属于本知识库覆盖的 6 部法规范围。"
)
UNAVAILABLE_ANSWER = "（未配置大模型，下面只给出召回的法条）"


class AnswerGenerator:
    """答案生成器：把检索结果变成带引用的答案。

    Input : Question + RetrievalResult
    Output: Answer

    `timeout` / `max_tokens` 是**兜底，不是调优**（与 `agent/llm.py` 同一套理由，那边记着
    24901 字 / 223 秒那次实测）。两个数的依据是 `data/traces/` 里 541 次真实答案生成的用量：
    非思考模型 p50 172、p95 442、最大 730（今天默认的 qwen-flash 是 135 次里最大 445），
    所以 1024 留了余量。**换成思考型当生成模型时必须调高** —— 思维链算进 `completion_tokens`：
    实测 `qwen3.8-max` 答 960 字用掉 5201 个，1024 会把它切在思维链中间。另注：`timeout` 拦的
    是「迟迟不来字节」（httpx 的 read timeout 是两次收到字节之间的上限），一直吐、一直在复读的
    那种只有 `max_tokens` 拦得住。
    """

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

    def generate(self, question: Question, retrieval: RetrievalResult) -> Answer:
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

    def stream(self, question: Question, retrieval: RetrievalResult):
        """流式生成：依次产出 `("delta", 文本片段)` 与末尾的 `("usage", dict)`。

        供 SSE 使用。**提示词与 `generate()` 完全共用** —— 两条路径若各写一份
        messages 构造，流式与非流式迟早会给出不一样的答案，而这种偏差极难发现。

        不做重试：已经吐出去的 token 收不回来。首字节之前的失败会抛出去由调用方
        （HTTP 层）翻译成错误事件，已经开始输出后的失败则让异常终止这次流。
        """
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
        """同 `_call_llm` 的两个上限，外加**要求末尾回一个 usage 块**。

        `stream_options={"include_usage": True}` 不是可选项：不加的话流式这条路上一个 token 数
        都拿不到（非流式的 `response.usage` 照常有），SSE 的 `done` 事件里 `usage` 恒为空 ——
        2026-09-22 实测确认。支持它的端点会在末尾补一个 `choices` 为空、只带 `usage` 的块，
        `stream()` 的循环本来就是照这个形状写的（先看 `chunk.choices` 再取 delta）。
        **不支持的端点会直接 400**，这是有意的：宁可响着坏，也不要静默少一个字段。
        """
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
        except Exception as exc:  # noqa: BLE001 - 首字节前失败，翻译成一句可读的话
            raise RuntimeError(f"调用 {self.cfg.model} 失败：{exc}") from exc

    def _call_llm(self, question: Question, evidences: list[Evidence]) -> tuple[str, dict]:
        from openai import OpenAI

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
            except Exception as exc:  # noqa: BLE001 - 统一重试
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"调用 {self.cfg.model} 失败：{last_error}") from last_error
