"""末端复核：逐个 [依据N] 判断它引的原文是否支撑那句结论，低分降级为「依据不足」。

    make_review_node : AgentState（只读 answer）→ {"answer": 换过文案的 Answer}

**分数由代码算，模型只给判据。** 模型逐条回 `{"n": k, "supported": bool}`，
`score = supported / total` 在这里聚合。让模型直接吐一个 0~1 的数，既没法复现也没法解释；
逐条判据则每一条都能拿原文去对，也才能在 trace 里指名道姓地说「哪一条不支撑」。

**编号对不上任何一条依据时，不花模型的钱，也永不算支撑。** 答案引了 `[依据9]` 而这次只给了
6 条依据 —— 那种「原文」根本不存在，代码当场就能判。实测 120 条真答案里有 5 条是这样
（最大 `依据97`），所以这条不是假想的分支。判据是**标签对不上**而不是「编号超出条数」：
给模型看的每一块原文都带着自己的标签，只有标签对得上，它判的才是答案引的那一条。

**`total` 是「引用到的编号个数」，不是「证据条数」。** 引了 5 条错了 1 条 = 4/5，
而不是 4/6 —— 分母是这次**主张**了几条依据，没引的那几条不该替它摊薄。

**一个 [依据N] 都没有时不打分、不拦截。** 走到这里的是两句常量文案（检索为空、未配置模型），
它们本来就是「不给结论」的形态；再拦一次等于把「没配 key」报成「依据不足」。

**失败即放行。** 复核是闸不是依赖：没模型、解析失败、调用抛异常，一律原样输出 + 一行
「复核未完成」的 note，唯一拦得住的理由是「分真的低」。这里的 `try` 与 `region.py`
不包 `chat` 是有意的差别 —— 那边判错只是检索范围变宽，这边抛出去会炸掉整次提问。

**不写 `state["usage"]`**：那个通道被 `_trajectory_notes` 当作「规划轮数」数，混进一次复核
会让「N 轮 LLM 规划」凭空多一轮（理由同 `region.py`）。这次调用的成本在 `--timing` 的
`node.review` 那一行里看得到。

**载荷用 f-string 拼，绝不用 `%` 或 `.format`**：法条原文是自由文本，实测 508 条父块里有
1 条含 `%` —— 走模板会被打穿。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from .. import config
from ..contracts import Evidence, Review
from ..qa.generator import REVIEW_DOWNGRADE_ANSWER
from .llm import ToolCallingLLM
from .prompts import REVIEW_SYSTEM_PROMPT
from .state import AgentState

__all__ = ["cited_labels", "downgrade_text", "parse_judgments", "make_review_node"]

CITE_RE = re.compile(r"[\[【]依据\s*(\d+)[\]】]")
"""答案里的引用标注。**半角 `[依据3]` 与全角 `【依据3】` 都认** —— 实测 120 条真答案里有 2 条
在同一条答案里两种混用；只认一种会把那几条当成「没引这一条」，分母随之偏小、分数虚高。"""


def cited_labels(text: str) -> list[int]:
    """答案里出现过的编号：去重、按首次出现排序。"""
    return list(dict.fromkeys(int(n) for n in CITE_RE.findall(text or "")))


def _as_bool(value: object) -> bool | None:
    """判据归一化：JSON `true` → True；`"true"` / `1` → True；`"false"` / `0` → False；
    认不出的（缺字段、`"yes"`、`2`、对象）→ None = 未判，按不支撑计。

    **这一层必须显式写。** `isinstance(True, int)` 是 True，所以任何 `if value` 或数值比较的
    写法都会把 `0`/`False`、`1`/`True` 混成一谈；而反过来写成 `value is True` 又会把模型
    爱吐的 `1` 判成「未判」—— 那等于让一个排版习惯决定这条依据算不算数。
    """
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
    """从模型输出里抠出逐条判据，失败返回空字典（= 复核未完成）。

    宽容解析照抄 `region._parse_region`：取「首个 `{` 到末个 `}`」，模型把 JSON 包在 ``` 里、
    前后再带一句解释都吃得下（围栏在花括号外面）。**不剥围栏再解析**、也不假设字段齐全 ——
    解析失败与字段缺失是两件事，前者放行，后者按不支撑计，见 `_as_bool`。
    """
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
            continue  # `{"n": true}` 不是编号，别拿 int(True)=1 当一条判据
        try:
            out[int(number)] = _as_bool(item.get("supported"))
        except (TypeError, ValueError):
            continue
    return out


def _labels_text(numbers) -> str:
    return "、".join(f"依据{n}" for n in numbers)


def downgrade_text(review: Review, evidences: tuple[Evidence, ...]) -> str:
    """替换掉原答案的那段话：一句定性 + 分数 + 需人工复审 + 候选法条清单。

    列的是**这次检索到的全部证据**（不是「不支撑的那几条」）：不支撑说的是「这条推不出那句结论」，
    不等于「这条法条不相干」—— 人工复审要看的恰恰是这个区别，替他先筛掉等于替他下结论。
    """
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
    """三样东西：问题、答案全文、**被引用到的那几条**法条原文。

    只给被引用到的（不是全部证据）：复核要判的是「这一条撑不撑得住它引的那句话」，
    没被引的条文进来只会稀释注意力、白烧 token。

    整条正文不截断 —— 实测 508 条父块最长 541 字，比截断省下的那点 token 值钱得多的是
    「拿半条法条判支撑」带来的误判。
    """
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


def make_review_node(llm: ToolCallingLLM, cfg: config.AgentConfig):
    """读：answer。写：`{"answer": 换过文案与 notes 的 Answer}`。

    **键集恰好是 `{"answer"}`** —— 复核结果挂在 `Answer.review` 上，不新增 state 键：
    langgraph 对未声明的键是**静默丢弃**（不报错、不警告），多加一个键会让这里看起来写了、
    实际什么都没留下。

    阈值 `cfg.review_min_score`：`< 0` 关掉整个节点（连模型都不调），`0` 只打分不拦截，
    `> 0` 才拦。用分数而不是布尔开关当总闸，是因为这个仓的 env 约定是「非空串即真」——
    `AGENT_REVIEW=0` 会被读成**开**。
    """

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
        # **按标签认证据，不按位置认。** `evidences[n-1]` 只在「标签恰好是 1..N 顺序」时才等价，
        # 而这一步错了是**静默**的：给模型看的第 3 块可能是标签写着「依据4」的原文，于是它判的
        # 是另一条法条，分数看着正常、其实判歪了。2026-09-22 的 B 脚本就是这样：它只拼了被引的
        # 4 条（标签 2~5），按位置取到「依据3」时拿到的是标签为「依据4」的原文。
        # 标签对不上 = 答案引了一条这次没给的依据 —— 与越界同一种错，一样永不支撑。
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
            except Exception as exc:  # noqa: BLE001 - 失败即放行，见模块开头
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
            model=llm.cfg.model,
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
