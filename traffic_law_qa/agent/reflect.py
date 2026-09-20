"""审核节点：现有证据够不够回答、不够的那部分补不补得上。

**它不绑工具** —— 这是它与 `agent` 节点的根本区别。只问两个判断题，所以停止信号是一句
明确的 yes/no，而不是「模型这次恰好没调工具」。为什么值得为它单开一个节点，见
`docs/DESIGN.md` §9「图」。

预算见底时**不调模型**直接短路：那一轮反正要收尾，问了也是白花钱。
"""

from __future__ import annotations

import json

from .. import config
from .llm import ToolCallingLLM
from .prompts import REFLECT_SYSTEM_PROMPT
from .state import AgentState

__all__ = ["make_reflect_node"]


def _parse_reflection(text: str) -> dict:
    """从模型输出里抠出判断结果。

    宽容解析：模型可能包了 ``` 代码块、可能前后带解释。**解析失败一律判为
    sufficient=True**（停止）而不是 False —— 失败时该往「少花一次检索的钱、直接作答」
    的方向倒，而不是让一个解析 bug 变成多跑一轮的循环。

    `retrievable` 缺失时默认 **True**（照旧回边再检一次），不是 False。理由：
    这个字段是给「缺口在不在库内」用的，而**不确定时不该替模型做停止的决定** ——
    默认 True 时行为与加这个字段之前逐位相同，默认 False 则会在模型不输出它时
    静默改掉循环行为。那不是「省了一轮」，是换了套逻辑。

    `next_query`（缺口写成的一句问话）缺失时默认 **空串**，空串 = 不给规划轮这句提示，
    也就是与加这个字段之前逐位相同。它只是给下一轮的**补充信息**，缺了不影响回边与否。
    """
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return {
                "sufficient": bool(data.get("sufficient", True)),
                "retrievable": bool(data.get("retrievable", True)),
                "reason": str(data.get("reason") or "").strip(),
                "missing": str(data.get("missing") or "").strip(),
                "next_query": str(data.get("next_query") or "").strip(),
            }
    return {
        "sufficient": True,
        "retrievable": True,
        "reason": f"审核输出无法解析，按已足够处理：{raw[:60]}",
        "missing": "",
        "next_query": "",
    }


def make_reflect_node(llm: ToolCallingLLM, cfg: config.AgentConfig, laws: list[str]):
    """读：question / history / messages / steps / max_steps / search_log
    写：{"reflections": [结论], "messages": [给规划轮的补充说明]}

    温度用 0：这是判断题，不要它发挥。

    `laws` 是**库内全部法规名**，作为「库的边界」写进提示词。没有它，审核分不清
    「这一轮没检出来」和「库里根本没有」这两件事，于是一律判「不够」，把预算全烧在
    补不上的缺口上。名单纯从 `parents` 现取，不写死一份：新增法规时不该有人记得来这里改。

    预算已尽时**不调模型**直接短路：没有剩余轮次可行动，「够不够」的答案不影响任何
    后续决策，问它纯属浪费一次调用。
    """

    # 前缀只算一次：库边界在一次运行内不变，而这是每次审核都要带上的固定开销。
    prompt_head = REFLECT_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {name}" for name in laws),
    }

    def reflect_node(state: AgentState) -> dict:
        max_steps = state.get("max_steps", cfg.max_steps)
        if max_steps - state.get("steps", 0) <= 0:
            return {
                "reflections": [
                    {
                        "sufficient": False,
                        "retrievable": True,
                        "reason": f"已达最大轮数 {max_steps}",
                        "missing": "",
                        "next_query": "",
                    }
                ]
            }

        prompt: list[dict] = [{"role": "system", "content": prompt_head}]
        prompt.extend({"role": r, "content": c} for r, c in state.get("history") or ())
        prompt.append({"role": "user", "content": state["question"]})
        prompt.extend(state.get("messages") or ())

        reply, _usage = llm.chat(prompt, temperature=0.0)
        reflection = _parse_reflection(reply.get("content") or "")

        update: dict = {"reflections": [reflection]}
        # 把「还缺什么」回灌给规划轮。用 user 角色：工具消息之后任何角色都合法，
        # 而 user 是各家兼容端点支持得最稳的那个。
        #
        # **只在真要回边时才回灌**：判成「库外」的那条缺口马上就收尾了，这条消息没人会读到，
        # 留在历史里反而与「就此停止」的决定自相矛盾。
        if not reflection["sufficient"] and reflection["retrievable"] and reflection["missing"]:
            content = f"[检索审核] 还缺：{reflection['missing']}"
            # 审核顺手把这个缺口写成了**一句问话**，规划轮可以直接拿它当检索词。
            # 没写（旧模型/解析不出来）就只发上面那半句 —— 与加这个字段之前逐位相同。
            if reflection.get("next_query"):
                content += f"\n[检索审核] 建议查：{reflection['next_query']}"
            update["messages"] = [{"role": "user", "content": content}]
        return update

    return reflect_node
