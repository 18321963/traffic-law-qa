"""入口节点：这道题**该按哪个地区的规定**回答。

裁决只有两个落点，值直接决定检索的作用域：

    库内法规覆盖到的地区名（如「深圳」）   该地区的法规 ∪ 国家法（地方条例不能脱离上位法单独用）
    其它一切（`?`、判不出、库里没有的地区）  不限定，全库

**地区只做加法，不做减法。** 「这是全国性问题」听起来像是一条更窄的限定（只检索国家法，
把地方条例整个排除），但它今天不做、也不该做：库里 6 部法规有 2 部是深圳条例，而题面
**一个字不提地区、gold 却是深圳条例**的题，corpus 有 23/194、multihop 有 11/63 —— 减法是
按「题面像不像在问地方规定」切的，这 34 道题面全都不像，判成「全国」等于把它们的 gold
整类打死。反过来，加法判错（不该属于地方的判成了深圳）最坏也只是多带上一部法规，
今天更是一点代价都没有：库内只有深圳这一个地区，它的作用域就是全库。

所以模型答「全国」时不特殊处理 —— 它落在「库里没有这个地区」那条路上，一样是不限定。
真要做减法，得等库里有第二个地区、且这一档的判定准确率有实测数，那是另一件事。

为什么是**一次 LLM 调用**，而不用它替掉的那套「条文定位」那样的纯规则：地区词与法律适用地
经常不是一回事。「我买了一辆深圳产的无人驾驶出租车」里的深圳是产地，「我在深圳开网约车，
公司要我指定保险」里的深圳是场景设定 —— 这两题的答案都在国家法里，关键词规则会把它们判成
深圳。

**判不出、无 LLM、解析失败一律退回「不限定」** —— 与没有地区这个概念时逐位相同。
"""

from __future__ import annotations

import json

from ..contracts import ParentChunk
from .llm import ToolCallingLLM
from .prompts import REGION_SYSTEM_PROMPT
from .state import AgentState

__all__ = [
    "REGION_UNKNOWN",
    "law_scope",
    "national_law_ids",
    "make_region_node",
]

REGION_UNKNOWN = "?"

_LOCAL_MARKERS = ("省", "市", "自治区", "经济特区")
"""法规名里带这些词 = 地方性法规。

判据放在**法规名**上而不是写死一份 law_id 清单：新增一部地方条例时，没人会记得来这里改。
代价是名字里含「市」却不针对某地的法规（《城市道路管理条例》那种）会被误判成地方性 ——
库内 6 部没有这种；真出现了再收紧成「『省』『市』前面还得有地名」。
"""


def national_law_ids(parents: dict[str, ParentChunk]) -> tuple[str, ...]:
    """库内不针对特定地区的法规。"""
    return tuple(
        sorted(
            {
                parent.law_id
                for parent in parents.values()
                if not any(marker in parent.law_name for marker in _LOCAL_MARKERS)
            }
        )
    )


def law_scope(region: str, parents: dict[str, ParentChunk]) -> tuple[str, ...]:
    """地区裁决 → 这次检索限定到哪些 law_id。**空元组 = 不限定（全库）**。

    两条路都退回空元组，且都是宁可退全库：`?`（判不出），以及**库里找不到这个地区**
    （模型答「广州」，或答「全国」这种不是地区名的词 —— 后者也走这条）。拿一个库内
    不存在的地区去过滤，结果是一条都检不到。
    """
    if not region or region == REGION_UNKNOWN:
        return ()
    local = tuple(sorted({p.law_id for p in parents.values() if region in p.law_name}))
    if not local:
        return ()
    return tuple(sorted(set(national_law_ids(parents)) | set(local)))


def _parse_region(text: str) -> str:
    """从模型输出里抠出裁决。

    宽容解析：模型可能包了 ``` 代码块、可能前后带解释。**失败一律退回 `?`** —— 那是
    「不限定」，与没有这个节点时逐位相同。

    退回一个**真实地区名**今天是安全的（库内只有一个地区，它的作用域就是全库），所以这条
    断言钉的不是「会不会出事」，而是契约本身：`?` 是这里唯一允许的失败值。哪天库里有了
    第二个地区，这个值就从「无所谓」变成「有后果」。
    """
    raw = (text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return str(data.get("region") or "").strip() or REGION_UNKNOWN
    return REGION_UNKNOWN


def make_region_node(
    llm: ToolCallingLLM, parents: dict[str, ParentChunk], laws: list[str]
):
    """读：question / history
    写：{"region": 裁决, "region_scope": 由它派生的 law_id 集合}

    `region` 是模型的裁决（原话），`region_scope` 是**代码**据此算出的作用域。两个都留下，
    是为了让「模型说了什么」和「代码做了什么」在 trace 里各自可见 —— 判错时能一眼看出
    错在哪一环。

    温度用 0：这是判断题。

    **不写 `state["usage"]`**：那个通道被 `_trajectory_notes` 当作「规划轮数」来数，
    把一次入口判定混进去会让「N 轮 LLM 规划」当场变成 N+1。这次调用的成本在
    `--timing` 的 `node.region` 那一行里看得到。
    """

    prompt_head = REGION_SYSTEM_PROMPT % {
        "law_count": len(laws),
        "laws": "\n".join(f"- {name}" for name in laws),
    }

    def region_node(state: AgentState) -> dict:
        if not llm.available:
            return {"region": REGION_UNKNOWN, "region_scope": ()}

        prompt: list[dict] = [{"role": "system", "content": prompt_head}]
        prompt.extend({"role": r, "content": c} for r, c in state.get("history") or ())
        prompt.append({"role": "user", "content": state["question"]})

        reply, _usage = llm.chat(prompt, temperature=0.0, name="llm.region")
        region = _parse_region(reply.get("content") or "")
        return {"region": region, "region_scope": law_scope(region, parents)}

    return region_node
