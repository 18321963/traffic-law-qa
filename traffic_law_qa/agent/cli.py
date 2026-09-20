"""`python -m traffic_law_qa.agent "问题"` 的入口。

`agent/__main__.py` 只是一次转发，让包级 `-m` 路径能落到这里。

选项的文档在 `USAGE` 里，改参数时两处一起改。
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from .. import config
from ..obs import Recorder
from .graph import AgentRunner
from .intent import INTENT_LOOKUP, INTENT_SEARCH
from .trace import render_timing, render_trace

__all__ = ["main", "USAGE"]


USAGE = """用法：python -m traffic_law_qa.agent "问题" [选项]

  --trace          打印完整决策链（默认打印）
  --json           输出 JSON（Answer.to_dict）
  --timing         额外打印每个节点的耗时表，走 stderr（不污染 --json）
  --max-steps N    规划轮数上限，默认 2。设 1 可做近似基线的 A/B
  --top-k N        证据条数上限，默认取 RAG_TOP_K
  --no-vector      只用 BM25 检索
  --linear         走单轮管道作对照（不发规划轮请求）
  --intent NAME    强制意图，用于对照：lookup/条文定位 或 search/法规检索。
                   默认按规则判。给含条号的问题加 --intent search，
                   就能看到「不做意图识别」时同一道题会怎样。

决策链在终端里会对判别行上色（够了=绿，还不够/未取到=黄）；重定向到文件时自动不上色。
"""

# --intent 的取值别名；中文原值也直接收，省得记两套词
INTENT_ALIASES = {
    "lookup": INTENT_LOOKUP,
    "条文定位": INTENT_LOOKUP,
    "search": INTENT_SEARCH,
    "法规检索": INTENT_SEARCH,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args or not args:
        print(USAGE)
        return 0

    question = args[0]

    def option(name: str, cast=int):
        if name in args and args.index(name) + 1 < len(args):
            return cast(args[args.index(name) + 1])
        return None

    overrides = {}
    if option("--max-steps"):
        overrides["max_steps"] = option("--max-steps")
    cfg = replace(config.agent_config(), **overrides) if overrides else None

    forced_intent = None
    if option("--intent", str) is not None:
        raw = option("--intent", str)
        forced_intent = INTENT_ALIASES.get(raw, "")
        if not forced_intent:
            print(f"--intent 只认 {'、'.join(INTENT_ALIASES)}，收到「{raw}」")
            return 1

    # 上色只在人对着终端看时才有意义；重定向进 data/traces/*.log 时必须整片关掉，
    # 否则日志里全是转义序列。
    color = sys.stdout.isatty()
    recorder = Recorder() if "--timing" in args else None

    try:
        runner = AgentRunner.load(
            with_vector="--no-vector" not in args,
            cfg=cfg,
            top_k=option("--top-k"),
            forced_intent=forced_intent,
            tracer=recorder,
        )
    except Exception as exc:  # noqa: BLE001 - Milvus 未起、索引未建等
        print(f"装配失败：{exc}")
        print("提示：先 docker compose up -d standalone，并确认索引已建（python -m traffic_law_qa.kb.indexer）")
        return 1

    print(f"[agent] {runner.describe()}")

    if "--linear" in args:
        # 同一个 rag：对照实验要的是「同一套检索 + 同一个生成器，只是不走图」
        answer = runner.rag.ask(question)
    else:
        try:
            state = runner.invoke(question)
        except RuntimeError as exc:
            print(str(exc))
            return 1
        if "--json" not in args:
            print(render_trace(state, color=color))
        answer = state.get("answer")

    # 走 stderr：`--json` 的 stdout 要留给 JSON，而 `[agent] describe` 已经占了第一行，
    # 计时表再挤进去只会让下游更难解析。
    if recorder is not None:
        print(render_timing(recorder.summary(), color=color), file=sys.stderr)

    if answer is None:
        print("未产出答案")
        return 1

    if "--json" in args:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print()
    print(answer.text)
    if answer.evidences:
        print()
        print("依据：")
        for evidence in answer.evidences:
            print(f"  {evidence.label} {evidence.citation}")
    if answer.usage:
        print(f"\n[tokens] {answer.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
