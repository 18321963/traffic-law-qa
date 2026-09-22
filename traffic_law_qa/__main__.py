"""`python -m traffic_law_qa "问题"` —— 一条命令拿到答案（等价于一次 `qa()` 调用）。

    python -m traffic_law_qa "醉驾怎么处罚"
    python -m traffic_law_qa "深圳 行人 在机动车道 罚款多少" --search --debug
    python -m traffic_law_qa "在深圳，不按规定使用安全带罚多少？" --agent

**问答只有这一条门**：带 `--agent` 走 Agent 循环，不带就是线性管道（基线 `qa()`）。
agent 那一半的实现仍在 `agent/cli.py`，但说明书、公共旗标与 `--agent` 分派都在这里 ——
于是 `agent/cli.py` 没有自己的 `-h` 与 `__main__` 块，`python -m traffic_law_qa.agent`
这个路径不存在（两处各写一份说明书必漂，见 README「所有入口」）。

失败只打印一行中文提示（含该执行的命令），不糊 pymilvus 的堆栈。
"""

from __future__ import annotations

import argparse
import json
import sys

from .api import MODE_SEARCH, QaError, qa, render
from .contracts import Answer

USAGE = """用法：python -m traffic_law_qa "问题" [选项]

默认走线性管道（确保索引就绪 → 检索 → 生成，也就是 `qa()`，评测的基线）；
加 `--agent` 走 Agent 循环（模型自己决定查什么、查几轮）。

两条路都认：
    --top-k N       覆盖默认召回条数 / 证据条数（默认取 RAG_TOP_K，6）
    --no-vector     只用 BM25 通道
    --json          输出 JSON 而不是人读文本

线性（默认）另认：
    --search        只检索，不调用大模型（不花 LLM 的钱）
    --debug         打印每条法条被稠密向量 / BM25 各自排到第几
    --rebuild       无视一致性检查强制重建索引（改了切分/解析逻辑后用）

Agent（--agent）另认（解析在 agent/cli.py，这里只列名目）：
    --trace          打印完整决策链（**这本来就是默认**；写出来只为让默认可点名，
                     给不给都一样。想反过来——要 JSON 也要轨迹——目前没有开关）
    --timing         额外打印每个节点的耗时表，走 stderr（不污染 --json）
    --max-steps N    规划轮数上限，默认 3。设 1 可做近似基线的 A/B
    --langfuse       把节点状态、模型调用与检索上报到 Langfuse 云端
                     （**这本来就是默认** —— .env 里配了 LANGFUSE_PUBLIC_KEY /
                     LANGFUSE_SECRET_KEY 就自动开，写出来只为让默认可点名，同 --trace。
                     还需 pip install -e ".[langfuse]"；没配就一个字节都不外发）
    --no-langfuse    强制关掉上报（跑批时不想让每一题都往云端灌就加这个）

决策链在终端里会给「没取到」那种行上色（黄）；重定向到文件时自动不上色。

示例：
    python -m traffic_law_qa "醉驾怎么处罚"
    python -m traffic_law_qa "深圳 行人 在机动车道 罚款多少" --search --debug
    python -m traffic_law_qa "在深圳，不按规定使用安全带罚多少？" --agent --timing
"""


def _parser() -> argparse.ArgumentParser:
    """线性那一半的解析器（agent 那一半在 `agent/cli.py`）。

    `add_help=False` + 在 `main` 开头手工认 `-h`：与 agent / eval 那几处同形 —— argparse
    自动生成的帮助不如上面那份 `USAGE` 写得全（里面有「本来就是默认」这类语义），不该抢它
    的活。这个解析器的唯一职责是**把不认识的开关和取不到值的开关变成错误**，而不是像原先
    那样静默忽略（`--help` 也不再只在第一个参数位置才认）。
    """
    parser = argparse.ArgumentParser(prog="python -m traffic_law_qa", add_help=False)
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--search", action="store_true", help="只检索，不调用大模型")
    parser.add_argument("--debug", action="store_true", help="打印双通道排名")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖默认召回条数")
    parser.add_argument("--no-vector", action="store_true", help="只用 BM25 通道")
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser


def _print_timing(result: object) -> None:
    retrieval = getattr(result, "retrieval", None)
    if retrieval is None:
        return
    parts = []
    usage = getattr(result, "usage", None)
    if usage:
        parts.append(f"tokens {usage}")
    parts.append(f"检索 {retrieval.elapsed_ms:.0f}ms")
    parts.append(f"总计 {getattr(result, 'elapsed_ms', 0.0):.0f}ms")
    print(f"\n[{' | '.join(parts)}]")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return 0
    if not args:
        print(USAGE)
        return 2

    # `--agent` **先摘出去再分派，顺序不能反**：线性解析器不认识 `--trace` / `--max-steps`，
    # 反过来先解析会把 agent 的旗标当未知开关打回。摘掉的只一个（问题正文里带这四个字是
    # 另一个 token，不会误伤）。
    # `import` 在分支里，是为了守住那条线：`import traffic_law_qa` 不许拉起 langgraph。
    if "--agent" in args:
        args.pop(args.index("--agent"))
        from .agent.cli import main as agent_main

        return agent_main(args)

    try:
        options = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if not options.question:
        print(USAGE)
        return 2

    debug = options.debug
    try:
        result = qa(
            options.question,
            mode=MODE_SEARCH if options.search else "ask",
            top_k=options.top_k,
            debug=debug,
            with_vector=not options.no_vector,
            rebuild=options.rebuild,
        )
    except QaError as exc:
        print(str(exc))
        return 1

    if options.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(render(result, debug=debug))
    if isinstance(result, Answer):
        _print_timing(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
