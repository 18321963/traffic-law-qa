"""`python -m traffic_law_rag.eval [--reference]` —— 保留了重构前的命令。

搬进子包之后路径本该变成 `…eval.harness`，多一层没有意义的重复。这里只做转发。

跨法多跳那一套不在这里，它有自己的路径：`python -m traffic_law_rag.eval.multihop`。
"""

from .harness import main

if __name__ == "__main__":
    raise SystemExit(main())
