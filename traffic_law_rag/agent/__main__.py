"""`python -m traffic_law_rag.agent` —— 保留了重构前的命令。

搬进子包之后 `-m` 的路径本来会变成 `…agent.graph`，多一层没有意义的重复。
这个文件只做一次转发，让文档里、`.env.example` 里、脑子里记着的那条命令继续能用。

（`python -m traffic_law_rag.agent.graph` 同样能跑 —— graph.py 自带 `__main__` 块。）
"""

from .graph import main

if __name__ == "__main__":
    raise SystemExit(main())
