"""`python -m traffic_law_qa.agent` —— 保留了重构前的命令。

搬进子包之后 `-m` 的路径本来会变成 `…agent.cli`，多一层没有意义的重复。
这个文件只做一次转发，让文档里、`.env.example` 里、脑子里记着的那条命令继续能用。
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
