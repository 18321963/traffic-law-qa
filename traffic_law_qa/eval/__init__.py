"""评测：三套题集、四条命令，测管道里的不同段。

    harness.py   题集 A/B 的跑分器：hit@k / MRR，外加「规则取条」探针臂
                 ——  `python -m traffic_law_qa.eval [--reference]`
    multihop.py  题集 C：跨法多跳，护栏 + agent/rag 轨迹对照
                 ——  `python -m traffic_law_qa.eval.multihop --check`
    singlehop.py 题集 A 上的 agent/rag 对照：单跳题上 Agent 那一层值不值
                 ——  `python -m traffic_law_qa.eval.singlehop --compare`（花钱）

三套题集的 gold 长短、分母、走管道哪一段都不同，**数字不可跨表比较**，见 docs/DESIGN.md。

这些模块只在**离线评测**时被调用，不在问答链路上 —— 所以 `qa()` 那条路径
永远不会 import 到这里。
"""
