"""评测：四份题集文件、四条命令，测管道里的不同段。

    corpus.py    题集文件的唯一生成者与读者：源语料 → 三份桶文件
                 ——  `python -m traffic_law_qa.eval.corpus build | check`
    harness.py   题集 A 的跑分器：hit@k / MRR，外加「规则取条」探针臂
                 ——  `python -m traffic_law_qa.eval [--reference]`
    multihop.py  题集 C：跨法多跳，护栏 + agent/rag 轨迹对照
                 ——  `python -m traffic_law_qa.eval.multihop --check`
    singlehop.py 题集 A 上的 agent/rag 对照：单跳题上 Agent 那一层值不值
                 ——  `python -m traffic_law_qa.eval.singlehop --compare`（花钱）

    data/eval_retrieval.json   82 道   检索指标（harness / singlehop）
    data/eval_reference.json  112 道   规则取条那条臂（harness --reference）
    data/eval_nogold.json      45 道   存档：构造不出 ground truth，每道带成因
    data/eval_multihop.json    63 道   跨法多跳，自己一套造题与护栏

四份题集的 gold 长短、分母、走管道哪一段都不同，**数字不可跨表比较**。
前三份由 corpus.py 从 data/eval_corpus.json 切出来，改完源语料记得 `corpus build`。

这些模块只在**离线评测**时被调用，不在问答链路上 —— 所以 `qa()` 那条路径
永远不会 import 到这里。
"""
