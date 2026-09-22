"""交通法规 RAG 管道：对外只开一个口子 —— `qa()`。

    from traffic_law_qa import qa

    print(qa("醉驾怎么处罚").render())                 # 确保索引就绪 → 检索 → 生成
    print(qa("深圳 行人 在机动车道", mode="search"))     # 只检索，不花 LLM 的钱
    print(qa("醉驾怎么处罚", debug=True))               # 附双通道排名

命令行入口（等价于一次 `qa()` 调用）：

    python -m traffic_law_qa "醉驾怎么处罚"
    python -m traffic_law_qa "深圳 行人 在机动车道" --search --debug

内部仍然是七层管道，每层一个类、输入输出固定：

    docx_reader -> law_parser -> chunker -> indexer（写 milvus_store）
                -> query_rewriter -> retriever -> generator
              （由 pipeline.RagPipeline 编排，rag.LegalRAG 是管道级门面）

**「门面」是字面意思**：`retriever` 那一层是 `LegalRAG` 的实现细节 ——
`qa/rag.py` 是包内唯一 import 它的模块，它的 `store` / `chunks` /
`rewriter` 都是私有属性。上层要什么能力，就在 `LegalRAG` 上加一个方法。
所以 `qa()`、`RagPipeline`、`AgentRunner`、FastAPI 服务调的都是同一个 `LegalRAG`，
四家不各拿一份检索器。

命令行入口全在下面这几处，其余模块都是纯库（`kb/` 那四个、`agent/cli.py`、`eval/harness.py`
都不带 `__main__` 块）—— **子包名就是包路径的一部分**：

    python -m traffic_law_qa "问题" [--agent]        # 问答唯一一条门；--agent 才走 Agent 循环
    python -m traffic_law_qa.pipeline build | status | layers
    python -m traffic_law_qa.pipeline docx | parse | chunk | index    # 建库四步，可以只重跑一步
    python -m traffic_law_qa.eval | …eval.corpus | …eval.singlehop | …eval.multihop
    python -m traffic_law_qa.server

全清单（含 `tlq-qa` / `tlq-serve` 两个短命令）与各自旗标见 README 的「所有入口」。
"""

from .api import QaError, qa, render

__all__ = ["qa", "QaError", "render"]
