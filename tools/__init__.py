"""交通法规 RAG 管道：对外只开一个口子 —— `qa()`。

    from tools import qa

    print(qa("醉驾怎么处罚").render())                 # 确保索引就绪 → 检索 → 生成
    print(qa("深圳 行人 在机动车道", mode="search"))     # 只检索，不花 LLM 的钱
    print(qa("醉驾怎么处罚", debug=True))               # 附双通道排名

命令行入口（等价于一次 `qa()` 调用）：

    python -m tools "醉驾怎么处罚"
    python -m tools "深圳 行人 在机动车道" --search --debug

内部仍然是七层管道，每层一个类、输入输出固定：

    docx_reader -> law_parser -> chunker -> indexer（写 milvus_store）
                -> query_rewriter -> retriever -> generator
              （由 pipeline.RagPipeline 编排，rag.LegalRAG 是管道级门面）

每个模块都可以单独作为脚本执行，便于分步调试：

    python -m tools.docx_reader
    python -m tools.law_parser --force
    python -m tools.chunker --show 第八条
    python -m tools.indexer --query 醉驾
    python -m tools.pipeline build | status | layers
"""

from .api import QaError, qa, render

__all__ = ["qa", "QaError", "render"]
