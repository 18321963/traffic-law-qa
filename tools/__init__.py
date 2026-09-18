"""交通法规 RAG 管道的各阶段实现。

管道顺序：
    docx_reader -> law_parser -> chunker -> indexer（写 milvus_store）
                -> query_rewriter -> retriever -> generator
              （由 pipeline.RagPipeline 编排，rag.LegalRAG 是对外门面）

每个模块都可以单独作为脚本执行，便于分步调试：
    python -m tools.docx_reader
    python -m tools.law_parser --force
    python -m tools.chunker --show 第八条
    python -m tools.indexer --query 醉驾
    python -m tools.pipeline ask "醉驾怎么处罚"
"""
