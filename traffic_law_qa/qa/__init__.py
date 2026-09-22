"""问答：一个问题进，一段带引用的答案出。三步，每步一个类。

    query_rewriter  rewrite   口语词法条用语对齐 + 法名线索   → RewrittenQuery
    retriever       retrieve  Milvus 双路召回 + 父块回灌       → RetrievalResult
    generator       generate  强制引用式作答（同步 + 流式）     → Answer
    rag             门面      把上面三步串成 LegalRAG，对外只开 search / ask / expand / stats

**`retriever` 是私有实现细节**：包内只有 `rag.py` 准 import 它，
上层（`pipeline` / `api` / agent / FastAPI）一律走 `LegalRAG`。
所以这里的 `__init__` 不再导出任何东西 —— 一导出就等于开了第二个口子。
"""
