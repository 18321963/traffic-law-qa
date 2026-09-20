"""建库：docx 进，Milvus 集合出。四步全部只在 `pipeline build` 时跑。

    docx_reader   read     标准库 zipfile + ElementTree 直读 docx → list[Paragraph]
    law_parser    parse    法 → 章 → 节 → 条 + sha1 增量门控        → list[LawDocument]
    chunker       chunk    条级父块 + 款级子块                      → ChunkSet
    milvus_store  存储     集合 schema + BM25 函数 + hybrid_search（Milvus 的门面）
    indexer       index    切块 → 向量化 → 写集合 + 索引快照         → IndexStats

这里**不做** `__init__` 再导出：一次 import 就把五步全拉进来，会让「只想用切块器」
的调用方（比如测试）连带拖上 pymilvus。要用哪个就 import 哪个，路径即语义。
"""
