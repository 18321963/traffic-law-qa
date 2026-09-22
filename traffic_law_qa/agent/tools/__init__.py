"""Agent 工具层：RAG 与 Agent 循环之间的边界（阶段二）。

**这一层不认识 LangGraph。** 它只做两件事：

1. 把 `LegalRAG` 包成一个模型能调用的工具（`SEARCH_LAW_TOOL` + 校验 + 渲染）；
2. 把 N 次检索的结果合并回**一个** `RetrievalResult`，好让收尾节点原样喂给
   既有的 `AnswerGenerator.generate()` —— 生成层因此一行都不用改。

之所以刻意不依赖框架：工具的参数校验、法名解析、合并去重都能脱离图单测，
延续项目「契约是唯一真源」的一贯做法。将来若把 LangGraph 换掉，这一层不用动。

    search_law 的完整边界：

        LLM → 工具   arguments 字符串 {"query": str, "law_name"?: str, "top_k"?: int}
        工具 → LLM   一段中文文本（引用 + 摘要 + 相关条号 + 降级提示）
                     —— 只印模型能据此动一下的东西，见 render.py 开头
        工具 → 图    一行 RetrievalResult.to_dict()（指针，不含正文）
        图  → 生成层  合并后的一个 RetrievalResult（含正文，正文由 parents 回灌）

    schemas.py    两个工具给模型看的 JSON schema + 名字与取值常量
    arguments.py  两份 schema 各自的 arguments 解析器 + tool_message
    articles.py   法名 / 条号 → 库内那一条（含精确取条的执行体）
    render.py     工具结果 → 给模型读的文本（摘要窗口 + 渲染）
    merge.py      N 次检索 → 一个 RetrievalResult，收尾节点与生成层之间的唯一接口

**这个 `__init__` 刻意什么都不 import**，与 `kb/` `qa/` `eval/` 同一条纪律：
子包只按路径说话（`from .tools.articles import lookup_article`），不在 `__init__`
里 re-export —— 多一个 import 口子，就多一处将来会分叉的地方。
"""
