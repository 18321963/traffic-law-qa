# 交通法规问答 Agent · RAG 管道

面向驾驶员 / 驾校学员 / 交管客服的交通法规问答。**RAG 只是管道里的一个工具**，
交付物是完整管道：`docx → 结构层 → 检索层 → 索引 → 检索工具 → 生成`。

当前知识库：4 部法规，380 条，619 个检索块。

| 法规 | 版本 | 条数 |
|---|---|---|
| 中华人民共和国道路交通安全法 | 2021-04-29 | 124 |
| 中华人民共和国道路交通安全法实施条例 | 2017-10-07 | 115 |
| 深圳经济特区智能网联汽车管理条例 | 2026-05-27 | 64 |
| 深圳经济特区道路交通安全违法行为处罚条例 | 2024-05-10 | 77 |

---

## 1. 管道分层（每层一个类，输入输出固定）

| 层 | 类 | 输入 | 输出 |
|----|----|------|------|
| read     | `DocxReader`       | docx 路径 `Path`               | `list[Paragraph]` |
| parse    | `ParseStage`       | docx 目录                      | `list[LawDocument]` |
| chunk    | `ChunkStage`       | `list[LawDocument]`            | `ChunkSet`（父块 + 子块） |
| index    | `Indexer`          | `ChunkSet`                     | `IndexStats` |
| rewrite  | `QueryRewriter`    | `Query`                        | `RewrittenQuery` |
| retrieve | `HybridRetriever`  | `Query`                        | `RetrievalResult` |
| generate | `AnswerGenerator`  | `Question` + `RetrievalResult` | `Answer` |
| 编排     | `RagPipeline`      | 以上全部                       | `PipelineReport` / `Answer` |

层间契约全部定义在 `tools/contracts.py`（含各对象的 JSON 读写），
**磁盘格式变更只需改这一个文件**。查看层表：`python -m tools.pipeline layers`。

## 2. 目录结构

```
tools/
├── config.py           # 目录布局 + 模型端点 + 检索参数（全部走环境变量）
├── contracts.py        # 层间数据契约（唯一真源）
├── docx_reader.py      # read     层：标准库 zipfile + ElementTree 直读 docx
├── law_parser.py       # parse    层：LawParser / LawLibrary / ParseStage
├── chunker.py          # chunk    层：LawChunker / ChunkStage
├── indexer.py          # index    层：EmbeddingClient / BM25Index / VectorIndex / Indexer
├── query_rewriter.py   # rewrite  层：口语词法条用语对齐 + 法名线索
├── retriever.py        # retrieve 层：RRF 混合召回 + 父块回灌
├── generator.py        # generate 层：强制引用式作答
├── rag.py              # 门面：LegalRAG（对外只暴露 search / ask）
└── pipeline.py         # 编排 + CLI

法规知识库/
├── docx/               # 唯一真源，永不改写
├── text/*.md           # 人读层：法条 Markdown（人工比对用）
├── parsed/*.json       # 结构层：法 → 章 → 节 → 条 + manifest.json（sha1 增量门控）
├── chunks/             # 检索层：parents.jsonl（条）+ chunks.jsonl（款）
└── index/              # 索引层：bm25.json + chroma/ + index_meta.json
```

## 3. 快速开始

```powershell
# 1) 依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 配置模型（没有 key 也能跑，只是退化为纯 BM25 且不生成答案）
Copy-Item .env.example .env
#   然后填入 LLM_API_KEY / EMBED_API_KEY

# 3) 建库（可重复执行；docx 的 sha1 未变则自动跳过解析）
python -m tools.pipeline build

# 4) 问答 / 检索
python -m tools.pipeline ask "醉驾怎么处罚"
python -m tools.pipeline search "深圳 行人 在机动车道 罚款多少"
python -m tools.pipeline status
```

也可以只用管道的一部分：

```python
from tools.pipeline import RagPipeline

pipe = RagPipeline()
pipe.build()                                   # 建库
answer = pipe.ask("醉驾怎么处罚")                # 检索 + 生成
result = pipe.search("智能网联汽车道路测试")       # 只检索，不花 LLM 的钱
print(answer.render())                         # 正文 + 参考文献 + 提示
```

## 4. 检索设计（决定效果的四件事）

1. **父子块**：款级子块进索引，命中后按 `parent_id` 回灌**整条**给 LLM。
   只给一款的话，模型会看到「（一）兜售物品、散发广告或者乞讨；」这种半句。
2. **列举项合并**：`（一）（二）` 类列举项无条件并入引出它的那一款。
   未处理时这类孤立子块占 867 个中的 248 个（29%），合并后降到 619 块、均值 79 字。
3. **RRF 融合**：向量分与 BM25 分不同量纲不能相加，改用排名融合，任一条通道缺失都能继续跑。
4. **查询改写**（两个实测踩过的坑）：
   - *口语词命不中法条用语*：问「醉驾」时法条写的是「醉酒驾驶」，BM25 只能靠「处罚」硬凑，
     Top-1 召回的是「不按交通信号灯通行」。别名展开后 Top-1 纠正为道交法第九十一条。
   - *跨法规选址错误*：问深圳的事却召回国家法律一般条款。抽法名线索（「深圳」等）后，
     对应法规条文分数 ×1.5，Top-1 纠正为深圳处罚条例第八条。

## 5. 设计取舍

- **不用 MCP / docling 做入库解析**：MCP 是给 LLM 运行时按需调用的接口，返回整段 Markdown，
  入库要的是字段级可控、可复现、可增量的结构化产物；docling 面向 PDF/扫描件/表格，
  会重排段落从而破坏「第X条独占一段」这个根基。本语料是原生 DOCX 纯段落，
  标准库 `zipfile` + `ElementTree` 直读精度最高、依赖最轻。
- **不用 MCP 但保留交叉校验位**：`manifest.json` 的 `cross_check` 字段留着，
  后续可接 `markitdown-mcp` 把 docx 转 Markdown 后 diff 条数/段落数。
- **向量索引可选**：未配置 `EMBED_API_KEY` 时自动降级为纯 BM25，不阻塞开发。
- **Chroma 会连带装 onnxruntime（约 200MB）**：因为它把 onnxruntime 列为必需依赖，
  即使我们显式传入向量、禁用其默认嵌入模型也一样。

## 6. 待办

- 离线评测集（问题 → 期望条号）与召回率/引用命中率指标
- 多轮对话（`Question.history` 已预留）
- FastAPI + SSE 流式输出
