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
docker-compose.yml      # Milvus Standalone：etcd + MinIO + Milvus（数据卷在 volumes/，已忽略）

tools/
├── config.py           # 目录布局 + 模型端点 + Milvus + 检索参数（全部走环境变量）
├── contracts.py        # 层间数据契约（唯一真源）
├── docx_reader.py      # read     层：标准库 zipfile + ElementTree 直读 docx
├── law_parser.py       # parse    层：LawParser / LawLibrary / ParseStage
├── chunker.py          # chunk    层：LawChunker / ChunkStage
├── milvus_store.py     # 存储层：集合 schema + BM25 函数 + hybrid_search
├── indexer.py          # index    层：EmbeddingClient / Indexer（建集合、写索引快照）
├── query_rewriter.py   # rewrite  层：口语词法条用语对齐 + 法名线索
├── retriever.py        # retrieve 层：Milvus 双路召回 + 父块回灌
├── generator.py        # generate 层：强制引用式作答
├── rag.py              # 门面：LegalRAG（对外只暴露 search / ask）
└── pipeline.py         # 编排 + CLI

法规知识库/
├── docx/               # 唯一真源，永不改写
├── text/*.md           # 人读层：法条 Markdown（人工比对用）
├── parsed/*.json       # 结构层：法 → 章 → 节 → 条 + manifest.json（sha1 增量门控）
├── chunks/             # 检索层：parents.jsonl（条）+ chunks.jsonl（款）
└── index/              # 索引快照：index_meta.json（向量数据在 Milvus 里）
```

## 3. 快速开始

```powershell
# 1) 依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 起 Milvus（etcd + MinIO + Milvus，首启约 60~90 秒转 healthy）
docker compose up -d --wait

# 3) 配置模型（没有 key 也能跑：集合会退化为纯 BM25，且不生成答案）
Copy-Item .env.example .env
#   然后填入 LLM_API_KEY / EMBED_API_KEY

# 4) 建库（重建 Milvus 集合；docx 的 sha1 未变则自动跳过解析）
python -m tools.pipeline build

# 5) 问答 / 检索 / 状态
python -m tools.pipeline ask "醉驾怎么处罚"
python -m tools.pipeline search "深圳 行人 在机动车道 罚款多少" --debug
python -m tools.pipeline status
```

`--debug` 会额外跑一次单通道检索，把每条法条是被稠密向量还是被 BM25 捞到的、
各自排名多少都打出来，调检索时很有用。

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
3. **RRF 融合交给 Milvus**：`hybrid_search` + `RRFRanker(k)` 在服务端完成，
   稠密通道用 COSINE、稀疏通道用 BM25；客户端不再自己拼排名。
4. **BM25 稀疏向量也是服务端生成的**：文本字段 `enable_analyzer=True`（jieba 分词）+
   一个 `FunctionType.BM25` 函数，插入与查询都只给原文。
   因此项目里没有自研倒排索引、没有 `bm25.json`、连 jieba 这个 pip 依赖都不需要了。
5. **查询改写**（两个实测踩过的坑）：
   - *口语词命不中法条用语*：问「醉驾」时法条写的是「醉酒驾驶」，BM25 只能靠「处罚」硬凑，
     Top-1 召回的是「不按交通信号灯通行」。别名展开后 Top-1 纠正为道交法第九十一条。
   - *跨法规选址错误*：问深圳的事却召回国家法律一般条款。抽法名线索（「深圳」等）后，
     对应法规条文分数 ×1.5，Top-1 纠正为深圳处罚条例第八条。

## 5. 设计取舍

- **数据库选 Milvus 而不是 Chroma**：Chroma 只有稠密向量，BM25 得自己实现
  （分词、df/postings、落盘），两路融合也得手写；Milvus 用 `FunctionType.BM25`
  在服务端把原文转成稀疏向量，`hybrid_search` 直接给出融合结果，
  一个集合就把稠密 + 稀疏 + 过滤三件事都覆盖了。
  代价是多了一层 Docker（etcd + MinIO + Milvus 三容器），不适合无 Docker 环境。
- **不用 MCP / docling 做入库解析**：MCP 是给 LLM 运行时按需调用的接口，返回整段 Markdown，
  入库要的是字段级可控、可复现、可增量的结构化产物；docling 面向 PDF/扫描件/表格，
  会重排段落从而破坏「第X条独占一段」这个根基。本语料是原生 DOCX 纯段落，
  标准库 `zipfile` + `ElementTree` 直读精度最高、依赖最轻。
- **不用 MCP 但保留交叉校验位**：`manifest.json` 的 `cross_check` 字段留着，
  后续可接 `markitdown-mcp` 把 docx 转 Markdown 后 diff 条数/段落数。
- **纯 BM25 可跑**：未配置 `EMBED_API_KEY` 时集合不建稠密字段，
  跑完 `build` 就是一套可用的关键词检索，不阻塞开发。

## 6. 待办

- 离线评测集（问题 → 期望条号）与召回率/引用命中率指标
- 多轮对话（`Question.history` 已预留）
- FastAPI + SSE 流式输出
