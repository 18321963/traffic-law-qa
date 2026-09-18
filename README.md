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

层间契约全部定义在 `traffic_law_rag/contracts.py`（含各对象的 JSON 读写），
**磁盘格式变更只需改这一个文件**。查看层表：`python -m traffic_law_rag.pipeline layers`。

七层之上只开一个口子：`from traffic_law_rag import qa`，一次调用拿到结果（见第 4 节）。

## 2. 目录结构

```
docker-compose.yml      # Milvus Standalone（etcd + MinIO + Milvus）+ 无状态的应用容器
Dockerfile              # 应用镜像；.dockerignore 挡住密钥与 162M 的 volumes/
pyproject.toml          # 打包与依赖，pip install -e ".[all]" 一次装齐

data/
└── eval_corpus.json    # 评测语料（779 条 instruction/output，用前必须先筛，见第 6 节）

examples/
└── demo.py             # 4 个典型问题跑一遍：问题 → 命中条号 → 答案要点

tests/                  # 离线测试：不连 Milvus、不发网络请求（见第 8 节）

traffic_law_rag/
├── api.py              # 对外唯一入口：qa()（确保索引就绪 + 检索 + 生成）
├── __main__.py         # python -m traffic_law_rag "问题"：命令行版的一次 qa() 调用
├── server_demo.py      # 单文件 HTTP demo：/qa、/qa/stream（SSE）、/health（见第 8 节）
├── eval.py             # 离线检索评测：hit@k / MRR，域内外分开报
├── config.py           # 目录布局 + 模型端点 + Milvus + 检索参数（全部走环境变量）
├── contracts.py        # 层间数据契约（唯一真源）
├── docx_reader.py      # read     层：标准库 zipfile + ElementTree 直读 docx
├── law_parser.py       # parse    层：LawParser / LawLibrary / ParseStage
├── chunker.py          # chunk    层：LawChunker / ChunkStage
├── milvus_store.py     # 存储层：集合 schema + BM25 函数 + hybrid_search
├── indexer.py          # index    层：EmbeddingClient / Indexer（建集合、写索引快照）
├── query_rewriter.py   # rewrite  层：口语词法条用语对齐 + 法名线索
├── retriever.py        # retrieve 层：Milvus 双路召回 + 父块回灌
├── generator.py        # generate 层：强制引用式作答（同步 + 流式两条路径）
├── rag.py              # 管道级门面：LegalRAG（search / ask，不含建库）
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
# 1) 依赖（agent / api / dev 都是可选组，[all] 一次装齐）
.\.venv\Scripts\python.exe -m pip install -e ".[all]"

# 2) 起 Milvus（etcd + MinIO + Milvus，首启约 60~90 秒转 healthy）
#    只起 standalone —— 整栈 docker compose up -d 会连应用容器一起起，
#    和下面第 5 步的本地进程抢 8000 端口。想跑容器里的应用见 §8「Docker」。
docker compose up -d --wait standalone

# 3) 配置模型（没有 key 也能跑：集合会退化为纯 BM25，且不生成答案）
Copy-Item .env.example .env
#   然后填入 LLM_API_KEY / EMBED_API_KEY

# 4) 建库（重建 Milvus 集合；docx 的 sha1 未变则自动跳过解析）
python -m traffic_law_rag.pipeline build

# 5) 问答 / 检索 / 状态
python -m traffic_law_rag.pipeline ask "醉驾怎么处罚"
python -m traffic_law_rag.pipeline search "深圳 行人 在机动车道 罚款多少" --debug
python -m traffic_law_rag.pipeline status
```

`--debug` 会额外跑两次单通道检索（稠密、BM25 各一次），把每条法条是被哪一路捞到的、
各自排名多少都打出来，调检索时很有用。

也可以只用管道的一部分：

```python
from traffic_law_rag.pipeline import RagPipeline

pipe = RagPipeline()
pipe.build()                                   # 建库
answer = pipe.ask("醉驾怎么处罚")                # 检索 + 生成
result = pipe.search("智能网联汽车道路测试")       # 只检索，不花 LLM 的钱
print(answer.render())                         # 正文 + 参考文献 + 提示
```

## 4. 一键使用（一个接口）

对外只开 `qa()` 一个口子：不必先手动 `build`，也不必自己装配检索器。

```python
from traffic_law_rag import qa

qa("醉驾怎么处罚")                        # 确保索引就绪 → 混合检索 → 生成（返回 Answer）
qa("深圳 行人 在机动车道", mode="search")  # 只检索，不花 LLM 的钱（返回 RetrievalResult）
qa("醉驾怎么处罚", debug=True)            # 附每条被向量 / BM25 各排到第几
```

命令行等价于一次 `qa()` 调用：`python -m traffic_law_rag "醉驾怎么处罚"`，加 `--search --debug`
只看检索（不花 LLM 的钱，约 3 秒；再加 `--no-vector` 走纯 BM25 是毫秒级），
加 `--rebuild` 强制重建。`python examples/demo.py` 则把 4 个典型问题依次跑一遍，
打印「问题 → 命中条号 → 答案要点」，一条命令看完全貌。

索引只在 docx 的 sha1、本地产物、Milvus 集合行数三者不一致时才重建（避免无谓重跑
向量化花钱）；失败抛 `QaError`，消息本身就是一行中文提示，命令行入口会把它接住。

## 5. 检索设计（决定效果的五件事）

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

## 6. 离线评测

```powershell
python -m traffic_law_rag.eval --in-domain     # 只跑域内题，约 1 分钟
python -m traffic_law_rag.eval --no-vector     # 只走 BM25，做 A/B 对照
```

语料 `data/eval_corpus.json`（779 条 instruction/output）**不是评测集**，必须先筛：298 条的答案里
没有能定位到本库的条号（构造不出 gold），213 条的题面自己就写着「第X条」或《法名》
（答案泄漏，检索必然"命中"）。剩下 268 条还得**分域报**——其中 133 条是美国自动驾驶
事故叙述（Waymo / Zoox 在旧金山…），本库是中国交通法规，覆盖不了；混在一起算会把
hit@3 从 85% 拉到 45%，看起来像"检索很差"。

| 域内 135 题（法条问答） | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| 稠密 + BM25 | 75.6% | 85.2% | 88.1% | 0.805 |
| 纯 BM25 | 74.1% | 84.4% | 89.6% | 0.796 |

**稠密通道在这套题上几乎不赚**：+0.8pp hit@3，代价是慢 8 倍（全量 268 题实测：
混合 102.8s，纯 BM25 12.6s）。因为这些题的题面是从法条原文生成的，词面重合度高，
BM25 本就接近最优；稠密通道真正值钱的场景是口语提问
（「醉驾」→「醉酒驾驶」），而本语料里没有这类题。**所以这是"这套题测不出它的价值"，
不是"稠密通道没用"**——要验证后者，得另造一批口语题。

未命中的 16 条里抽查出至少 2 条是 gold 标错：问"道路两侧植物遮挡信号灯"，gold 给了
道交法第二十九条（道路规划建设），真正的答案是第二十八条，而检索返回的正是第二十八条。
gold 是从模型生成的 `output` 里解析出来的，不是人工标注，带噪声。

## 7. 设计取舍

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

## 8. 工程化：打包、测试、HTTP demo

### 打包

`pyproject.toml`（PEP 621），基础安装刻意保持精简，可选能力按需装：

| 安装 | 内容 |
|---|---|
| `pip install -e .` | `python-dotenv` + `openai` + `pymilvus`，够跑完整管道 |
| `pip install -e ".[api]"` | 加 `fastapi` + `uvicorn`，跑 HTTP demo |
| `pip install -e ".[dev]"` | 加 `pytest` + `ruff` |
| `pip install -e ".[all]"` | 全部 |

命令行入口：`tlr-qa "醉驾怎么处罚"`、`tlr-serve`。

### 测试

183 个测试，**全部离线可跑**（不连 Milvus、不发网络请求），`pytest -q` 约 8 秒：

```powershell
python -m pytest -q
python -m ruff check .
```

测的是**不变量和 golden 数字**，不是实现细节：4 部法规 / 380 条 / 619 块 /
均长 79 字 / 逐部条数 124·115·64·77 / `leftovers` 恒空 / **零孤立列举项**。
其中零孤立列举项这条尤其值得留着 —— 它是第 5 节那个「248/867 → 0」的头条成果，
此前只在 README 里写着，没有任何自动化保护，改切分规则很容易悄悄退化回去。

生成层用假 LLM 客户端驱动，因此「流式与非流式必须发同一份提示词」这类断言无需网络即可跑。

### HTTP demo

单文件，把 `qa()` 包成 FastAPI：

```powershell
uvicorn traffic_law_rag.server_demo:app --port 8000
# 或：tlr-serve --port 8000

curl localhost:8000/health
curl -X POST localhost:8000/qa -H 'Content-Type: application/json' -d '{"question":"醉驾怎么处罚"}'
curl -N -X POST localhost:8000/qa/stream -H 'Content-Type: application/json' -d '{"question":"醉驾怎么处罚"}'
```

| 端点 | 说明 |
|---|---|
| `GET /health` | 存活 + 库内规模 + **装配耗时**；未就绪返回 503 和原因（不崩进程） |
| `POST /qa` | `mode=ask` 检索+生成，`mode=search` 只检索不花钱 |
| `POST /qa/stream` | SSE：先推 `evidence`（依据清单），再逐块推 `delta`，末尾 `done` |

**这个 demo 真正想演示的不是「能起服务」，而是「装配只做一次」。**
`qa()` 每次调用都会重跑索引就绪检查、重读 `chunks.jsonl`、重连 Milvus ——
命令行里无所谓（进程活一次就退），放进 HTTP 服务就是每个请求都付这个钱。
所以检索器和生成器在 `lifespan` 里装配一次缓存在 `app.state`，请求只做「检索 + 生成」。
代价对比是显式暴露的：装配耗时在 `/health` 的 `boot_ms` 里，单请求耗时在每个响应的
`request_ms` 里，自己比。

流式那条路径**复用同一份 `SYSTEM_PROMPT` / `USER_TEMPLATE`**，不另写提示词 ——
否则同一个问题在 `/qa` 与 `/qa/stream` 下会给出不一样的答案，而这种偏差极难发现。
流式不重试（已吐出的 token 收不回来），首字节之前的失败才重试。

#### 冷启动：一次实测出来的性能问题

服务刚起来时，**第一个**检索请求要 ~1.9 秒，之后只要 ~0.3 秒。查下来的结论是：

| 嫌疑 | 实测 | 结论 |
|---|---|---|
| Milvus 侧预热（集合加载 / 段缓存） | 全新进程首次 `hybrid_search` 只用 **22.8ms** | 排除 |
| 我们的装配（读 jsonl + 连 Milvus） | `HybridRetriever.load()` 只要 **38.9ms** | 排除 |
| embedding 端点的首次建连 | 首次 `embed_one` **2951ms**，之后 ~160ms；换没见过的文本仍是 ~160ms（**不是缓存**） | **就是这个** |

也就是说，冷启动的代价**全是首次调 embedding 时的 TCP + TLS 握手**，而它被惰性地
算在了第一个用户请求头上 —— 恰好戳破这个 demo 主打的「装配只做一次」。
修法是在 `lifespan` 里多预热一次（`HybridRetriever.warm()`），把它从请求挪到启动：

```
修前：装配 0.56s                                      第一个请求 1942ms → 之后 297ms
修后：装配 2.03s（含「稠密通道预热 1620ms」）            第一个请求  293ms → 之后 175ms
```

预热失败不算错 —— 检索层本来就有「向量端点不可用就退回 BM25」的降级路径，
预热同理（`warm()` 吞掉异常返回 `None`，启动日志里如实写明）。命令行一次性调用**不**预热：
早晚都要付，挪一下没有意义。

边界：没有鉴权、没有限流、没有把 server 拆成包。端到端三个端点，
够跑通、够被 curl 验。

### Docker

应用镜像是**无状态**的（`Dockerfile`），数据全在有状态的那半边（Milvus）。
两者在同一个 `docker-compose.yml` 里，所以一条命令起来：

```powershell
docker compose up -d --build        # 起 Milvus + 应用（首次构建约 2 分钟）
docker compose ps                   # 四个容器都要 healthy
docker compose logs -f app          # 看装配日志
curl localhost:8000/health          # 宿主机直接访问
```

实测启动日志：

```
[serve] 装配完成（2.03s）：复用已有索引：4 部法规 / 380 条 / 380 父块，
        集合 619 行（稠密+BM25）｜docx、本地产物与集合三者一致｜Milvus 2.6.24
[serve] 稠密通道预热 1620ms
```

（那 1.6 秒的预热就是上面说的冷启动修复，见 §8「冷启动」。）

容器化踩到的四个坑，都写在对应文件里了：

| 坑 | 处理 |
|---|---|
| `server_demo` 默认绑 `127.0.0.1` | 容器里必须 `--host 0.0.0.0`，否则宿主机映射的端口连不上 |
| 容器里 `localhost` 指容器自己 | `MILVUS_URI` 覆盖成服务名 `http://standalone:19530`（`.env` 里那个是给宿主机裸跑用的） |
| Milvus `Up` ≠ 能接受连接（还差 30~90 秒） | `depends_on: condition: service_healthy` |
| 命名卷首次创建继承镜像里同路径目录的属主 | 镜像里先 `mkdir` 那两个派生目录再 `chown`，否则非 root 用户写不进去 |

`chunks/` 与 `index/` 挂在命名卷上，构建机上有就随镜像带进去 —— 首次启动直接
「复用已有索引」，免得在容器里重算一次 embedding（那要花钱）。索引本身仍在 Milvus 里。

**新 clone 也能起来**：那两个目录被 `.gitignore` 忽略，所以镜像里可能根本没有。
已实测这条路径 —— 清空后首次启动会自己重建，10.0 秒走完「检测到检索层产物缺失 →
chunk 619 个 → 向量化 → 建出 619 行稠密+BM25 集合」。其中 parse 一步因 `parsed/`
的 sha1 未变而跳过（`parsed/` 在版本管理里，新 clone 也有）；真删掉它才会从 docx 重解析。

> 验证这一次是**接桩端点**跑的：用一个返回确定性伪随机向量的本地 HTTP 服务替掉真实
> embedding（桩收到 63 批 / 620 条），并且用独立的 collection 名，全程零 API 费用、
> 真索引一行没动。要验证的既然是「重建路径能不能跑通」而不是召回效果，就没必要为此花钱。

密钥不进镜像：`.env` 被 `.dockerignore` 挡在构建上下文外，运行时由 compose 的
`env_file` 注入。已实测确认镜像里既无 `.env` 也无 `volumes/`（162M 的 Milvus 数据卷）。

## 9. 待办

- 评测集的 gold 噪声较大（从模型 output 解析），需人工校验一批或改为「从条文反向造题」
- 口语题评测集：域内题全是从法条原文生成的，测不出稠密通道的价值
- 生成侧指标：引用命中率、答案正确率（要接 LLM 评判）
- 多轮对话（`Question.history` 已预留）
- Agentic RAG：在管道之上加一层工具调用循环（`search_law` / `get_article` / `list_laws`
  等），让「深圳开车玩手机，罚多少、扣几分」这类跨法规多跳问题不必靠一次检索碰运气
