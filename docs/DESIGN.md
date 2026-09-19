# 交通法规问答 Agent · 设计文档

> 这是**深挖版**：完整管道分层、评测口径、每一处设计取舍和踩过的坑。
> 想先知道「这是什么、跑起来什么样、怎么启动」，看 [../README.md](../README.md)。
>
> 文中「第 N 节」指的都是本文档自己的章节。

面向驾驶员 / 驾校学员 / 交管客服的交通法规问答。**RAG 只是管道里的一个工具**，
交付物是完整管道：`docx → 结构层 → 检索层 → 索引 → 检索工具 → 生成`。

当前知识库：6 部法规，508 条，812 个检索块。

| 法规 | 版本 | 条数 |
|---|---|---|
| 中华人民共和国道路交通安全法 | 2021-04-29 | 124 |
| 中华人民共和国道路交通安全法实施条例 | 2017-10-07 | 115 |
| 中华人民共和国道路运输条例 | 2026-01-30 | 82 |
| 深圳经济特区道路交通安全违法行为处罚条例 | 2024-05-10 | 77 |
| 深圳经济特区智能网联汽车管理条例 | 2026-05-27 | 64 |
| 机动车交通事故责任强制保险条例 | 2019-03-02 | 46 |

**评测是三套题集，测的不是同一件事。** 它们对着同一套知识库，但走的是管道里的不同段
（208 那套触发时**根本不检索**），gold 长短与分母也各不相同，**数字不可跨表比较** ——
把哪两个数放一起比，是这个仓库最容易犯的错：

| 题集 | 题数 | gold 是什么 | 测什么 | 命令 |
|---|---|---|---|---|
| 既有域内题 | 135 | 单条法条 | **检索**：hit@k / MRR | `python -m traffic_law_rag.eval` |
| 泄漏桶（题面自己写了条号） | 208 | 单条法条 | **规则取条**能否唯一定位 | `python -m traffic_law_rag.eval --reference` |
| 跨法多跳 | 100 | **两部法规各一条** | agent 与 rag 的**跨法**对照 | `python -m traffic_law_rag.eval.multihop --check`（护栏，全离线）<br>`python -m traffic_law_rag.eval.multihop --compare`（对照，花钱） |

前两套出自同一个文件 `data/eval_corpus.json`（475 条），按「有无 gold」「题面是否泄漏」切成三份：
135 条留下当评测集，208 条因题面泄漏被剔除、但正好改当探针集，**这两份不重叠**；
另有 132 条构造不出 gold，直接丢弃、不参与任何一臂。第三套是独立的生成产物
`data/eval_multihop.json`，因为跨法多跳在现有语料里**挖不出来**（`refs` 的 42 条引用边
100% 同法），只能从条文反向生成，gold 由模型给出。各自的口径与局限见 §6 与 §9。

---

## 技术栈

**运行时与基础设施**

| 项 | 选型 |
|---|---|
| 语言 | Python 3.11（`requires-python >=3.10`） |
| 向量库 | Milvus **2.6** standalone —— etcd + MinIO + Milvus 三容器，docker compose 起 |
| 中文分词 | Milvus **服务端**内置 jieba analyzer（`MILVUS_ANALYZER`），**本地不装 jieba** |
| BM25 | Milvus 服务端 `FunctionType.BM25` + `RRFRanker` 融合，**本地不装 rank_bm25** |
| 服务层 | FastAPI + uvicorn，单文件 [server_demo.py](../traffic_law_rag/server_demo.py)，`/qa/stream` 走 SSE |
| 部署 | 单镜像 Dockerfile + compose，容器内跑 uvicorn（见第 8 节） |

**模型**（外部 HTTP API，全部走 OpenAI 兼容协议）

| 用途 | 模型 | 端点 |
|---|---|---|
| 生成 / agent 规划 | `qwen3-max` | 阿里百炼 DashScope compatible-mode |
| 向量化 | `text-embedding-v4`（1024 维） | 同上 |

换生成模型只改 `.env`，代码零改动。但**向量模型不能随手换**：集合是按 `EMBED_MODEL` 建的，
换了要全库重建、且既有基线数字全部作废 —— 所以 `EMBED_API_KEY` 必须显式填，不能留空吃回退
（理由写在 `.env.example` 里，踩过一次）。

**依赖**（分四组，`pip install -e ".[all]"` 一次装齐，明细见第 8 节）

| 组 | 内容 |
|---|---|
| 基础 | `openai` · `pymilvus` · `python-dotenv` |
| `[agent]` | `langgraph` · `langchain-core` · `langchain-openai` |
| `[api]` | `fastapi` · `uvicorn[standard]` |
| `[dev]` | `pytest` · `ruff` |

基础组只有三个包，这是**刻意**的：解析 docx 走标准库，分词 / BM25 / 分页全推给 Milvus 服务端，
本地不需要任何 NLP 或检索库。

**几条定调的技术选择**（取舍理由见第 7 节）：

- **不用 LangChain 搭 RAG。** 七层管道是手写的，每层一个类、输入输出固定。LangGraph 只用在
  agent 那一层的状态机上，**不用它的 LLM 抽象层** —— [agent/graph.py](../traffic_law_rag/agent/graph.py)
  直接走 `openai` SDK，换来的是 state 里的 `messages` 就是 OpenAI 线上格式的 `list[dict]`，
  `json.dumps` 直接可过，不需要 `add_messages` 那层会改变消息形态的转换。
  代码里**一个 `langchain` 的 import 都没有**（只有 `from langgraph.graph import StateGraph`）；
  `[agent]` 组里的 `langchain-core` / `langchain-openai` 是 langgraph 拖进来的传递依赖，
  单独钉住是为了挡住它们把 `openai` 从 3.x 降级 —— 详见 `pyproject.toml` 的注释。
- **docx 解析不引 python-docx。** 标准库 `zipfile` + `ElementTree` 直读，精度最高、依赖最轻。
- **契约单一真源。** [contracts.py](../traffic_law_rag/contracts.py) 管住所有磁盘 JSON 格式，
  改格式只动这一个文件。
- **边界靠 AST 钉住，不靠自觉。** `HybridRetriever` 只有 `qa/rag.py` 准 import、
  `agent/tools.py` 不准依赖 langgraph、`import traffic_law_rag` 不准拉 langgraph —— 每条都有测试守着。

## 1. 管道分层（每层一个类，输入输出固定）

| 层 | 类 | 所在模块 | 输入 | 输出 |
|----|----|---------|------|------|
| read     | `DocxReader`       | `kb/docx_reader.py`     | docx 路径 `Path`               | `list[Paragraph]` |
| parse    | `ParseStage`       | `kb/law_parser.py`      | docx 目录                      | `list[LawDocument]` |
| chunk    | `ChunkStage`       | `kb/chunker.py`         | `list[LawDocument]`            | `ChunkSet`（父块 + 子块） |
| index    | `Indexer`          | `kb/indexer.py`         | `ChunkSet`                     | `IndexStats` |
| rewrite  | `QueryRewriter`    | `qa/query_rewriter.py`  | `Query`                        | `RewrittenQuery` |
| retrieve | `HybridRetriever`  | `qa/retriever.py`〔私有〕 | `Query`                        | `RetrievalResult` |
| generate | `AnswerGenerator`  | `qa/generator.py`       | `Question` + `RetrievalResult` | `Answer` |
| 编排     | `RagPipeline`      | `pipeline.py`           | 以上全部                       | `PipelineReport` / `Answer` |

前四层只在建库时跑（`kb/`），后三层每次提问都跑（`qa/`）—— 子包就是按这条时间线切的。
`retrieve` 那层标了〔私有〕：只有 `qa/rag.py` 准认识它，上层一律走门面（见第 7 节）。
索引的存储侧（集合 schema / BM25 函数 / `hybrid_search`）在 `kb/milvus_store.py`，
它不是一层，是 `index` 与 `retrieve` 共用的底座。

层间契约全部定义在 `traffic_law_rag/contracts.py`（含各对象的 JSON 读写），
**磁盘格式变更只需改这一个文件**。查看层表：`python -m traffic_law_rag.pipeline layers`。

七层之上只开一个口子：`from traffic_law_rag import qa`，一次调用拿到结果（见第 4 节）。

## 2. 目录结构

仓库根目录只有配置、代码和数据目录，**没有散件** —— 跑出来的东西各归其位。

```
docker-compose.yml      # Milvus Standalone（etcd + MinIO + Milvus）+ 无状态的应用容器
Dockerfile              # 应用镜像；.dockerignore 挡住密钥与 162M 的 volumes/
pyproject.toml          # 打包与依赖，pip install -e ".[all]" 一次装齐

data/
├── eval_corpus.json    # 评测语料（475 条 instruction/output，用前必须先筛，见第 6 节）
├── eval_multihop.json  # 跨法多跳题集（100 道，生成产物，见第 9 节）
└── traces/             # --trace / --compare 落盘的轨迹（gitignore：跑出来的，可重跑）

examples/
└── demo.py             # 4 个典型问题跑一遍：问题 → 命中条号 → 答案要点

tests/                  # 离线测试：不连 Milvus、不发网络请求（见第 8 节）

volumes/                # Milvus 数据卷（docker compose 生成，162M，gitignore）

traffic_law_rag/
├── api.py                 # 对外唯一入口：qa()（确保索引就绪 + 检索 + 生成）
├── __main__.py            # python -m traffic_law_rag "问题"：命令行版的一次 qa() 调用
├── server_demo.py         # 单文件 HTTP demo：/qa、/qa/stream（SSE）、/health（见第 8 节）
├── pipeline.py            # 编排 + CLI
├── config.py              # 目录布局 + 模型端点 + Milvus + 检索参数（全部走环境变量）
├── contracts.py           # 层间数据契约（唯一真源）
│
├── kb/                    # 建库：docx 进，Milvus 集合出。只在 pipeline build 时跑
│   ├── docx_reader.py     #   read    层：标准库 zipfile + ElementTree 直读 docx
│   ├── law_parser.py      #   parse   层：LawParser / LawLibrary / ParseStage
│   ├── chunker.py         #   chunk   层：LawChunker / ChunkStage
│   ├── milvus_store.py    #   存储    层：集合 schema + BM25 函数 + hybrid_search
│   └── indexer.py         #   index   层：EmbeddingClient / Indexer（建集合、写索引快照）
│
├── qa/                    # 问答：一个问题进，一段带引用的答案出
│   ├── query_rewriter.py  #   rewrite 层：口语词法条用语对齐 + 法名线索
│   ├── retriever.py       #   retrieve 层：Milvus 双路召回 + 父块回灌
│   ├── generator.py       #   generate 层：强制引用式作答（同步 + 流式两条路径）
│   └── rag.py             #   管道级门面：LegalRAG（search / ask / expand / stats，不含建库）
│
├── agent/                 # Agentic RAG（需 langgraph，见第 9 节）
│   ├── graph.py           #   图、节点、AgentRunner、ToolCallingLLM
│   ├── tools.py           #   两个工具：search_law / get_article（**不依赖** langgraph）
│   └── __main__.py        #   python -m traffic_law_rag.agent "问题"
│
└── eval/                  # 离线评测：题集 A/B 见第 6 节，题集 C 见第 9 节
    ├── harness.py         #   题集 A/B：hit@k / MRR，外加「规则取条」探针臂
    ├── multihop.py        #   题集 C：跨法多跳的生成器 / 护栏 / --check / --compare
    └── __main__.py        #   python -m traffic_law_rag.eval [--reference]

法规知识库/
├── pdf/                # 上游原始素材（下载件）。**不属于管道** —— 管道只读 docx，
│                       #   要入库得先转成 docx 放进 docx/。放这里只是不让散件堆到根目录。
├── docx/               # 唯一真源，永不改写
├── text/*.md           # 人读层：法条 Markdown（人工比对用）
├── parsed/*.json       # 结构层：法 → 章 → 节 → 条 + manifest.json（sha1 增量门控）
├── chunks/             # 检索层：parents.jsonl（条）+ chunks.jsonl（款）〔gitignore，build 重建〕
└── index/              # 索引快照：index_meta.json（向量数据在 Milvus 里）〔gitignore，build 重建〕
```

进版本管理的是 **docx + text + parsed + 两份题集**；`chunks/`、`index/`、`traces/`、`pdf/`、
`volumes/` 五处被 gitignore —— 前两个能重建，后三个是跑出来的或可重新下载的。

`traffic_law_rag/` 下四个子包的 `__init__.py` **一律只写文档字符串、不做 re-export**。
理由有两条，都不是洁癖：一是 `kb/` 一导出就会让「只想用切块器」的调用方连带拖上 pymilvus；
二是 `agent/` 一导出就会把 langgraph 拉进 `import traffic_law_rag`（有一条测试专门钉着这件事）。
**包路径即语义**，要用哪个就 import 哪个。

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
#   然后填入 LLM_API_KEY / EMBED_API_KEY —— 两个都要显式填，别让 EMBED_API_KEY 留空。
#   留空会回退吃 LLM_API_KEY，于是「换个生成模型」会把向量侧一起打断，
#   而集合是按 EMBED_MODEL 建的：换向量模型 = 全库重建 + 既有基线全部作废。

# 4) 建库（重建 Milvus 集合；docx 的 sha1 未变则自动跳过解析）
python -m traffic_law_rag.pipeline build

# 5) 问答 / 检索 / 状态
python -m traffic_law_rag "醉驾怎么处罚"
python -m traffic_law_rag "深圳 行人 在机动车道 罚款多少" --search --debug
python -m traffic_law_rag.pipeline status
```

`--debug` 会额外跑两次单通道检索（稠密、BM25 各一次），把每条法条是被哪一路捞到的、
各自排名多少都打出来，调检索时很有用。

> **`pipeline` 子命令只管建库和查状态，不再有 `ask` / `search`。** 这两个曾经存在，
> 是 `qa()` 之前的遗留入口：能力是 `python -m traffic_law_rag` 的真子集（没有
> `--search` / `--debug` / `--rebuild`），而且**少了 `ensure_ready()`** —— 索引缺失或过期时，
> 它们不会自动重建，而是直接撞到 pymilvus 的堆栈上。统一入口做出来之后它们就是两条
> 行为不一致的路，故删。现在问一句只有一条命令。

> **`python -m traffic_law_rag.pipeline --help` 现在是安全的，但它曾经不是。**
> 命令推断写的是 `args[0] if args and not args[0].startswith("--") else "build"`，
> `--help` 以 `--` 开头 → 落进 `else "build"` → **真的跑了一次完整重建**（42 秒、812 行
> 重新 embedding）。想查用法，付出的是一次 embedding 的钱。现在 `--help` / `-h` / `help`
> 在推断之前就被拦下，直接打印本文档开头那段用法；`test_pipeline_cli.py` 用哨兵钉死了
> 「帮助路径绝不构造 `RagPipeline`」。教训不在这个分支本身，而在**「没有子命令就跑 build」
> 这个默认值**：任何以 `-` 开头的未知参数都会被它静默地当成"建库"。

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
   未处理时这类孤立子块占 1164 个中的 352 个（30%），合并后降到 812 块、均值 78.8 字。
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
python -m traffic_law_rag.eval                 # 全量 135 题，约 1 分钟
python -m traffic_law_rag.eval --no-vector     # 只走 BM25，做 A/B 对照
```

语料 `data/eval_corpus.json`（475 条 instruction/output）**不是评测集**，必须先筛：
132 条的答案里没有能定位到本库的条号（构造不出 gold），208 条的题面自己就写着
「第X条」或《法名》（答案泄漏，检索必然"命中"）。**剩下的 135 条就是评测集。**

> 这 135 题的 gold 都是**单条或同法内几条**，测不到"答案横跨两部法规"的情形。
> 第三套题集 `data/eval_multihop.json`（100 道跨法多跳）在 §9，它另有自己的护栏与命令。

> 语料原为 779 条，2026-09 删掉 304 条：296 条是美国自动驾驶事故叙述
> （Waymo / Zoox 在旧金山…），8 条是深圳条例的英文题面。删它们不是为了让数字好看 ——
> 那批美国题的 gold 是**硬凑的**（本库是中国交通法规，检索返回深圳条例反而是对的），
> 域外 hit@3 只有 **4.5%**，混进合计会把 hit@3 从 85.2% 拉低到 45.1%，
> 看起来像"检索很差"，其实测的是不该测的东西。那 8 条英文题则是跨语言检索，
> 也不在这个评测的射程内。清完之后评测集**全域内**，`--in-domain` 开关随之删掉。
> 留下的 135 题**一道没动**——删的 304 条与它们没有交集，所以下表这些数字与删除前
> **逐位相同**。这正是本次改动最硬的回归断言：分母小了 134 条，指标一位都不许漂。

| 135 题（法条问答） | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| 稠密 + BM25 | 75.6% | 85.2% | 88.1% | 0.805 |
| 纯 BM25 | 74.1% | 84.4% | 89.6% | 0.796 |

**稠密通道在这套题上几乎不赚**：+0.8pp hit@3，代价是慢 8 倍（135 题实测：
混合 50.4s，纯 BM25 6.1s）。因为这些题的题面是从法条原文生成的，词面重合度高，
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
- **门面只开一个：`LegalRAG` 是上层认识检索的唯一途径**。`HybridRetriever` 的
  `store` / `chunks` / `rewriter` 都是私有属性，`qa/rag.py` 是包内唯一 import 它的模块
  （由 `tests/test_rag_boundary.py` 的 AST 断言钉住）。这不是洁癖：重构前上层有六处
  直接伸手取零件，于是**同一件事有三个真源** —— `describe()` 的「稠密通道状态」
  在 `qa/rag.py` 和 `agent/graph.py` 里各写一遍同样的 try/except；`agent/tools.py` 的
  `expand_query` 靠 `getattr(retriever, "rewriter")` 摸进改写器，改写器一改名它就返回原话，
  **不报错、不失败，只是摘要窗口悄悄退回修好之前的位置**（那正是上一轮刚修掉的缺陷）。
  现在这两件事分别是 `LegalRAG.stats()` 与 `LegalRAG.expand()`：一个真源，
  改名的代价是一次 `AttributeError` 而不是一次静默劣化。
  换掉检索实现时，要动的只有 `qa/rag.py` 一个文件。

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

322 个测试，**全部离线可跑**（不连 Milvus、不发网络请求），`pytest -q` 约 9 秒：

```powershell
python -m pytest -q
python -m ruff check .
```

测的是**不变量和 golden 数字**，不是实现细节：6 部法规 / 508 条 / 812 块 /
均长 78.8 字 / 逐部条数 124·115·82·77·64·46 / `leftovers` 恒空 / **零孤立列举项**。
其中零孤立列举项这条尤其值得留着 —— 它是第 5 节那个「352/1164 → 0」的头条成果，
此前只在文档里写着，没有任何自动化保护，改切分规则很容易悄悄退化回去。

生成层用假 LLM 客户端驱动，因此「流式与非流式必须发同一份提示词」这类断言无需网络即可跑。

`test_rag_boundary.py`（6 条）是**结构性**的一层：它用 AST 遍历整个包，断言
`HybridRetriever` 只被 `qa/rag.py` import、也只被 `qa/rag.py` 在代码里提到。这条约束原本只写在
`qa/rag.py` 的自述里 —— 而当时上层有六处直接 import 检索器、伸手取 `.store` / `.chunks` /
`.rewriter`，自述是假的。用 AST 而不是正则是必须的：`pipeline.py` 有一个运行时常量元组
`("retrieve ", "HybridRetriever", ...)`，几个模块的自述里也有这个词，正则会把它们全算成
引用。同组还有一条**属性**断言：`store` / `chunks` / `rewriter` 在真检索器上必须是私有的，
伸手进来当场 `AttributeError` —— 这才是真正把门关上的那一手，AST 只管 import、管不到穿透。

它遍历的是 `rglob("*.py")` 而**不是** `glob("*.py")`，这一点值得单独记一笔：子包化之前它
只扫顶层，搬进 `kb/` / `qa/` 之后会**静默地**少扫一大半 —— 测试照样全绿，只是边界没人守了。
所以它除了用相对路径做身份（否则六个 `__init__.py` 会互相覆盖），还有一条断言专门盯着
「四个子包各至少扫到一个模块」，单靠总数是看不出一整个子包漏掉的。

`test_rag.py`（12 条）补的是门面自己的行为。**这个文件之前不存在** —— `LegalRAG` 是上层
唯一的检索入口，却一条测试都没有，于是「把参数透错了」这类错误只能靠人眼盯。最典型的是
`law_filter`：它的默认值必须是 `()` 不能是 `None`（`MilvusStore.law_filter_expr` 直接迭代
入参），写错就是一个必炸的 `TypeError`，而没有任何一条既有测试会红。

Agent 那层另有两组：`test_agent_tools.py`（工具层，47 条，**不需要 langgraph**）与
`test_agent.py`（图与分支，33 条，`importorskip("langgraph")`）。图那组的头条是
**提示词逐字一致**：Agent 路径与直接调 `generate(question, merge_retrievals(同一批 logs))`
发给假 client 的 `messages` 必须完全相等 —— 这条一旦断了，Agent 就不再是「管道之上的一层」，
而变成了一条悄悄改了提示词的平行实现。

多跳题集那层是 `test_multihop.py`（32 条）。它的**负向用例比正向重要** —— 题集的 gold 是模型
给的，护栏是它唯一的可信度来源，所以必须证明「gold 全在同一部法」「题面写了第X条」「题面写了
法名简称」「gold 是立法依据条」都会被拒。其中一条专门钉住一个**差点写错的判据**：纯适用范围条
的识别不能按「含『适用本条例』」来判 —— 那会误伤《交强险条例》第二条，而它是全库被当作 gold
最多的一条（21 次，全部正当）。

`test_pipeline_cli.py`（4 条）只干一件事：把 `RagPipeline` 换成哨兵，断言 `--help` 不会构造管道。
这条测试的存在本身就是一个教训 —— 见 §3 命令行那段引用块。

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

实测启动日志（计数为当前语料，秒数随机器冷热浮动）：

```
[serve] 装配完成（1.36s）：复用已有索引：6 部法规 / 508 条 / 508 父块，
        集合 812 行（稠密+BM25）｜docx、本地产物与集合三者一致｜Milvus 2.6.24
[serve] 稠密通道预热 1620ms
```

（第一行是 `api.ensure_ready().describe()` 的实际输出；预热那一行是此前跑容器时测的，
不随语料变化。那 1.6 秒的预热就是上面说的冷启动修复，见 §8「冷启动」。）

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

> 这段是 **4 部法时期**（380 条 / 619 块）实测的，日志里的 619 属于那次运行。
> 现在的语料是 6 部 / 508 条 / 812 块，同样这条路径会重建出 **812 行**；
> 秒数没重测，不替它编数。

> 验证这一次是**接桩端点**跑的：用一个返回确定性伪随机向量的本地 HTTP 服务替掉真实
> embedding（桩收到 63 批 / 620 条），并且用独立的 collection 名，全程零 API 费用、
> 真索引一行没动。要验证的既然是「重建路径能不能跑通」而不是召回效果，就没必要为此花钱。

密钥不进镜像：`.env` 被 `.dockerignore` 挡在构建上下文外，运行时由 compose 的
`env_file` 注入。已实测确认镜像里既无 `.env` 也无 `volumes/`（162M 的 Milvus 数据卷）。

## 9. Agentic RAG（阶段二）

**不是替代，是在管道之上加的一层。** `qa(mode="ask")` 仍是默认路径且行为逐字节不变；
Agent 是 `python -m traffic_law_rag.agent` 这条独立入口，装 `[agent]` 可选组才有。
`import traffic_law_rag` 永远不触发 langgraph 导入（有测试钉住）。

### 图

```
START → classify ─(条文定位)→ lookup_plan ─┐
             └──(法规检索)→ agent ────────┤─(有 tool_calls)→ tools → reflect
                                          │                          │
                            (无 tool_calls)┘              (不够且补得上且有预算)┘
                                          └──→ finalize ←──(够了 / 补不上 / 预算尽)
                                                 │
                                                END
```

**意图识别在入口，纯规则、零 LLM 调用。** 正则抽条号 → 用 `(law_id, 条号)` 索引校验
唯一可解析 → 条文定位，否则法规检索。校验放在这里而不是丢给工具报错：解析不到就根本不走这条路，
不浪费一轮。**条号跨多部法规（6 部法各有「第一条」）即判为法规检索** —— 宁可走通用路径，不猜法规。
实测 124 个不同条号里**只有 9 个**只属于一部法规，其余 115 个都跨法 —— 这个判据几乎是常态而非例外。

回边指向 `agent` 而不是 `classify`：意图只在入口定一次，第二轮再分类一遍纯属浪费，
还可能分到不同意图导致循环抖动。

**`reflect` 回边的条件是三个「与」**：证据不够、**缺口补得上**、还有预算。

第二条是实测补上的。审核原先只答「证据够不够回答问题」，路由却把它当成「要不要再检索一次」
—— 两者并不等价。用户问到**库外**的东西时（《治安管理处罚法》的责任、商业保险的合同约定、
紧急避险的免责），审核诚实地答「不够」，循环就去再检一次，可那个缺口再检一百次也补不上。
100 题那次审核一共跑了 **207 轮**，其中 **162 轮**是真的调了模型（另 45 轮预算已尽、reflect
短路，压根没问）—— 白跑就烧在这 162 轮里。

代价是提示词里必须带上**库的法规清单**（从 `parents` 现取，不写死一份）—— 否则审核无从判断
「库内」的边界在哪。`retrievable` 字段缺席时默认 `True`，即行为与加这个字段之前逐位相同：
默认成 `False` 会在模型不输出它时静默改掉循环，那不是「省了一轮」，是换了套逻辑。

> **这一条目前只过了离线测试，没有跑过真实模型。** 测试证明的是「管线接对了」
> （路由真值表、字段缺席时的默认、提示词里确实带了法名清单），**不是**「模型会老实输出
> `retrievable`」。**收益有多大同样没人量过** —— 上面那两个数（207 / 162）只是审核轮次的
> 规模，不是「省下了多少」。要坐实它得重跑一次 100 题对照。

**`lookup_plan` 刻意不直接取法条**，只产出一条带 `tool_calls` 的助手消息。这样 `tools` 节点、
`reflect` 的视角、`search_log` 的形状、路由判据四件事全部零改动。

### 两个工具

| 工具 | 用途 |
|---|---|
| `search_law` | 语义检索。不知道确切条号时用 |
| `get_article` | 按条号精确取原文。**已经知道要查哪一条时用它，比检索更准** —— 检索只保证「相关」，不保证「就是那条」 |

第二个工具的必要性是现有输出证明的：`search_law` 每次返回的 `refs`（「相关条：第九十九条」）
就是模型在说「我想看这一条」，而此前**没有任何工具能去取它**。现在规划轮能跟随 `refs` 取条 ——
《深圳经济特区智能网联汽车管理条例》第五十六条「违反本条例第十三条的规定，擅自开展道路测试
或者示范应用的……处十万元以上五十万元以下罚款」就指向被违反的那条定义（第十三条），先检索到
罚则、再按 `refs` 取回第十三条，这是一条真多跳。

> 写这条时必须写全法名：库里**有两部深圳条例**，`深圳经济特区道路交通**安全违法行为处罚**条例`
> 的第五十六条 `refs` 是空的，它讲的是「办理业务时应提供真实有效的联系方式」。只写「深圳条例
> 第五十六条」会让人对到错的那一部上去。

> **但 `refs` 的覆盖率只有 27/508（5.3%）** —— 508 条里只有 27 条写了「相关条」引用。
> 更要紧的是分布：实测这 42 条引用边**全部落在同一部法规内部，跨法边 0 条**。跨法的引用
> 只有 9 条，且一律是概括性转致（「依照《道路交通安全法》的规定处十五日以下拘留」）——
> **只点名法规、不点名条号**。
>
> 所以「跟随 refs 多跳」这条故事线是**能力有、触发面窄、而且只在法内**。如实记一笔。
> 「跨**法规**多跳」因此没有机械 gold 源，只能从条文反向生成 —— 见下面第 9 节。

### 评测：两条臂，测的不是同一件事

构造评测题时会**剔除**题面含条号或法名的题（答案泄漏，检索必然命中），于是这 135 题里
含「第…条」的是 **0 道** —— **hit@k 这套指标在结构上永远衡量不到规则取条这条新路径**。
被剔除的那 208 条反而是一份现成的、已标注的探针集：203 条题面写明了条号，其中 **202 条
那个条号就是 gold**（202/203 = **99.5%**，唯一例外是下面那道题集自相矛盾的题）。
别再把这个数写成 97.1% —— 那是 202/**208**，分母是整桶、不是"写明了条号"的那 203 条，
是下一行"基线 hit@1"的口径，两者不是同一件事。

```powershell
python -m traffic_law_rag.eval --reference               # 探针臂全量
python -m traffic_law_rag.eval --reference --no-compare  # 纯离线，不连 Milvus
```

| 臂 | 题集 | 结果 |
|---|---|---|
| 既有（**未变动**） | 135 | hit@1 75.6% / hit@3 85.2% / hit@6 88.1% / MRR 0.805 |
| 规则取条 | 泄漏桶 208 | 触发 200/208（96.2%），其中就是 gold 199（**精确率 99.5%**）；回退 8 |
| 基线检索（同一批 208） | 泄漏桶 208 | hit@1 202/208（97.1%） hit@3 206/208（99.0%） |

**结论：没有命中率增益 —— 这一臂比基线还少 3 题。** 不粉饰。两个数不是同一件事：
规则回退的 8 题里，基线有 7 题第 1 名就是 gold；反过来规则也独得 4 题。
合起来「触发就用规则、回退就走检索」的上界 **206/208（99.0%）**。

> 口径提醒：206 是**第 1 名**口径（规则答对 ∪ 基线第 1 名是 gold）。换成**基线 top-6 内**
> 这个更宽的口径，基线能捞到 206 题，与规则答对的 199 题合起来是 **208/208** ——
> 这批题里没有一条落在"规则也错、检索也够不着"的角落里。两个数都对，别混着引用。

（这两个上界都只对**泄漏桶**成立：题面自己把条号或法名写出来了，本身就是最好定位的一批。
它证明的是"规则取条接得住它该接的那部分"，不是"检索问题已经解决"。）

那它赚在哪：

1. **成本**：触发的 200 题 **0 次检索调用**（无 embedding、无 BM25）、**0 轮 LLM 规划**。基线每条都要走一遍。这一臂完全离线，208 题耗时 < 0.1s。
2. **确定性**：触发时给的是唯一一条法条原文，不是一份排名 —— 精确率 199/200。
3. **判据从严**：回退的题交给检索，**不是丢了**。合起来才是上线形态。

8 道回退的构成（全部是**正确的**回退，不是失手）：
- 5 道题面只点名了法规、没给条号（`在《深圳经济特区道路交通安全违法行为处罚条例》中…`），
  定位器拿不到条号，本来就不该猜；
- 3 道给了条号但**跨法歧义**：`第六十一条` 在 5 部法里都有、`第九十四条` 在 2 部法里都有，
  题面又没点名问的是哪一部，配错比配不上糟，退回检索。

前两批里各有重复问法（同一道题在语料里出现两次），所以是 5 + 3 行、去重后 5 个问法。

**唯一那道错取是题集自身的矛盾**：题面写「根据《道路交通安全法》第九十五条…」，
gold 标的是第九十六条 —— 语料的 gold 是从模型生成的 `output` 里解析出来的，带噪声。
定位逻辑没有错，它忠实取回了题面点的那一条。

### 跨法多跳：100 道新题，gold 是模型给的

上面两条臂的 gold 都是**单条或同法内几条**，没有任何一题要求答案横跨两部法规。
补这一臂时先撞上一件必须先说清的事：**跨法多跳没有机械 gold 源**。

- 全库 27 条带 `refs`、共 42 条引用边，**100% 同法**，跨法边 **0 条**；
- 跨法的引用只点名**法规**、不点名条号，形态一律是概括性转致
  （「依照《中华人民共和国道路交通安全法》的规定处十五日以下拘留」）——
  它给的是法规级关系，仍然产不出条级 gold；
- 现语料里 475 条题中 gold 跨 ≥2 条的只有 10 条，其中两条 gold 之间真有 `refs` 边的只有 **1 条**。

所以这一臂的题只能**从条文反向生成**，`gold` 是模型给的断言，不是可机械验证的边 ——
这和 135 题集是**同一个噪声来源**。护栏能拦住的只有机械可判的那部分：

| 编号 | 护栏 |
|---|---|
| G1 | 题面不含条号、不含《法名》（含**不带书名号的法名简称**，实测踩过） |
| G2 | 每条 gold 都能解析到**真实存在**的条文 |
| G3 | gold **必须跨 ≥2 部法规** —— 硬闸，否则这道题不算跨法 |
| G4 | 题面去重（归一化后） |
| G5 | 立法依据条（「根据《X》，制定本条例」）不能当 gold —— 它不含任何实质规则 |

```powershell
python -m traffic_law_rag.eval.multihop --check                 # 全离线复跑护栏
python -m traffic_law_rag.eval.multihop --generate --target 100 # 花钱，产物进版本管理
python -m traffic_law_rag.eval.multihop --compare --limit 100   # agent/rag 对照（= --trace + 汇总）
```

实测产出：**尝试 101 次 → 采用 100 道（失败 1 次）**，每题 gold 恰好跨 2 部法，共 200 条。

| 法规 | 被多少道题的 gold 需要 |
|---|---|
| 中华人民共和国道路交通安全法 | 68 |
| 机动车交通事故责任强制保险条例 | **51** |
| 中华人民共和国道路交通安全法实施条例 | 23 |
| 中华人民共和国道路运输条例 | 21 |
| 深圳经济特区道路交通安全违法行为处罚条例 | 20 |
| 深圳经济特区智能网联汽车管理条例 | 17 |

**交强险那 51 道是要害**：它 `refs` 为 0，只能靠生成触及，而它正是"新入库的法规能不能被检索到"
这个问题的主角。题面与 gold 原文的最长公共片段中位 **3 字**、最长 9 字 —— 说明题面是在提问，
不是把条文抄进题目里（G1 拦不住这种，只能靠这个数看见）。

**两条必须一起读的局限：**

1. **「另一条真的必要吗」无法机械验证。** 基线 top-k 里 gold 的位置只作**诊断**记录，
   **绝不据此筛题** —— 那样等于把题集调成"rag 必输"，对比就失去意义。
2. **题集瑕疵一律只报数、不拦截。** `--check` 会把两类可疑但**有正当反例**的情况摆出来
   （同一段代码，两行数）：一是 **37/100 道题的 gold 引了深圳经济特区法规，其中 15 道题面
   没交代"在深圳"** —— 深圳两条条例只及于特区，不点明地点的 gold 是可争议的，可相当一部分
   处罚只有深圳条例写了，拒掉就等于把"新法能不能被检索到"一起拒掉；二是 **2/200 条 gold
   落在纯适用范围条上**（「……都应当遵守本法。」这类，只有管辖、没有规则），而同为第二条的
   《交强险条例》第二条是**投保义务**条款，按关键字拦会把它连同 21 条正当 gold 一起杀光。

### 已知限制

- 条号跨法歧义时**一律回退**，即使题面其实给足了线索
- `refs` 覆盖率 27/508，且 42 条引用边**全在法内**，跨法多跳无机械 gold 源
- 意图只有两类，且**域外不做意图**：运行时的域外靠 `tool_choice="auto"`（模型自己不调工具）
  + finalize 兜底。没加 LLM 分类节点 —— 那会引入一个新的错误源，却换不来可测量的收益。
  **这是运行时设计，与评测集删掉域外题无关**：线上用户问什么都可能，这里刻意不拦；
  评测集则反过来，只留本库真能覆盖的题 —— 两件事别混为一谈
- 不加第三个工具（`list_laws`）：它的能力已由 `resolve_law_id` 失败时回吐法名清单白捡
- 不做多轮对话、不做流式 agent

### 一个值得记的坑

`recursion_limit` 按 **superstep** 计，不是按轮。实测各条路能跑完的最小值（二分出来的，不是推出来的）：
检索路 `3*max_steps + 3`，定位路恒为 6。**加入意图识别时旧值恰好等于新图的最小值**，
一个 superstep 都插不进去 —— 图变了就得重测最小值，别信推导。现在是 `3*max_steps + 6`。

## 10. 待办

- 评测集的 gold 噪声较大（从模型 output 解析），需人工校验一批或改为「从条文反向造题」
  —— **跨法多跳那 100 道已经改用了反向造题**（见 §9），但换掉的只是"从模型 output 里解析 gold"
  这一层噪声，"模型断言这条真的必要"这一层还在。135 题集本身**未改动**，这条待办对它仍然成立
- 口语题评测集：这 135 题全是从法条原文生成的，测不出稠密通道的价值
- 生成侧指标：引用命中率、答案正确率（要接 LLM 评判）
- 多轮对话（`Question.history` 已预留）
- Agentic RAG 的后续：`refs` 覆盖面上不去的话，多跳这条线就只是「能跑通」
