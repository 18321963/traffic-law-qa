# 交通法规问答 Agent

用大白话问交通法规问题，答案里每句结论都挂着法条出处 —— `[依据N]` 能翻回原文；法条里没有的，它会说「库中未收录」并指出缺哪类规定，而不是编一条出来。

6 部法规 / 508 条 / 812 个检索块，从 docx 一路建到可问答：解析 → 法→章→节→条 → 父子块 → 向量索引 → 稠密 + BM25 混合检索 → 交叉编码器重排 → 强制引用式生成；外面套一层 Agentic RAG（模型自己决定查什么、查几轮，末端逐个 `[依据N]` 复核支撑）。

## 效果

| 题集 | 规模 | 结果 |
|---|---|---|
| 域内检索 | 82 题 | hit@1 75.6% · hit@3 86.6% · hit@6 91.5% · MRR 0.819 |
| 跨法多跳 · 纯检索 | 63 题 / 126 条 gold | gold 进 top-6 70/126，两条 gold 全中 15/63 |
| 题面自带条号（离线，0 次检索调用） | 112 题 | 唯一定位 107/112，其中取对 107/107 |
| 单跳 · rag vs agent | 82 题 · `qwen3-next-80b` | 首轮 hit@k 逐位相同；同预算 78/93 vs 78/93；全预算 78/93 → 79/93 |
| 跨法多跳 · 跑满 | 63 题 · `qwen-max` | 同预算 59/126 vs 59/126；全预算 59/126 → 68/126 |

口径须知：

- 前三行是纯检索，不花钱：本地 `bge-m3`（1024 维，CLS pooling + L2 归一化，无查询前缀）+ `bge-reranker-v2-m3`（sigmoid），全程不出网。
- 「命中」＝ gold 落在最终 top-6 里；多跳一题 2 条 gold，所以分母是 126 条。
- 重排的净效果（同一个候选池 candidates=20）：单跳 hit@1 72.0% → 75.6%，多跳 58/126 → 70/126；池子放宽到 50 只多捞 3 条 gold，耗时翻倍，不划算。
- 后两行含生成模型（要花钱），是换本地向量/重排之前的数字，未随本次更新；多跳那行的题集后来还发现构建有问题（同一组 gold 反复撞车），绝对数不可信。
- 检索延迟（容器内热态 p50，同一批问题）：纯 BM25 **7ms** ｜ 混合融合 **96ms** ｜ 混合 + 重排 **274ms**。

## 跑起来

### 一路起全（Docker）

```powershell
Copy-Item .env.example .env     # 填 LLM_API_KEY；嵌入/重排已是本地权重，没有远端端点要配
docker compose up -d --build    # 起 Milvus + 服务
docker compose ps               # 等四个容器都 healthy
curl.exe localhost:8000/health  # 必须带 .exe：PowerShell 的 curl 是 IWR 别名
```

`--build` 省不得：镜像把源码烤进去了，不带它跑的还是上一版，且不报任何错。首次构建要下 CUDA 版 torch（约 2.8GB，走镜像源），之后有缓存就快。

没有 LLM key 也能跑 —— 生成层拒答，检索照常（嵌入与重排都在本地权重上跑）；只有 `法规知识库/models/` 里没有权重时，才退化成一套可用的纯 BM25 方案。

### 裸跑（改代码即时生效）

```powershell
pip install -e ".[all]"
docker compose up -d --wait standalone          # 只起 Milvus
# 嵌入/重排权重放 法规知识库/models/（bge-m3、bge-reranker-v2-m3，ModelScope 可下，拉一次之后检索不出网）
python -m traffic_law_qa.cli.build build        # 建库，docx 没变就跳过解析
python -m traffic_law_qa "醉驾怎么处罚"          # 提问
```

### 当库用

```python
from traffic_law_qa import qa

print(qa("醉驾怎么处罚").render())                                 # 检索 + 生成
print(qa("深圳 行人在机动车道 罚款多少", mode="search").render())  # 只检索，不花钱
```

## 入口

| 命令 | 干什么 |
|---|---|
| `python -m traffic_law_qa "问题" [--agent]` | 提问。不带 `--agent` 是线性管道（评测基线），带上走 Agent 循环 |
| `python -m traffic_law_qa.cli.build <子命令>` | 建库：`build` / `status` / `layers` / `docx` / `parse` / `chunk` / `index` |
| `python -m traffic_law_qa.cli.serve [--host H] [--port P]` | 起服务 |
| `python -m traffic_law_qa.eval [--reference]` | 评测：域内 hit@k ｜ 点名桶规则取条 |
| `python -m traffic_law_qa.eval.singlehop` | 单跳两臂对照 |
| `python -m traffic_law_qa.eval.multihop` | 跨法多跳 |
| `python -m traffic_law_qa.eval.corpus build\|check` | 题集桶文件：重建 / 只读比对 |
| `tlq-qa` / `tlq-build` / `tlq-serve` | 上面前三条的短命令 |

一个命令一条路径，`--help` 一律只打印用法，不碰磁盘、不连 Milvus。

## 端点

| 端点 | 干什么 |
|---|---|
| `GET /health` | 启动耗时、法规/条数/块数、通道与重排态、索引是否复用、就绪标记 |
| `POST /qa` | 一次问答：`ask`（检索+生成，默认）｜ `search`（只检索，不花钱）｜ `agent`（多轮工具调用，慢） |
| `POST /qa/stream` | 同上，SSE 边生成边推（`agent` 不走这条） |
| `POST /documents` | 传 docx / md / txt：`session` 只进本次会话（工具面可查，答完即弃）｜ `permanent` 干跑校验后入知识库，落台账 |
| `GET /documents` | 上传台账；`DELETE /documents/{id}` 撤掉一份 |
| `POST /reindex` | 重建索引并热替换运行中的 runtime |

## 它长什么样

`qa("醉驾怎么处罚")` 的真实输出节选：

> 醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。 [依据1]
>
> 参考文献：
>   【依据1】 《中华人民共和国道路交通安全法》(2021-04-29)第九十一条 — 第九十一条　饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。…

正文里的 `[依据N]` 是模型写的，`参考文献` 那段是 `Answer.render()` 按检索结果拼的 —— 编号与原文的对应不经过模型的手。

## 架构

三条链路共用同一份检索与生成：

- **在线问答** `qa()` / `/qa`：改写 → 混合检索 → 重排 → 强制引用式生成，一次到底。
- **Agentic RAG** `--agent`：LangGraph 状态机，模型自己决定查什么、查几轮，末端逐条复核引用是否被原文支撑，低分降级。
- **离线建库** `cli.build`：docx → 条 → 父子块 → Milvus 集合；就绪门在启动时比对 docx、本地产物与集合三者，不一致就重建。

可换的东西都有名字：`ports.py` 里 7 个 ABC（`Embedder` / `Reranker` / `LLM` / `VectorStore` / `RagService` / `IndexStatus` / `IndexBuilder`），换实现只改继承。装配只有一处 —— `container.py`，别处不许自己 new 具体实现。子包 `__init__.py` 一律空着，不做 re-export。

这些不靠自觉：`tests/test_architecture.py` 用 AST 加子进程逐条检查 —— 谁能 import 谁、装配只准在哪发生、agent 与重依赖不许被 `import traffic_law_qa` 顺手拉起来、退役的老包名不许复活、同一句文案只许写一遍、提示词教的工具与真给模型的工具必须一致。

## 技术栈

| 层 | 选型 |
|---|---|
| 向量库 | Milvus 2.6：稠密 + BM25 双路召回，分词与融合都在服务端 |
| 模型 | 生成 `qwen3-30b-a3b-instruct-2507`（`.env.example` 默认 `qwen-flash`）· 向量 `BAAI/bge-m3` · 重排 `BAAI/bge-reranker-v2-m3`（后两个在进程内跑，权重在 `法规知识库/models/`，有 GPU 就用 GPU） |
| 服务 | FastAPI + uvicorn，`/qa/stream` 走 SSE |
| Agent | LangGraph 状态机，LLM 调用直接走 `openai` SDK |
| 依赖 | 基础组只有四个包：`openai` · `pymilvus` · `python-dotenv` · `httpx`；嵌入/重排的 `torch` + `transformers` 在 `local` 可选组，不装也能跑，退化成纯 BM25 |

## 项目结构

```
traffic_law_qa/
├── cli/           三条命令：ask（含 --agent）· build · serve
├── api/           对外一问一答 qa() / render()，HTTP 路由与 app
├── app/           LegalRAG · 上传入库 · 就绪门 · mode 分派
├── agents/        Agentic RAG：判地区 → 单节点循环 → 作答 → 末端复核
├── tools/         工具面：search_law / get_article / search_materials
├── search/        改写 → 混合检索；精确取条与材料合并也在这
├── generation/    强制引用式生成
├── indexing/      建库四步：docx 进，Milvus 集合出
├── infra/         向量库 · 本地模型 · LLM 客户端 · 台账 · 联网检索
├── observability/ 轨迹与 Langfuse
├── contracts/     报文 · 回答 · 磁盘产物 · 报告
├── ports.py       7 个 ABC（换实现只改继承）
├── container.py   ★ 唯一装配点
└── eval/          评测四路

法规知识库/   docx（公开法规原文，唯一真源）→ text → parsed → chunks → index
data/         题集源语料 + 四份桶文件（跑批轨迹不进版本库）
tests/        70 条离线用例，pytest -q 约 6 秒，不碰 Milvus 也不调模型
```

联网检索（`infra/websearch.py`，按博查 API 实现）已实现、有 8 条离线单测，但**当前没挂进工具面**；要恢复就把 `WEB_SEARCH_TOOL` 加回 `agents/nodes.py` 的 `TOOLS`。

## 进阶

自用台账，与上面的项目介绍无关。

- 注释一定有过时的，readme大多可以相信，可以删去其他所有注释重写
    —— 已做：注释与 docstring 全删过一轮；别再加回来。
- agent后面加一个审查正确性agent,再加打分agent测正确率
    —— 前半已落：末端复核节点（判支撑 + 算分 + 低分降级）。还差后半：阈值标定要跑批；
       检测力不足（「给一条依据追加它原文里没有的话」探过两次都判支撑）得换更强模型或加代码校验。
- 测试 —— 已落 70 条离线用例。
- 提示词经过多次迭代有矛盾
- 加网页搜索工具 —— 已实现，但没挂进工具面，见上。
- 加PDF解析
- 微调
- 强化学习
- 记忆（这项目好像需求不高）
- 环境，依赖问题
- 再进阶：数据库优化 / redis / 消息队列 / 更清晰的架构
