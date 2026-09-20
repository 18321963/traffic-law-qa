# 交通法规问答 Agent

面向驾驶员 / 驾校学员 / 交管客服的交通法规问答。用大白话提问，得到每句都挂着法条出处的答案。

底层是一条完整管道：docx 解析 → 结构层 → 切块 → 向量索引 → 混合检索 → 强制引用式生成，外面再套一层 Agentic RAG（模型自己决定查什么、查几轮）。6 部法规、508 条、812 个检索块，从原始 docx 一路建到可问答，每一层都能单独跑、单独验。

## 效果

三套题集测的不是同一件事，数字不能跨表比较。表里第三行不是新题集，是第一行那 82 道上的两臂对照。

| 题集 | 规模 | 测什么 | 结果 |
|---|---|---|---|
| 域内问答 | 82 题 | 检索命中率 | hit@1 69.5% · hit@3 84.1% · hit@6 91.5% · MRR 0.777 |
| 题面自带条号 | 112 题 | 能否绕过检索直接定位 | 唯一定位 107/112，其中取对 107/107 |
| 单跳题 · 两臂并排 | 82 题 · `qwen-flash` | agent 值不值（同预算比首轮） | hit@1 69.5% → 73.2%，hit@3 / hit@6 与 rag 持平 |
| 跨法多跳 | 63 题 · `qwen-max` | 一个答案横跨两部法 | 全预算 59/126 → 68/126；同预算 59/126 vs 59/126 打平 |

读表前先知道三件事：

1. **第一行是这套配置的属性，不是这个项目的属性。** 同一条管道，云端 `text-embedding-v4` 跑出过 hit@1 75.6% / MRR 0.805；默认的本地 `bge-large-zh-v1.5` 不花钱、不出网，代价就是上表那组。引用这些数时带上向量模型。
2. **第三行要带上 LLM。** agent 那一臂的检索词是模型自己写的，换个模型这格就会动；表头默认的 `qwen-max` 跑不出同一组数。
3. **第四行两组数要一起读。** 全预算 agent 68/126 比 rag 的 59/126 高 7pp，但那是用 1.7 次检索换来的；两边都只看第一次检索，完全一样。也就是说在这道题上 agent 没有查得更准，多中的 9 条全部来自「愿意再查一次」。还有一个数没进表：agent 查到 68 条，最终只引用了 61 条。

394 条测试全部离线（不连 Milvus、不发网络请求），其中一批钉住知识库的规模（6 部 / 508 条 / 812 块 / 均长 78.8 字）。解析或切块改坏了，测试先红，而不是等评测数字悄悄掉下来。

## 跑起来

一条命令，服务连 Milvus 一起进容器：

```powershell
Copy-Item .env.example .env     # 填 LLM_API_KEY；向量那几行已指向本地 ollama
docker compose up -d --build    # 起 Milvus + 服务，首次约 2 分钟
docker compose ps               # 等四个容器都 healthy 再访问
curl.exe localhost:8000/health  # 必须带 .exe：PowerShell 的 curl 是 IWR 别名，不认这个地址
```

`--build` 不能省。镜像把源码烤进去了，不带它 compose 会直接复用旧镜像 —— 改了代码却敲 `docker compose up -d`，跑的还是上一版，不报任何错。

想在本机裸跑（改代码即时生效，且能用上 Agent 那条路 —— 容器里的服务只跑线性管道）：

```powershell
pip install -e ".[all]"                        # 依赖
docker compose up -d --wait standalone         # 只起 Milvus，首启 60~90 秒
ollama pull dengcao/bge-large-zh-v1.5          # 向量模型，拉一次之后检索不出网
python -m traffic_law_qa.pipeline build        # 建库，docx 的 sha1 没变就跳过解析
python -m traffic_law_qa "醉驾怎么处罚"         # 提问
```

两条路都不强依赖 key：没有 LLM key 时生成层拒答，检索退化为一套可用的 BM25 关键词方案。装包时带了两个短命令：`tlq-qa "醉驾怎么处罚"`（等价裸跑的最后一步）和 `tlq-serve`（起 HTTP 服务）。

**两个不报错但会咬人的坑**，先记着：

- **向量端点连不上，不报错。** 两条路默认都走本地 Ollama（`.env` 指着 `localhost:11434`，compose 在容器里覆盖成 `host.docker.internal:11434`）。没装或没 pull 到都不报错，但表现分两种：**索引还没建**时编码失败、集合退回纯 BM25，`/health` 的 `channels` 如实报「纯 BM25」；**索引已经建好**（构建机上的 `chunks/`、`index/` 随镜像带了进去）时集合是稠密的，`/health` 照常报「稠密+BM25」，只有查询侧退回 BM25。**不报错，但不是没线索**：启动日志里「稠密通道预热」那行会变成「未执行」，每次检索的结果自己就带着通道状态 —— HTTP 响应是 `retrieval.used_vector: false`（附一句为什么连不上），`mode="search"` 的 `render()` 印「通道：向量=关 BM25=开」。`/health` 也说得清：`channels` 报此刻**实际**在走的通道（这时会变成「纯 BM25」），`dense_built` 报集合**建库时**带没带稠密 —— 两个一比就知道是端点坏了，不是索引本来就只建了 BM25。
- **命名卷的属主是历史遗留。** `chunks/` 与 `index/` 挂在两个命名卷上，而卷**只在首次创建时**继承镜像里同路径目录的属主。镜像现在会先 `mkdir` 再 `chown`，所以新卷没问题；但如果你在这条修复之前跑过一次，手里那两个卷仍是 root 属主 —— 换新镜像也救不回来。它只在**需要重建索引**时才咬人：卷里有一套对得上的产物时走复用分支，一个字节都不写，照样起得来；等 docx 改了、或卷被清过，才撞上 `PermissionError`，容器反复重启（`docker compose logs app` 里能看到）。恢复：`docker compose down -v` 删掉那两个命名卷，下次启动重建索引（Milvus 的数据在 `./volumes/` 里，是 bind mount，不受影响）。**重建要重新向量化，这一步花钱。**

当库用只有一个口子：

```python
from traffic_law_qa import qa

print(qa("醉驾怎么处罚").render())                        # 检索 + 生成
print(qa("深圳 行人在机动车道 罚款多少", mode="search"))   # 只检索，不花 LLM 的钱
```

## 它长什么样

`qa("醉驾怎么处罚")` 的真实输出节选：

> 醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。〔依据1〕
>
> 若醉驾导致交通事故，保险公司在交强险责任限额范围内垫付抢救费用，并有权向致害人追偿；对受害人的财产损失不承担赔偿责任。〔依据4〕
>
> **参考文献：**
> 【依据1】《中华人民共和国道路交通安全法》(2021-04-29) 第九十一条 —— 饮酒后驾驶机动车的……
> 【依据4】《机动车交通事故责任强制保险条例》(2019-03-02) 第二十一条 —— ……

每句话都挂着依据编号，编号能翻回法条原文。法条里没有的，模型会明说「现行依据中未规定」，而不是编一条出来。这是它能不能给交管客服用的前提。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| 向量库 | Milvus 2.6，稠密 + BM25 双路召回，分词与融合都在服务端 |
| 模型 | 生成 `qwen-max` / 向量 `bge-large-zh-v1.5`（本地 Ollama，OpenAI 兼容协议，换模型只改 `.env`） |
| 服务 | FastAPI + uvicorn，`/qa/stream` 走 SSE 流式 |
| Agent | LangGraph 状态机，LLM 调用直接走 `openai` SDK |
| 依赖 | 基础组只有三个包：`openai` · `pymilvus` · `python-dotenv` |

分词、BM25、分页全推给 Milvus 服务端，本地不需要任何 NLP 或检索库；docx 解析用标准库 `zipfile` + `ElementTree`，没引 python-docx。能用标准库就不加包是这里的一条主线。

## 项目结构

```
traffic_law_qa/
├── api.py  __main__.py  pipeline.py  config.py  contracts.py  server.py  obs.py
├── kb/     建库四步：docx 进，Milvus 集合出（只在 build 时跑）
├── qa/     问答三步：改写 → 混合检索 → 强制引用式生成
├── agent/  Agentic RAG：模型自己规划查什么、查几轮
└── eval/   三套题集 + 一组双跑：hit@k / 规则取条探针 / 跨法多跳 / 单跳两臂对照

法规知识库/   docx（唯一真源）→ text → parsed → chunks → index
tests/        394 条，全部离线
data/         两套评测题集
docs/         DESIGN.md
```

子包只按路径说话，`__init__.py` 一律不做 re-export。`kb/` 一导出就会让只想用切块器的调用方连带拖上 pymilvus，`agent/` 一导出就会把 langgraph 拉进 `import traffic_law_qa`。两条都有测试钉着。

## 想深挖

[docs/DESIGN.md](docs/DESIGN.md)。七层管道每层的输入输出契约、检索效果的五个决定因素、每条取舍的代价（为什么是 Milvus 不是 Chroma、为什么不用 MCP 做入库解析），以及一条结构性约束怎么变成可执行的测试。
