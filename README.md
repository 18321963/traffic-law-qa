# 交通法规问答 Agent

面向驾驶员 / 驾校学员 / 交管客服的交通法规问答。用大白话提问，得到每句都挂着法条出处的答案。

底层是一条完整管道：docx 解析 → 结构层 → 切块 → 向量索引 → 混合检索 → 强制引用式生成，外面再套一层 Agentic RAG（模型自己决定查什么、查几轮）。6 部法规、508 条、812 个检索块，从原始 docx 一路建到可问答，每一层都能单独跑、单独验。

## 效果

**能跑。**三套题集测的不是同一件事，数字不能跨表比较。表里第三行不是新题集，是第一行那 82 道上的两臂对照；第五行也不是，是第四行换一个模型再跑一遍。

| 题集 | 规模 | 测什么 | 结果 |
|---|---|---|---|
| 域内问答 | 82 题 | 检索命中率 | hit@1 69.5% · hit@3 84.1% · hit@6 91.5% · MRR 0.777 |
| 题面自带条号 | 112 题 | 能否绕过检索直接定位 | 唯一定位 107/112，其中取对 107/107 |
| 单跳题 · 两臂并排 | 82 题 · `qwen-flash` | agent 值不值（同预算比首轮） | hit@1 69.5% → 73.2%，hit@3 / hit@6 与 rag 持平 |
| 跨法多跳 | 63 题 · `qwen-max` | 一个答案横跨两部法 | 全预算 59/126 → 68/126；同预算 59/126 vs 59/126 逐条打平 |
| 跨法多跳 · 换模型复跑 | 50 题 · `qwen-turbo` | 上面那个「打平」扛不扛得住换模型 | 全预算 43/100 → 54/100；同预算 43/100 vs agent 首次 37/100 —— **打平不再**，但 p = 0.109 |

读表前先知道四件事：

1. **第一行是这套配置的属性，不是这个项目的属性。** 同一条管道，云端 `text-embedding-v4` 跑出过 hit@1 75.6% / MRR 0.805；默认的本地 `bge-large-zh-v1.5` 不花钱、不出网，代价就是上表那组。引用这些数时带上向量模型。
2. **第三行要带上 LLM。** agent 那一臂的检索词是模型自己写的，换个模型这格就会动 —— 这组数出自 `qwen-flash`（也是 `.env.example` 的默认值）。
3. **第四行两组数要一起读。** 全预算 agent 68/126 比 rag 的 59/126 高 7pp，但那是用 1.7 次检索换来的；两边都只看第一次检索，逐条命中完全相同。也就是说在这道题上 agent 没有查得更准，多中的 9 条全部来自「愿意再查一次」。还有一个数没进表：agent 查到 68 条，最终只引用了 61 条。换模型后这个「打平」就不成立了，见第五行。
4. **第四行是 `qwen-max` 一个模型的结论，别当与模型无关的数读。** 那一轮 agent 的审核节点其实没接上配置的审核模型，规划与审核是同一个模型在跑 —— 也就是说「愿不愿意再查一次」完全由 `qwen-max` 决定。查证过程见 [docs/DESIGN.md](docs/DESIGN.md) §9「两次接线事故」。

401 条测试全部离线（不连 Milvus、不发网络请求），其中一批钉住知识库的规模（6 部 / 508 条 / 812 块 / 均长 78.8 字）。解析或切块改坏了，测试先红，而不是等评测数字悄悄掉下来。

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

两条路都不强依赖 key：没有 LLM key 时生成层拒答，检索退化为一套可用的 BM25 关键词方案。装包时另带了两个短命令，完整清单见下面「所有入口」。

当库用只有一个口子：

```python
from traffic_law_qa import qa

print(qa("醉驾怎么处罚").render())                                  # 检索 + 生成
print(qa("深圳 行人在机动车道 罚款多少", mode="search").render())  # 只检索，不花 LLM 的钱
```

## 所有入口

装好包（`pip install -e .`）之后能敲的全部路径。`kb/` 那四个不认 `--help` —— 它们手搓解析 `sys.argv`，`--help` 会被当成普通参数**真的开始干活**，其余七个都由 argparse 在解析阶段拦下。

| 命令 | 干什么 |
|---|---|
| `python -m traffic_law_qa "问题"` | 提问：确保索引就绪 → 检索 → 生成 |
| `python -m traffic_law_qa.pipeline build \| status \| layers` | 建库 / 看库内规模 / 看七层管道的输入输出 |
| `python -m traffic_law_qa.kb.docx_reader [docx …]` | 建库第一步：docx → 段落，不给路径就跑全部 |
| `python -m traffic_law_qa.kb.law_parser [--force] [--only 法id]` | 第二步：段落 → 法→章→节→条，sha1 没变就跳过 |
| `python -m traffic_law_qa.kb.chunker [--show 条号]` | 第三步：条文 → 父子块 |
| `python -m traffic_law_qa.kb.indexer [--no-vector] [--query 词]` | 第四步：块 → Milvus 集合 |
| `python -m traffic_law_qa.agent "问题" [--trace]` | Agentic RAG：模型自己决定查什么、查几轮 |
| `python -m traffic_law_qa.eval [--reference]` | 域内 82 题：hit@k / MRR；`--reference` 换跑 112 道点名桶，量规则取条能否唯一定位 |
| `python -m traffic_law_qa.eval.singlehop` | 单跳 82 题：rag vs agent 两臂对照 |
| `python -m traffic_law_qa.eval.multihop` | 跨法多跳题集 |
| `python -m traffic_law_qa.server [--host H] [--port P]` | 起 HTTP 服务，默认 8000 |
| `tlq-qa "问题"` | 短命令，等价 `python -m traffic_law_qa` |
| `tlq-serve` | 短命令，等价 `python -m traffic_law_qa.server` |

`…eval` 与 `…agent` 各是一次纯转发：实现在 `eval/harness.py` 和 `agent/cli.py`，那两个模块也能直接 `-m` 跑（`…eval.harness` / `…agent.cli`），转发文件存在的意义只是让重构前记熟的那条命令继续能用。

十三条路径都验过：十一条 `-m` 起得来，两个短命令正常，服务那条 `/health` 返回 200。

## 它长什么样

`qa("醉驾怎么处罚")` 的真实输出节选（`qwen-flash` 实跑：7 句结论、6 条依据，这里各留 2 条）：

> 醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证 [依据1]。
>
> 保险公司可在机动车交通事故责任强制保险责任限额范围内垫付抢救费用，并有权向致害人追偿，但不承担财产损失赔偿责任 [依据4]。
>
> 参考文献：
>   【依据1】 《中华人民共和国道路交通安全法》(2021-04-29)第九十一条 — 第九十一条　饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。因饮酒后驾驶机动车被处罚，再次饮…
>   【依据4】 《机动车交通事故责任强制保险条例》(2019-03-02)第二十二条 — 第二十二条　有下列情形之一的，保险公司在机动车交通事故责任强制保险责任限额范围内垫付抢救费用，并有权向致害人追偿： (一…

正文里的 `[依据N]` 是模型写的，`参考文献：` 那一段是 `Answer.render()` 按检索结果拼的 —— 编号与法条原文的对应关系不经过模型的手。每句话都挂着依据编号，编号能翻回法条原文；法条里没有的，模型会明说「现行依据中未规定」，而不是编一条出来。这是它能不能给交管客服用的前提。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| 向量库 | Milvus 2.6，稠密 + BM25 双路召回，分词与融合都在服务端 |
| 模型 | 生成 `qwen-flash` / 向量 `bge-large-zh-v1.5`（本地 Ollama，OpenAI 兼容协议，换模型只改 `.env`） |
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
└── eval/   四条评测路径：域内 hit@k / 规则取条探针 / 跨法多跳 / 单跳两臂对照

法规知识库/   docx（唯一真源）→ text → parsed → chunks → index
tests/        401 条，全部离线（一定很多过时的）
data/         题集源文件两份（派生上表那三套题）+ 历史跑批轨迹
docs/         DESIGN.md
```

子包只按路径说话，`__init__.py` 一律不做 re-export。`kb/` 一导出就会让只想用切块器的调用方连带拖上 pymilvus，`agent/` 一导出就会把 langgraph 拉进 `import traffic_law_qa`。两条都有测试钉着。

## 想深挖

[docs/DESIGN.md](docs/DESIGN.md)。七层管道每层的输入输出契约、检索效果的五个决定因素、每条取舍的代价（为什么是 Milvus 不是 Chroma、为什么不用 MCP 做入库解析），以及一条结构性约束怎么变成可执行的测试。

## 进阶
注释一定有过时的，readme大多可以相信，可以删去其他所有注释重写
测试过时更多，要删了重写
审查agent后面加一个打分agent测正确率
加网页搜索工具
加PDF解析
微调
强化学习
记忆（这项目好像需求不高）
最快效果最好的方法是提示词，现在提示词有问题


## 再进阶
数据库优化
redis
消息队列
更清晰的架构


