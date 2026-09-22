# 交通法规问答 Agent

面向驾驶员 / 驾校学员 / 交管客服的交通法规问答。用大白话提问，得到每句都挂着法条出处的答案。

底层是一条完整管道：docx 解析 → 结构层 → 切块 → 向量索引 → 混合检索 → 强制引用式生成，外面再套一层 Agentic RAG（模型自己决定查什么、查几轮）。6 部法规、508 条、812 个检索块，从原始 docx 一路建到可问答，每一层都能单独跑、单独验。

## 效果

**能跑。**三套题集测的不是同一件事，数字不能跨表比较。表里第三行不是新题集，是第一行那 82 道上的两臂对照；第五行也不是，是第四行换一个模型再跑一遍。

| 题集 | 规模 | 测什么 | 结果 |
|---|---|---|---|
| 域内问答 | 82 题 | 检索命中率 | hit@1 69.5% · hit@3 84.1% · hit@6 91.5% · MRR 0.777 |
| 题面自带条号 | 112 题 | 库层规则能否唯一定位条号（不经 agent） | 唯一定位 107/112，其中取对 107/107 |
| 单跳题 · 两臂并排 | 82 题 · `qwen3-next-80b-a3b-instruct` | agent 值不值（同预算比首轮） | 三个臂的 hit@k 逐位相同（69.5% · 84.1% · 91.5%），MRR 0.777 / 0.777 / 0.779；同预算 78/93 vs 78/93；全预算 78/93 → 79/93 |
| 跨法多跳 | 38/63 题 · `qwen3-next-80b-a3b-instruct` | 一个答案横跨两部法 | 全预算 31/76 → 36/76；同预算 31/76 vs 31/76 |
| 跨法多跳 · 同 38 题换模型 | 38 题 · `qwen-max` | 上面那个「打平」扛不扛得住换模型 | 全预算 31/76 → 39/76；同预算 31/76 vs 31/76 |

读表前先知道五件事：

1. **第一行是这套配置的属性，不是这个项目的属性。** 同一条管道，云端 `text-embedding-v4` 跑出过 hit@1 75.6% / MRR 0.805；默认的本地 `bge-large-zh-v1.5` 不花钱、不出网，代价就是上表那组。引用这些数时带上向量模型。
2. **后三行都要带上 LLM。** agent 那一臂的检索词是模型自己写的，换个模型这几格就会动 —— 这组数出自 `qwen3-next-80b-a3b-instruct`（本机 `.env` 的值；`.env.example` 的默认值仍是 `qwen-flash`），最后一行是 `qwen-max`。
3. **第四行两组数要一起读。** 全预算 agent 36/76 比 rag 的 31/76 高，但那是用 1.9 次检索换来的；两边都只看第一次检索，总数完全相同 —— 38 题里 4 题多中、4 题少中、1 题数量相同但换了条，正好抵消。多中的那几条全来自「愿意再查一次」，不是第一次就查得更准。还有一个数没进表：agent 查到 36 条，最终只引用了 33 条。
4. **同预算那格逐题重不重合，两行不一样。** 单跳 82 题逐题重合（三个臂逐位相同）—— 首轮检索词有 17 题与原话有出入，但都是删个问号、去掉一句「这个条款中的……是什么」这类等价改写，命中结果一字不差。多跳这 38 题有 9 题出入：那 9 题的第 1 轮把场景压成了关键词短查询（「我骑共享单车闯红灯被拍了，交警说要罚我，但我不是机动车……」→「骑共享单车闯红灯怎么处罚」）。
5. **第四行只有 38 题，是全 63 题的前 38 题** —— 跑到第 39 题时模型端免费额度用尽，断了。是截断，不是抽样：38 题不能当 63 题读。最后一行是同一批 38 题在旧模型上的数，两行之间只差模型，与题集无关。

表里 agent 那几格跑的是现在的循环：单节点，模型自己判断证据够不够、自己决定停，没有独立审核轮。轨迹都在 `data/traces/` 里（`single82_q3next.json` / `hop63_q3next.json`），每一格都能从轨迹重算。

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

想一眼看全检索质量，这 4 条依次问一遍 —— 都加 `--search --debug`，不花 LLM 的钱，每条盯一个已知技术点。默认两条通道一起跑，每条几秒，实测这几秒几乎全花在把查询向量化上（纯 BM25 的 `--no-vector` 是 22 毫秒级），检索本身不是瓶颈：

| 问题 | 盯着的技术点 |
|---|---|
| `醉驾怎么处罚` | 口语对齐：醉驾 → 醉酒驾驶 |
| `深圳 行人 在机动车道 罚款多少` | 法名线索加成：命中法规的分数 ×1.5 |
| `智能网联汽车 道路测试要满足什么条件` | 同一部特别条例的召回 |
| `开车不系安全带怎么处罚` | 基线：前三条的改动不该影响普通问题 |

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
| `python -m traffic_law_qa.eval.corpus build \| check` | 题集文件：把源语料切成三份桶文件（82 / 112 / 45）；`check` 只读比对有没有跟源语料走散 |
| `python -m traffic_law_qa.eval [--reference]` | 读 `data/eval_retrieval.json` 跑域内 82 题：hit@k / MRR；`--reference` 换读 112 道点名桶，量规则取条能否唯一定位 |
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
data/         题集源语料 + 由它切出的三份题集文件 + 跨法多跳一份 + 历史跑批轨迹
```

子包只按路径说话，`__init__.py` 一律不做 re-export。`kb/` 一导出就会让只想用切块器的调用方连带拖上 pymilvus，`agent/` 一导出就会把 langgraph 拉进 `import traffic_law_qa`。

## 进阶
注释一定有过时的，readme大多可以相信，可以删去其他所有注释重写
agent后面加一个审查正确性agent,再加打分agent测正确率
测试
提示词经过多次迭代有矛盾
加网页搜索工具
加PDF解析
微调
强化学习
记忆（这项目好像需求不高）
环境，依赖问题
入口疑似太多了

## 再进阶
数据库优化
redis
消息队列
更清晰的架构


