# 交通法规问答 Agent

用大白话问交通法规问题，答案里每句结论都挂着法条出处 —— `[依据N]` 能翻回原文；法条里没有的，它会说「库中未收录」并指出缺哪类规定，而不是编一条出来。

6 部法规 / 508 条 / 812 个检索块，从 docx 一路建到可问答：解析 → 法→章→节→条 → 父子块 → 向量索引 → 稠密 + BM25 混合检索 → 强制引用式生成；外面套一层 Agentic RAG（模型自己决定查什么、查几轮，末端逐个 `[依据N]` 复核支撑）。

## 效果

| 题集 | 规模 | 结果 |
|---|---|---|
| 域内检索 | 82 题 | hit@1 69.5% · hit@3 84.1% · hit@6 91.5% · MRR 0.777 |
| 题面自带条号（离线，0 次检索调用） | 112 题 | 唯一定位 107/112，其中取对 107/107 |
| 单跳 · rag vs agent | 82 题 · `qwen3-next-80b` | 首轮 hit@k 逐位相同；同预算 78/93 vs 78/93；全预算 78/93 → 79/93 |
| 跨法多跳 · 跑满 | 63 题 · `qwen-max` | 同预算 59/126 vs 59/126；全预算 59/126 → 68/126 |


## 跑起来

```powershell
Copy-Item .env.example .env     # 填 LLM_API_KEY；向量那几行已指向本地 ollama
docker compose up -d --build    # 起 Milvus + 服务，首次约 2 分钟；--build 不能省
docker compose ps               # 等四个容器都 healthy
curl.exe localhost:8000/health  # 必须带 .exe：PowerShell 的 curl 是 IWR 别名
```

`--build` 省不得：镜像把源码烤进去了，不带它跑的还是上一版，且不报任何错。没有 LLM key 也能跑 —— 生成层拒答，检索退化成一套可用的 BM25 方案。

裸跑（改代码即时生效，命令行 `--agent` 也在这里）：

```powershell
pip install -e ".[all]"
docker compose up -d --wait standalone         # 只起 Milvus
ollama pull dengcao/bge-large-zh-v1.5          # 向量模型，拉一次之后检索不出网
python -m traffic_law_qa.pipeline build        # 建库，docx 没变就跳过解析
python -m traffic_law_qa "醉驾怎么处罚"         # 提问
```

当库用只有一个口子：

```python
from traffic_law_qa import qa

print(qa("醉驾怎么处罚").render())                                 # 检索 + 生成
print(qa("深圳 行人在机动车道 罚款多少", mode="search").render())  # 只检索，不花钱
```

## 入口

```powershell
python -m traffic_law_qa "问题" [--agent]           # 提问。不带 --agent 是线性管道（评测基线）
python -m traffic_law_qa.pipeline <子命令>          # 建库：build / status / layers / docx / parse / chunk / index
python -m traffic_law_qa.main [--host H] [--port P] # 起服务：/qa、/qa/stream、/documents、/reindex、/health
python -m traffic_law_qa.eval [--reference]         # 评测：域内 hit@k ｜ 点名桶规则取条
python -m traffic_law_qa.eval.singlehop             # 单跳两臂对照
python -m traffic_law_qa.eval.multihop              # 跨法多跳
python -m traffic_law_qa.eval.corpus build|check    # 题集桶文件：重建 / 只读比对
tlq-qa / tlq-serve                                  # 上面第 1、3 条的短命令
```

一个命令一条路径，`--help` 一律只打印用法，不碰磁盘、不连 Milvus。

## 它长什么样

`qa("醉驾怎么处罚")` 的真实输出节选：

> 醉酒驾驶机动车的，由公安机关交通管理部门约束至酒醒，吊销机动车驾驶证，依法追究刑事责任；五年内不得重新取得机动车驾驶证。 [依据1]
>
> 参考文献：
>   【依据1】 《中华人民共和国道路交通安全法》(2021-04-29)第九十一条 — 第九十一条　饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。…

正文里的 `[依据N]` 是模型写的，`参考文献` 那段是 `Answer.render()` 按检索结果拼的 —— 编号与原文的对应不经过模型的手。

## 技术栈

| 层 | 选型 |
|---|---|
| 向量库 | Milvus 2.6：稠密 + BM25 双路召回，分词与融合都在服务端 |
| 模型 | 生成 `qwen3-30b-a3b-instruct-2507`（`.env.example` 默认 `qwen-flash`）· 向量 `bge-large-zh-v1.5`（本地 Ollama，换模型只改 `.env`） |
| 服务 | FastAPI + uvicorn，`/qa/stream` 走 SSE |
| Agent | LangGraph 状态机，LLM 调用直接走 `openai` SDK |
| 依赖 | 基础组只有四个包：`openai` · `pymilvus` · `python-dotenv` · `httpx` |

## 项目结构

```
traffic_law_qa/
├── api/       对外一问一答 qa() / render()，HTTP 路由在 routes.py
├── kb/        建库四步：docx 进，Milvus 集合出
├── qa/        问答三步：改写 → 混合检索 → 强制引用式生成
├── agents/    Agentic RAG：判地区 → 单节点循环 → 作答 → 末端复核
├── tools/     工具面：search_law / get_article / search_materials
├── services/  llm · embedding · ingest（上传材料与永久入库）
├── database/  上传台账（sqlite）
└── eval/      评测四路

法规知识库/   docx（公开法规原文，唯一真源）→ text → parsed → chunks → index
data/         题集源语料 + 四份桶文件（跑批轨迹不进版本库）
tests/        45 条离线用例，pytest -q 约 5 秒，不碰 Milvus 也不调模型
```

子包只按路径说话，`__init__.py` 一律不做 re-export：`import traffic_law_qa` 不许拉起 langgraph —— 这条写进了回归检查。联网检索（`tools/web_search.py`，按博查 API 实现）已实现、有 8 条离线单测，但**当前没挂进工具面**；要恢复就把 `WEB_SEARCH_TOOL` 加回 `agents/nodes.py` 的 `TOOLS`。

## 进阶

自用台账，与上面的项目介绍无关。

- 注释一定有过时的，readme大多可以相信，可以删去其他所有注释重写
    —— 已做：注释与 docstring 全删过一轮；别再加回来。
- agent后面加一个审查正确性agent,再加打分agent测正确率
    —— 前半已落：末端复核节点（判支撑 + 算分 + 低分降级）。还差后半：阈值标定要跑批；
       检测力不足（「给一条依据追加它原文里没有的话」探过两次都判支撑）得换更强模型或加代码校验。
- 测试 —— 已落 45 条离线用例。
- 提示词经过多次迭代有矛盾
- 加网页搜索工具 —— 已实现，但没挂进工具面，见上。
- 加PDF解析
- 微调
- 强化学习
- 记忆（这项目好像需求不高）
- 环境，依赖问题
- 再进阶：数据库优化 / redis / 消息队列 / 更清晰的架构
