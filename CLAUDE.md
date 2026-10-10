# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

面向在本仓工作的 coding agent。只写规则，不写背景；项目介绍与效果数字在 README，不在本文件复述。

## 架构

- 六包拓扑已冻结：`rag_contracts`（两端共用内核：域类型 / ports / config / LLM 客户端 / 埋点，叶子包）· `api_contracts`（HTTP 契约：`openapi.json` 冻结面 + `RagClient`）· `rag_service`（知识库进程）· `agent_service`（Agent 进程）· `eval`（评测与变异自检）· `mcp_server`（stdio 适配层）；加文件可以，别重组目录或换框架。
- 两个进程、一条 HTTP 缝：`agent_service ──RagClient──▶ rag_service`；双向都不许 import 对方。可换件都挂 `rag_contracts/ports.py` 的 ABC；装配只准发生在各自的 `container.py`（agent 侧连装配点都只准造 `RagClient`），别处只收注入。
- 代码里不写注释、不写 docstring（全仓刻意为之）；用法文本放模块级 `USAGE` 常量，且必须真被引用。
- 运行期状态（台账 / 会话 / 上传 / 轨迹 / 备份 / 模型权重）不进镜像：`.dockerignore` 挡着。
- 这些都不靠自觉：`tests/test_architecture.py`、`tests/test_contracts_architecture.py` 用 AST + 子进程逐条钉死 import 方向、装配点、重依赖隔离、退役包名、文案单源、提示词与工具一致、eval 直取边、前端 SSE 帧词汇、零注释、README 引用。改结构或文案前先读这两个文件，别绕门。

## 常用命令

全部经 `.venv/Scripts/python.exe`（理由见「本机操作」）；命令全表在 README「入口」。

- `python -m pytest`：全量离线用例（不碰 Milvus、不调模型）。单文件加路径，单条再 `::test_name`。
- `ruff check .`：静态检查。
- `python -m eval.mutations.run all`：变异自检四张表（retrieval / contract / gates / mcp）；单张表、`--only 名字片段` 见其 USAGE。
- `python -m eval`：域内检索评测，不花钱；`eval.singlehop` / `eval.multihop` 要模型端点（花钱，先报量）。
- `python -m rag_service "问题"` / `python -m agent_service "问题" --timing`：两条问答路径；`python -m rag_service.cli.build build` 建库。
- `deploy/build.sh`：重建三个镜像并滚动上线；`docker compose -f deploy/docker-compose.yml up -d` 起全栈（dev 覆盖文件恢复开发端口）。
- `python -m api_contracts.regen`：改接口后重出 `openapi.json`，同时改 `docs/04_接口文档.md`。

## 测试与自检

- 判绿 = 退出码 + 点阵里的 F/E，整份输出落盘再判（本环境 `pytest -q` 不打汇总行；`addopts` 已带 `-q`，别再追加）。
- 全绿不等于覆盖：改行为面必须配变异枪 —— 验收标准是「把这条规则改坏，对应测试必须翻红」。脚本放仓外 `D:\projects\_tlq_*.py`；跑完用 `git diff` 核对无残留（被打断的变异跑会把脏源码当下一轮基线）。
- CI：push 到 main 跑 `.github/workflows/test.yml`（Python 3.11 + pytest + ruff；前端 npm ci + build）——不建镜像、不跑 E2E。
- worktree 里跑测试：`cd <worktree> && D:\projects\交通法规问答agent\.venv\Scripts\python.exe -m pytest`（`-m` 把 cwd 置顶，测的是 worktree 的代码；直接调 `.venv` 里的 pytest 会把测试跑回主仓）。

## 文档

- README 是唯一对外文档：只放结果与口径须知；文末「进阶」是用户自留台账，原样保留、别清理。
- 多处自检文案按名引用 README 小节（`README「入口」`）：改 README 章节标题前先 `grep -rn "README「"`，门禁会翻红。
- 改接口必须同步 `docs/04_接口文档.md`（签名级契约）；文档清单与口径在 `docs/README.md`。
- 对外文案不用「demo」一词，用「服务」「示例」。

## 本机操作

- 跑 Python 一律 `.venv/Scripts/python.exe`：裸 `python` 落到 anaconda 环境，报错会伪装成「Milvus 连不上」；代码里起子进程必须用 `sys.executable`。
- 中文不要过 Git Bash argv（会变 GBK）：请求体落盘后 `--data-binary @file`，中文提交信息用 `-F` 落盘。
- 外网（GitHub、LLM API）走代理 `127.0.0.1:7890`；报 Connection error 先看它是不是没在跑。
- 容器跑的是烤进镜像的代码：改源码不跑 `deploy/build.sh`，跑的还是旧版且不报错。**禁用** `docker compose up --build`（本机 compose v5.4.0 内嵌 bake 坏，报错像网络问题）与 `docker builder prune`；不要强杀 Docker Desktop（卡死先重启 Windows），Docker 数据盘在 D:。
- 容器起着时宿主别载入大模型（OOM 1455 / 段错误 139，不是 GPU 坏）；要延迟数字就在容器里量。
- 改源码用 Edit 工具，不用脚本 `write_text`（Windows 上会把 LF 变 CRLF）；`pip install -e .` 带 `--no-build-isolation`（带隔离会重建仓库根 egg-info，盖住已装元数据）。
- 往根 `.env` 注入过 keys/限额给容器冒烟后，判宿主全量 pytest 前先还原（否则 wire 面一批 401 假红）。

## 成本与授权

- 批量外呼模型（LLM、网络搜索、跑评测）前先报量：几次调用、大概多久、量什么，并给一个「不跑也能收口」的备选。
- 删除任何既有文件（含 egg-info、Docker 镜像）需要用户在本人消息里点名；AskUserQuestion 的确认不算。
- 提交形状：特性分支 → 中文提交信息（`-F` 落盘，不塞 argv）→ `merge: 合入 <分支>（中文摘要）` 进 main；推送等用户明说。
