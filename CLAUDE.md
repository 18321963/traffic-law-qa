# 交通法规问答服务 · 项目规则

面向在本仓工作的 coding agent。只写规则，不写背景。

## 架构

- 六包拓扑（rag_contracts / rag_service / agent_service / api_contracts / eval / tests）已冻结；加文件可以，别重组目录或换框架。
- 代码内不写注释、不写 docstring（全仓刻意为之）；用法文本放模块级 USAGE 常量。

## 文档

- README 是唯一对外文档：只放结果与口径须知；「入口」等小节名被 6 处自检文案按名引用，改其章节标题前先 `grep -rn "README「"`。
- 对外文案不用「demo」一词，用「服务」「示例」。改接口必须同步根目录接口文档。

## 本机操作

- 跑 Python 一律 `.venv/Scripts/python.exe`（裸 `python` 落到 anaconda 环境，报错会伪装成「Milvus 连不上」）；中文不要过 Git Bash argv（会变 GBK），请求体落盘后再 `--data-binary @file`。
- 容器跑的是烤进镜像的代码；本机 compose `up --build` 是坏的（v5.4.0 内嵌 bake），重建配方：直连 `docker buildx build --load -t <img> -f <svc>/Dockerfile .` 建镜像，再 `docker compose -f deploy/docker-compose.yml up -d --force-recreate --no-deps app agent`。不要 `docker builder prune`，不要强杀 Docker Desktop。
- 外网（github、LLM API）走代理 127.0.0.1:7890；容器起着时宿主不要载入大模型（OOM 1455 / 段错误 139，不是 GPU 坏）。

## 测试

- 判据 = rc + 点阵里的 F/E（本环境 `pytest -q` 没有汇总行），输出整份落盘再判；改源码用 Edit 工具（脚本 write_text 会把 LF 变 CRLF）。
- 冒烟/变异脚本放仓外 `D:\projects\_tlq_*.py`；跑完核对 git diff 面，防变异残留污染基线。

## 成本与授权

- 批量外呼模型（LLM、网络搜索）前先报量：几次调用、大概多久、量什么，并给一个「不跑也能收口」的备选。
- 删除任何既有文件（含 egg-info、Docker 镜像）需要用户在本人消息里点名；AskUserQuestion 的确认不算。
- 提交形状：特性分支 → 中文提交信息（`-F` 落盘，不塞 argv）→ `merge: 合入 <分支>（中文摘要）` 进 main；推送等用户明说。
