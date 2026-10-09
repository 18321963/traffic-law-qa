# 生产化规格：小范围生产运行

> 内部工作文档（非对外；对外只认 README）。分工见 `prompt_plan.md`，验收见 `eval_plan.md`。
> 范围 = 2026-10-09 评估出的 P0/P1；P2（运行账本/事件表）明确不做。

## 0. 目标环境假设（D1）

- 单机 Docker（当前 Windows + Docker Desktop 形态；换 Linux 单机不改变任何条目）。
- 可达面：内网 / 白名单，**不面向公网**；使用者 ≤ 20 人，同时问答 ≤ 4 路。
- 部署单元：`app`(rag) + `agent` + Milvus 三件套，同一 compose；agent 是唯一对外入口。
- 若改公网：追加 TLS 终结 + IP 层限流 + key 轮换（本轮不做，另起条目）。

## 1. 现状基线（已达标，别当欠账）

- healthcheck / depends_on(service_healthy) / `restart: unless-stopped` 都在。
- 契约冻结 + 漂移测试 + AST 结构门 + 变异矩阵纪律。
- 观测：Langfuse（LLM=generation、检索=span、失败显红、被拦调用=WARNING span）。
- 超时与降级：`AGENT_TOOL_TIMEOUT` → `QaTimeout`；检索失败/超时/法规无法确定都有回执，模型可换招。
- 结论：代码质量面已超过多数小范围生产；本轮补的全是运营面。

## 2. 需求（R1–R12）

### R1 鉴权（决策 D2：agent 应用内中间件）
- `AGENT_API_KEYS`（逗号分隔；空 = 关，默认关）。非空时，除 `/health`、`/laws` 外所有路径要求请求头 `X-API-Key`。
- 校验用 `hmac.compare_digest` 逐 key 比较；不通过 → 401 `{"detail": "API key 缺失或无效"}`。
- rag 服务不加应用鉴权：不对外暴露，纵深靠网络面（base compose 里它没有 host 端口）。
- 文件面：`agent_service/api/auth.py`(新)、`agent_service/api/app.py`、`tests/test_agent_auth.py`(新)。

### R2 限流（按 key，分钟级）
- `AGENT_RATE_LIMIT_RPM`（默认 30；0 = 关）。按 key 固定窗口计数（进程内，单副本口径）。
- 超限 → 429 + `Retry-After`（到窗口结束的秒数）+ 中文 detail。`/health`、`/laws` 豁免。
- 与 R1 同一张卡片（S1）。

### R3 成本闸（全局日额度）
- `AGENT_DAILY_TOKEN_BUDGET`（默认 0 = 关）。>0 时：进入 run 前查当日已用 tokens，已达 → 429（detail 含预算与已用量）+ `Retry-After`（到次日 UTC 0 点）。
- 账本：`data/usage.db`，表 `usage_daily(day TEXT PRIMARY KEY, tokens INTEGER, calls INTEGER)`，day = UTC 日期；run 终态按 `usage` 通道求和累加。
- 写库失败只打 stderr、不拦请求（可用性优先）。全局额度不按 key（决策 D3，按 key 留到有计费需求时）。
- 文件面：S5 卡片。

### R4 检索并发闸（rag 侧）
- `RAG_MAX_CONCURRENCY`（默认 4；0 = 关）信号量包住重活段：`/qa`、`/answer`、`/answer/stream`、`/materials/search`。
- `RAG_QUEUE_TIMEOUT`（默认 30.0 秒）：等信号量超时 → 503 + `Retry-After: 5` + 中文 detail。
- `/health`、`/laws`、`/articles/lookup` 不进闸。503 到 agent 侧走既有链路（QaError → 「检索失败」回执），agent 不改。
- 文件面：S3 卡片。

### R5 暴露面收缩
- base compose（生产面）：删除 milvus(19530/9091)、minio(9000/9001)、rag(8000) 的 host 端口映射；agent 8001 保留为内网入口。
- MinIO 凭据改 compose 必填插值 `${MINIO_ROOT_USER:?}` / `${MINIO_ROOT_PASSWORD:?}`。
- 各服务统一 `logging`：json-file，max-size 10m，max-file 3。
- 开发便利走覆盖文件 `deploy/docker-compose.dev.yml`（恢复全部端口 + 默认凭据）；用法：`docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.dev.yml up -d`。
- 文件面：S4 卡片。

### R6 会话保留清理
- `AGENT_SESSION_TTL_DAYS`（默认 30；0 = 不清理）。判活 = 该 thread 最新 checkpoint 的 `checkpoint_id` 时间（langgraph 用 uuid6，时间有序；先用真 saver 的探针测试证实，不成立就停下报告）。
- CLI 跑在 agent 容器内（命名卷宿主不可直接访问，决策 D6）：`docker compose exec agent python -m agent_service.agents.session_cleanup [--ttl N] [--apply]`；默认 dry-run 打印（线程总数/可删数/最老与最新时间/预计行数），`--apply` 走 `delete_thread` 后 VACUUM，收尾打一行机器可读汇总。
- 定时由宿主计划任务调 exec（手册草稿由 S2 交、W3 落笔）。
- 文件面：S2 卡片。

### R7 探活告警
- `deploy/watchdog.sh`：单次探测模式（给计划任务）+ `--loop N`；探 `/health`，agent 判据同 compose healthcheck（`status==ok && rag.status==ok`）。
- `WATCHDOG_WEBHOOK_URL` 非空时失败 curl POST 告警 JSON；空则只打日志。落法二选一写进手册（宿主计划任务 / Uptime Kuma 探同判据）。
- 文件面：S4 卡片。

### R8 备份与恢复
- 冷备（决策 D4：不引 milvus-backup）：`deploy/backup.sh` 停写（stop app agent standalone）→ 打包 bind 卷目录（volumes/etcd、volumes/minio、volumes/milvus，宿主 tar）+ 命名卷 tlr_sessions（helper 容器 tar）→ 起栈。
- 产物 `deploy/backup/<YYYYmmdd-HHMM>/`；`--keep N` 保留最近 N 份；`--restore <dir>` 反向恢复（含缺卷自动建）。
- 不备份可重建物（index/chunks 卷，重建配方在 README/手册）。
- 文件面：S6 卡片；演练剧本在 eval_plan §5。

### R9 CI
- `.github/workflows/test.yml`：ubuntu-latest + Python 3.11；`pip install -e ".[all]"`（torch 是唯一重件，装太慢就换 CPU 索引）；跑 `pytest -q` + `ruff check .`；on: push / PR → main。
- 注：本机推送后才触发（推送等用户明说）；CI 内不需要代理。

### R10 构建固化
- `deploy/build.sh`：固化直连 buildx 配方（两条 `docker buildx build --load`）+ `compose up -d --force-recreate --no-deps app agent`；头注释写明禁用 `compose up --build` 且不得 prune。

### R11 口径落文（W3 统一写）
- `deploy/运维手册.md`：部署（build.sh/dev 覆盖）、日常（ps/logs/healthcheck 看什么）、告警（watchdog/计划任务/Kuma）、会话清理、备份恢复、以及四条口径：单副本（SQLite 会话库写者假设，横向扩先换 Postgres checkpointer）、复核是辅助（检测力边界）、web_search 不可见但盲猜可达的 egress 说明、SSE 过场流无补发（done 帧才是权威）。
- `接口文档.md`：按 §4 清单落。
- README：仅当启动命令变化时最小同步内容；**不动小节标题**（先 `grep -rn "README「"`）。

### R12 依赖漂移口径
- 不引 lockfile（决策 D5）：维持「镜像即制品」；CI 每次打印 `pip freeze` 进日志备查。

## 3. 非目标（本轮不做，别顺手）

- P2 运行账本与事件表（agent_runs/model_calls/tool_calls/run_events）：Langfuse 已覆盖观测；仅当前端要时间线/断线补发时另立项。
- 公网加固、多租户 org_id、按 key 计费。
- 横向扩容（多副本）。
- 答案质量调参与全量 eval 重跑（花钱，另报量）。
- 六包拓扑；注释/docstring 房规。

## 4. 接口面变更汇总（W3 落接口文档用）

- 新请求头：`X-API-Key`（仅 `AGENT_API_KEYS` 非空时被要求）。
- 新错误面：401（key 无效）、429（限流，带 Retry-After）、429（日额度）、503（检索并发满，带 Retry-After）。
- 新环境变量：`AGENT_API_KEYS`、`AGENT_RATE_LIMIT_RPM`、`AGENT_DAILY_TOKEN_BUDGET`、`AGENT_SESSION_TTL_DAYS`、`RAG_MAX_CONCURRENCY`、`RAG_QUEUE_TIMEOUT`。
- 新文件：`data/usage.db`（用量账）；sessions.db 有保留期；compose 拆 base/dev 两份；`deploy/build.sh`、`deploy/watchdog.sh`、`deploy/backup.sh`。

## 5. 决策记录

- D1 目标环境 = 内网单机（§0）。
- D2 鉴权放 agent 应用内：可进 pytest（本仓验收纪律），不引网关；要 TLS 时（公网）再叠反代。
- D3 预算闸 = 全局日额度，不按 key。
- D4 备份 = 冷备打包卷，分钟级停写可接受。
- D5 依赖不锁 lockfile，维持镜像即制品。
- D6 会话清理跑容器内（命名卷宿主不可见）。

## 6. 上线前置（非代码，人工勾）

1. LLM 换付费稳定源（免费/限流 key 不上线）。
2. 生成 `AGENT_API_KEYS` 并只发给使用者（不进聊天记录）。
3. 探活挂上宿主计划任务或 Uptime Kuma。
4. 备份挂上计划任务并演练过一次恢复。
