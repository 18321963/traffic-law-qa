# 生产化验收设计

> 配套 `spec.md`（需求）与 `prompt_plan.md`（分工）。三层判据：L1 单元/契约 → L2 服务级 → L3 演练。
> 通用口径：pytest 判据 = rc + 点阵里的 F/E（本环境无汇总行），输出整份落盘；改行为面必配变异枪（仓外脚本），枪红 + 还原 + S0 快照 diff 无残。

## 1. L1 单元/契约（每流在收工清单里自带）

- 各流新用例见 `prompt_plan.md` 卡片「测试」节；枪数 ≥2。
- 全量回归基线：`_tlq_prod_<流>.log` 里 0 F/0 E 且 rc=0。
- 特别项：S2 的 uuid6/`delete_thread`/计数三验、W0 的 worktree 两问探针（import 解析 + pytest 真跑在 worktree 代码上）——探针失败即停、报告，不许带病并行。

## 2. L2 服务级验收（W3 集成时逐条跑，窗口内起栈）

约定：`$H=http://127.0.0.1:8001`（agent）；中文请求体落盘再 `--data-binary @file`（argv 会变 GBK）。

| # | 检查 | 预期 |
|---|---|---|
| 1 | 未配 `AGENT_API_KEYS` 起栈，`POST /qa` 不带 key | 200，done 帧（默认关，向后兼容） |
| 2 | 配 key 重启，`/qa` 不带头 | 401 `{"detail": "API key 缺失或无效"}` |
| 3 | 带正确头 `X-API-Key` | 200 且 done 帧含 answer/session_id/request_ms |
| 4 | `GET /health` 不带 key | 200 |
| 5 | `AGENT_RATE_LIMIT_RPM=3`，连打 5 次 | 前 3 次 200，第 4 次起 429 且带 `Retry-After` |
| 6 | `AGENT_DAILY_TOKEN_BUDGET` 设小值（如 800），连问 2 次 | 第 2 次 429；`data/usage.db` 当日行 tokens>0 |
| 7 | 预算关（0）重跑 | 不再 429；usage 行仍在累积 |
| 8 | `RAG_MAX_CONCURRENCY=1` + 慢问题并发 2 路 | 一路 200、一路排队或 503（带 `Retry-After: 5`）；无 500 |
| 9 | `docker compose -f 生产面 config` 渲染 | 输出无 19530/9000/8000 映射 |
| 10 | S2 清理：`session_cleanup` dry-run（在线 exec）记数 → 停机 `--apply`（stop → run --rm → start）→ 再看行数 | 差值 = dry-run 报的可删数；未过期线程仍在 |
| 11 | `docker compose exec agent python -m agent_service.agents.session_cleanup`（dry-run）与 `docker compose run --rm agent ... --apply`（停机） | 均可执行（镜像已含新代码） |
| 12 | 流式面排队：占满 `RAG_MAX_CONCURRENCY=1` 后打 `/answer/stream` | HTTP 200 + `event: error` 帧（不是 503） |
| 13 | `budget>0` 时走一次 clarify 中断（interrupt 帧后不发 resume） | usage.db 当日行已累加（中断也记账） |

## 3. 端到端冒烟（部署后固定三问）

1. **单跳**：直问一条文（如「醉驾怎么处罚」）→ done 帧答案含条号引用。
2. **会话续问**：带同一 `session_id` 追问（如「那罚款呢」）→ done 帧正常、`/qa` 语义连贯。
3. **clarify 中断流**（`AGENT_CLARIFY=1` 时）：给模糊题 → 出 `interrupt` 帧 → `/qa/resume` 带 region → done 帧完成。
4. 期间核对 `data/usage.db` 的 calls 增量 > 0（成本面在记账）。注意现役模型第 1 轮不发工具调用，每题多付一次调用——记账数字偏大属预期。

## 4. 量法（怎么量、记什么）

- **并发**：用同一脚本对 `/qa` 打不同问题（别用同一题），`RAG_MAX_CONCURRENCY` 取 0 vs 4 各跑一轮，记 p50/p95 与错误率；判据 = 开闸后 p95 不劣于关闸空闲口径的 3 倍（宽松线，防伪精确）。
- **限流/预算**：参数设小值（RPM=3、预算=800）验证语义，不测真实量级。
- **watchdog 对拍**：停 rag 容器 → `watchdog.sh` 退出非零且出告警行；`WATCHDOG_WEBHOOK_URL` 指向本地假接收端可验 POST 体。
- 全部输出落盘 `D:\projects\_tlq_prod_L2_*.log`。

## 5. 恢复演练剧本（R8）

1. `deploy/backup.sh` 打一份备份（自动停写→打包→起栈），记产物目录。
2. 停栈，把 `volumes/milvus` 改名藏起（模拟丢卷），再停栈一次清干净。
3. `deploy/backup.sh --restore <产物目录>` → 起栈。
4. 验收：两服务 `/health` 全绿；冒烟第 1 问答案含条号（语料可用）；sessions 卷回来（带原 session_id 追问命中历史）。
5. 复原 `--keep` 口径：连打 N+1 份，最老一份被清。

## 6. 记录与留档

- 日志一律仓外 `D:\projects\_tlq_prod_*.log`；变异脚本 `_tlq_prod_mut_*.py`。
- W3 报告格式：每条 L2/L3 一行「# 结果 证据文件」；附全量 pytest rc/F/E 与未推 commit 数。

## 7. 不测面（明说）

- 告警送达端到端（webhook 真通知渠道）——人工确认一条即可。
- TLS / 公网攻击面 / IP 级限流——本轮非目标（spec §0）。
- LLM 供应商故障恢复——靠既有回执链口径，不另造演练。
- 答案质量——本轮不重跑 eval（花钱，另报量）。
