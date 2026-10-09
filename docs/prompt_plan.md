# 生产化实现计划（多 agent 并行）

> 规格 = `spec.md`；验收 = `eval_plan.md`；房规 = 根目录 CLAUDE.md。本文件是**唯一的任务分配面**。
> 一条工作流 = 一张卡片 = 一条分支 = 一个 agent；agent 只改卡片 own 面。

## 0. 怎么开干

- **起手语**（每个 agent 的 prompt 模板）：
  「读 CLAUDE.md、docs/spec.md、docs/prompt_plan.md；执行 <卡号>；分支 <分支名>；收工按第 5 节清单走；只改卡片 own 面的文件。」
- 每个 agent 在自己的 worktree / 克隆里工作。**worktree 里跑测试必须**：
  `cd <worktree> && D:\projects\交通法规问答agent\.venv\Scripts\python.exe -m pytest`
  （`-m` 把 cwd 置顶，测的是 worktree 的代码；直接调 `.venv` 里的 pytest 会把测试跑回主仓——W0 用探针验证，探针不过就不许并行）。
- 波次：W0 串行打地基 → W1 四路并行（S1–S4）→ W2 两路（S5、S6）→ W3 集成收口。
- 合并：W3 按 S1→S2→S3→S4→S5→S6 顺序 merge 进 main（每条一个中文 `merge: 合入 <分支>（摘要）`）；冲突谁合谁解；**推送等用户明说**。
- **W3 在主仓、串行、独占**：它直接在 main 上收且要真起栈跑 L2/L3——开工前等所有流收工，期间不许别的 agent 动工作区。
- 每流提交用中文信息 `-F` 落盘（不塞 argv）；接口/手册的「草稿块」放在该流提交信息末尾，W3 统一落笔。

## 1. 波次表

| 波次 | 流 | 分支 | 依赖 | 说明 |
|---|---|---|---|---|
| W0 | 地基 | `prod-w0-config` | — | 配置键一次加齐 + worktree 测试隔离探针 |
| W1 | S1 鉴权+限流 | `prod-auth-ratelimit` | W0 | R1+R2 |
| W1 | S2 会话保留 | `prod-session-retention` | W0 | R6 |
| W1 | S3 检索并发闸 | `prod-rag-concurrency` | W0 | R4 |
| W1 | S4 部署面 | `prod-deploy-face` | — | R5+R7+R9+R10 |
| W2 | S5 成本闸 | `prod-budget-gate` | S1 已合 | R3（app.py 触点排在 S1 后） |
| W2 | S6 备份 | `prod-backup` | S4 已合 | R8 |
| W3 | 集成 | 直接在 main 上收（主仓、串行、独占） | 全部 | 合并 + 接口文档/README/手册落笔 + 端到端演练 |

## 2. 单写者文件表（别人不许碰）

| 文件 | 写者 |
|---|---|
| `rag_contracts/config.py`、`.env.example` | W0（一次加齐全部键） |
| `deploy/docker-compose.yml`、`deploy/docker-compose.dev.yml` | S4 |
| `接口文档.md`、`README.md`、`deploy/运维手册.md` | W3（各流只交草稿块） |
| `tests/conftest.py`、`pyproject.toml` | 默认不动；确需共享夹具/依赖 → 提请 W0 |
| `CLAUDE.md`、`docs/*.md`（本三件套） | 谁都不改 |

## 3. 工作流卡片

### W0 地基（串行）

- own：`rag_contracts/config.py`、`.env.example`。
- 做：按 spec R1–R6 加齐 6 个键与解析（沿用 `_env*` 助手；keys 解析成 tuple）：
  1. `AGENT_API_KEYS`（str→tuple，默认空）
  2. `AGENT_RATE_LIMIT_RPM`（int，30）
  3. `AGENT_DAILY_TOKEN_BUDGET`（int，0）
  4. `AGENT_SESSION_TTL_DAYS`（int，30）
  5. `RAG_MAX_CONCURRENCY`（int，4）
  6. `RAG_QUEUE_TIMEOUT`（float，30.0）
  `.env.example` 对应加注释行。默认值一律「关/保守」——既有测试必须零改动全绿。
- 探针（两问，落盘 `D:\projects\_tlq_prod_w0_probe.log`）：① import 解析——建临时 worktree，里面打印 `agent_service.__file__`，断言路径含 worktree 名；② pytest 运行面——在 worktree 里建临时用例 `tests/test___probe_worktree.py`（断言 `agent_service.__file__` 含 worktree 名），`cd <worktree> && 主仓解释器 -m pytest tests/test___probe_worktree.py -q` 看绿 → 把断言改成不可能值再看红 → 删临时文件。绿→红→删齐全才算通过（只验 import 不够：得证明单测真跑在 worktree 代码上）。
- 完成判据：全量 pytest rc=0；探针两问全过。

### S1 鉴权 + 限流（R1+R2）

- own：`agent_service/api/auth.py`(新)、`agent_service/api/app.py`、`tests/test_agent_auth.py`(新)。禁改 config/conftest/接口文档/compose。
- 规格：见 spec R1/R2（中间件实现，流式端点也要拦在生成器之前；`/health`、`/laws` 豁免）。
- 测试：开关关 = 既有测试全绿即证；开：无 key→401 / 错 key→401 / 对 key→200 / RPM=3 打 5 次→第 4 次起 429 带 `Retry-After` / `/health` 无 key 可达。
- 枪 ≥2：删鉴权判断、删限流判断 → 各红对应用例，跑完全量 pytest 再还原。
- 草稿块：接口文档「鉴权」「限流」小节（`X-API-Key` 行 + 401/429 示例）。

### S2 会话保留清理（R6）

- own：`agent_service/agents/session.py`、`agent_service/agents/session_cleanup.py`(新)、`tests/test_agent_session_cleanup.py`(新)。
- 规格：见 spec R6。**探针三验，任一不成立就停下报告、别硬上**：① uuid6——用真 SqliteSaver 建 checkpoint，解析 `checkpoint_id` 的第一个时间分量断言 ≈ 现在；② `delete_thread` 可达——现役版本存在该方法，删后 `get_tuple` 返 None（不成立改直删 SQL 兜底，但走同一把锁/同一连接，不许另开裸连接）；③ 计数一致——dry-run 报的可删数 = apply 后实际消失的线程数。
- 测试：dry-run 不动库（行数不变）；apply 后过期删/未过期留；`--ttl 0` 直接退出；汇总行可被 grep。
- 枪 ≥2：关 TTL 判断、dry-run 变直删。
- 草稿块：手册「会话清理」节（**dry-run 在线 exec；`--apply` 停机**：stop → run --rm → start 三条命令 + 计划任务示例 + 镜像需重建）。

### S3 检索并发闸（R4）

- own：`rag_service/api/app.py`、`tests/test_rag_concurrency.py`(新)。
- 规格：见 spec R4（`threading.BoundedSemaphore`，sync 端点在线程池里跑；闸只包住列出的路径）。**释放点按 spec 写死**：sync 包整个处理函数；流式在生成器内 acquire、finally release（不许在返回 StreamingResponse 之前 acquire），流式面排队超时走 `event: error` 帧（HTTP 200）。
- 测试：用慢桩占住信号量→后续请求排队、等待超时→503 带 `Retry-After`（流式面同场景→`error` 帧）；`RAG_MAX_CONCURRENCY=0` 关闸无感；流式客户端中途断开后信号量计数回满（拿内部计数或后一发请求的 200 证明）；文案含中文提示。
- 枪 ≥2：闸失效（直接放行）、超时分支反转。
- 注意：不改 agent 侧——503 → QaError →「检索失败」回执是既有行为，口径刚好自洽。

### S4 部署面（R5+R7+R9+R10）

- own：`deploy/docker-compose.yml`、`deploy/docker-compose.dev.yml`(新)、`deploy/build.sh`(新)、`deploy/watchdog.sh`(新)、`.github/workflows/test.yml`(新)。
- 规格：见 spec R5/R7/R9/R10。healthcheck / restart / GPU 配置原样保留；`deploy/reinstall.sh` 不动。
- CI 兜底写死（谁判=S4 自己，判完在提交信息里记一行）：`.[all]` 若 CI 首次安装 >10 分钟，改成 `pip install -e ".[api,dev,agent,mcp,milvus,pdf]"` + `torch --index-url https://download.pytorch.org/whl/cpu`（依赖 local 的用例按 importorskip 跳过即接受）。
- 验收：`docker compose -f base config` 渲染通过且无 19530/9000 端口；`docker compose -f base -f dev config` 恢复端口；`bash -n` 两脚本；watchdog 对拍（停 rag → 退出非零/出告警行）。
- 枪：脚本面不强求 pytest 枪，判据 = `bash -n` + eval_plan 的 L2/L3 演练。
- 草稿块：手册「部署」「告警」两节 + dev 覆盖用法。

### S5 成本闸（R3；等 S1 合入后开）

- own：`agent_service/api/budget.py`(新)、`agent_service/api/app.py`、`agent_service/agents/graph.py`、`tests/test_agent_budget.py`(新)。
- 规格：见 spec R3。**记账点写死**：run 出口（done / 中断 / 异常三路）统一取最新状态快照的 `usage` 通道**全部事件求和**（先核 usage 字典键名；一轮里每次 LLM 调用都进通道，不是取最后一次）；usage.db 开 `journal_mode=WAL` + `busy_timeout=3000`；写失败 stderr 固定前缀 `[usage.db] 记账失败`、不拦请求；另给 `--show`（打印当日行，读失败非零退出）供巡检。前置检查在 app 层，流式同样拦。
- 测试：预算关无感；tmp 目录建库；预算设常数→第二问 429；记账数值 = 注入假 usage 之和（一轮放 ≥2 条事件，证明是求和不是取末）；中断出口、异常出口各一条记账用例；db 不可写不炸请求；`--show` 读坏文件非零退出。
- 枪 ≥2：删前置检查、删记账。
- 草稿块：接口文档「日额度」小节 + `data/usage.db` 文件行；手册两行（异常 run 可能少记最后一步的口径 + 预算开启时 watchdog 折 `budget --show` 巡检）。

### S6 备份恢复（R8；等 S4 合入后开）

- own：`deploy/backup.sh`(新)。
- 规格：见 spec R8（bind 卷宿主 tar + 命名卷 helper 容器 tar；`--keep`/`--restore`）。
- 验收：`bash -n` + eval_plan §5 恢复演练剧本跑一遍（真停栈真恢复）。
- 草稿块：手册「备份/恢复」节（口径：冷备 = 分钟级停写；index/chunks 不备份可重建）。

### W3 集成（一个 agent 收口；主仓、串行、独占）

1. 开工前提：六条分支全部收工；期间不许任何并行流动工作区（L2/L3 要真起栈，环境就是主仓 + 本机 Docker，谁在跑 W3 别人就等）。
2. 按序合并六条分支（S1→S2→S3→S4→S5→S6），每条一个 merge 提交，中文 `-F`。
3. 落笔：`接口文档.md`（spec §4 全部条目）、`deploy/运维手册.md`（汇总各流草稿块 + spec R11 口径）、README 仅按 R11 最小同步（先 `grep -rn "README「"`）。
4. 验收：全量 pytest 落盘 + eval_plan 的 L1 全表 + L2 服务级验收 + §3 端到端冒烟（本机 build.sh + dev 覆盖起栈）。
5. 收尾：核对 `git status` 面；回报时附「未推 commit 数」。

## 4. 冲突规则

- 两流撞同一文件 = 计划错误：停下来先报，不许「顺手」合并改动。
- 接口文档/README 的中文冲突一律 W3 重写，不 cherry-pick 片段。

## 5. 每流收工清单（五步）

1. 全量 pytest：`... -m pytest -q` 整份落盘 `D:\projects\_tlq_prod_<流>.log`；判据 = rc + 点阵里的 F/E（本环境没有汇总行）。
2. 改到行为面 → 变异枪：脚本放仓外 `D:\projects\_tlq_prod_mut_<流>.py`；每枪必红、逐枪还原；跑完 diff S0 快照防残。
3. `git status -s` / `git diff --stat` 面核对：只含卡片 own 的文件。
4. 提交：中文信息 `-F` 落盘；接口/手册草稿块贴提交信息末尾。
5. 不推送；报告「分支名 + rc + 枪果 + 面核对」。

## 6. 红线

- 不碰六包拓扑；不加注释/docstring（房规）；不删任何既有文件（要用户在本人消息里点名）。
- 不动既有测试断言语义；`tests/conftest.py` 默认不改。
- 不跑自费的批量 LLM 外呼；一切脚本走 `.venv/Scripts/python.exe`；中文不进 Git Bash argv。
- 不在 main 上直接改（W3 除外）；不 prune Docker、不全量 eval 重跑。
