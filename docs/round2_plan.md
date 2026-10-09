# round2 计划：后端审查 × 前端 MVP（双会话并行）

本文件为本轮唯一真源，由协调者独占维护。docs 三件套（spec / prompt_plan / eval_plan）与 CLAUDE.md 仍全员禁改。口径来源：2026-10-09/10 用户与协调者的 grill 轮 Q1–Q11 全部裁定。

## 1. 两条工作流 + 协调者

**流 A · 后端审查**（只读）
- 工作区：主仓 `D:\projects\交通法规问答agent` 主工作树（干净，基线定钉提交 `5bbb210`）。
- 范围：六包 + deploy 面全量深审；只报增量（先读共享记忆与 README 自用台账里的「判过不动」清单）。
- 交付：六节报告（结论 / 主流程概览 / 问题清单 / 专项判断 / 建议主流程 / 待确认）；每条 = 级别 + file:line + 裁定建议（动 / 不动 + 理由）；不凑数；定级前先核异常边界。报告在会话内 + 落盘 `D:\projects\_tlq_review_round2.md`。
- 禁止写仓、跑变异枪、任何 LLM 与外网调用。

**流 B · 前端 MVP**
- 工作区：worktree `D:\projects\tlq-frontend`，分支 `feat-frontend`（基于 5bbb210）。
- 栈与落位：Vite + React + TypeScript；新增顶层 `frontend/`（第七顶层登记：只加不改六包拓扑）。
- MVP：①流式问答 ②澄清中断 / 恢复 ③依据展示 ④健康 / 降级状态条；M2（本轮后段）：⑤多轮历史 ⑥材料 / 附件。
- 口径：mock 先行（手搓 SSE 夹具，0 外呼）；联调等协调者信号；真实调用先报量；只碰 `frontend/**`。
- 验收：`npm run build` 通过 + 演示脚本（§4）逐条人工过。

**协调者**（本会话）
- 独占面：`docs/round2_plan.md`；`rag_contracts/config.py` + `.env.example`；`deploy/**`；`接口文档.md` / `README.md` / `运维手册.md`；`feat-cors` 的 agent_service 改动。
- feat-cors（worktree `D:\projects\tlq-cors`，分支 `feat-cors`）：仅 agent app 挂 CORS 中间件；`CORS_ORIGINS` 环境变量，默认 `http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173`；allow_headers 含 `Content-Type` 与 `X-API-Key`；方法 GET/POST/DELETE/OPTIONS；配套测试 + 变异枪 + 全量 pytest 落盘；`接口文档.md` 同步跨源一句。

## 2. 文件所有权矩阵

| 面 | 流A 审查 | 流B 前端 | 协调者 |
| --- | --- | --- | --- |
| `frontend/**` | 禁 | 独占 | 禁 |
| 六包后端 | 只读 | 禁 | 仅 feat-cors 的 CORS 相关改动 |
| `rag_contracts/config.py`、`.env.example` | 只读 | 禁 | 独占（单写者） |
| `deploy/**` | 只读 | 禁 | 独占 |
| `接口文档.md` / `README.md` / `运维手册.md` | 只读 | 禁 | 独占 |
| `tests/**` | 只读（可跑） | 禁 | 随 feat-cors 增量 |
| `docs/round2_plan.md` / 三件套 | 只读 | 只读 | round2_plan 独占 / 三件套禁改 |
| 共享记忆目录 | 只读 | 只读 | 登记入口 |

## 3. 里程碑

- **M0（现在）**：worktree 就位、计划与开卡落文、基线 pytest 验绿。
- **M1（并行）**：审查全量跑；前端 mock 全状态；协调者落 feat-cors。
- **M2（联调）**：用户明说 → 协调者合并 feat-cors + 直连 buildx 重建 agent 镜像 + force-recreate → 发「联调信号」（用户转达）→ 前端对 `http://127.0.0.1:8001` 联调（真实调用先报量）。
- **M3（收口）**：演示脚本全过；前端收工报告；审查报告 → 行动清单。
- **M4（可选后续）**：静态托管进 compose（nginx）；Playwright e2e；审查行动项修复。

## 4. 演示脚本 v0（前端会话可细化，四条覆盖须保留）

1. **普通问答（不沾地方）**：「机动车闯红灯一次记多少分？」→ 流式 delta 渐出、done 后替换为权威答案、依据条文列表可见。
2. **澄清恢复**：「深圳的电动自行车载人怎么规定的？」→ interrupt 帧出现 → 界面出地区选择 → 选后走 `/qa/resume/stream` → done。
3. **长答案流式**：任一长问题 → delta 连续出现且 done 帧完整替换（无拼接残影）。
4. **降级 / 错误面（本地可选演练）**：`LLM_API_KEY` 未配置态 → `/qa` 降级单轮检索、`/qa/resume` 返回 503「规划模型暂不可用，无法恢复被中断的会话；请直接重新提问。」；演练完必须还原 env。

## 5. 开工卡（开新会话时整段粘贴）

### 5A 给「后端审查流」

【后端审查流 · 开工卡】
你是本仓「后端审查流」的唯一 agent，只读模式，全程不写、不改仓库里的任何文件。
工作区：D:\projects\交通法规问答agent（主工作树，当前干净，基线定钉在提交 5bbb210；如仓库在审查期间前进，仍以 5bbb210 为准，可 `git show` 取任何文件的历史版本）。协调者（另一个会话）审查期间只会在本树新增 docs/round2_plan.md 与仓外日志，不动任何代码；如果你看到别的变动，先停下问用户。
先读（按序）：① 本轮唯一真源 D:\projects\交通法规问答agent\docs\round2_plan.md；②「判过不动」台账——必须读，防重复报告：C:\Users\陈淼森\.claude\projects\d--projects-------agent\memory\ 下全部记忆文件（大量已裁定项在里面）；③ README.md 末尾的自用台账一节；④ 接口文档.md、docs/02_设计说明书.md、docs/03_功能文档.md、deploy/运维手册.md。
任务：对六个 Python 包（rag_contracts / rag_service / agent_service / api_contracts / eval / tests）与 deploy 面做全量深审：架构一致性、正确性、并发与状态、安全面（鉴权、出站 egress、输入面）、测试与守卫质量、「文档声称 vs 代码实况」的不一致处。
报告形状（用户认可过的六节）：结论 / 主流程概览 / 问题清单 / 专项判断 / 建议主流程 / 待确认。问题清单每条 = 级别（P0/P1/P2/P3；宁缺毋滥、不凑数）+ file:line 证据 + 裁定建议（动 / 不动 + 理由）。定性定级前先核异常边界（try/except 在哪层、被谁兜成什么响应、有没有无效重试）；未核实的因果链措辞收紧、放进「待确认」。
允许：跑全量 pytest 作证据——本环境判据 = rc + 点阵里的 F/E（-q 无汇总行），输出整份落盘，例如：`cd /d/projects/交通法规问答agent && PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest -q > /d/projects/_tlq_review_round2_pytest.log 2>&1; echo rc=$?`；读任意文件；写仓外日志与报告文件。Python 一律用仓内 .venv/Scripts/python.exe。
禁止：写 / 改仓库任何文件、跑变异枪（本流不改代码）、任何 LLM 与外网调用、merge / 推送。
产出：会话内先给六节报告；全文落盘 D:\projects\_tlq_review_round2.md；结束时对用户说一句「审查报告好了」。

### 5B 给「前端 MVP 流」

【前端 MVP 流 · 开工卡】
你是本仓「前端 MVP 流」的唯一 agent。工作区：D:\projects\tlq-frontend（独立 git worktree，分支 feat-frontend，基于提交 5bbb210）。仓库本体与文档在 D:\projects\交通法规问答agent（只读参考）。
先读：① 本轮唯一真源 D:\projects\交通法规问答agent\docs\round2_plan.md（目标 / 边界 / 里程碑 / 验收 / 演示脚本草案都在里面）；② 仓根《接口文档.md》——契约，只按它写代码；重点：/qa/stream 是 POST，SSE 事件序为 step×N → delta×N → 之后 interrupt（出了它就没有 done，是终点；带 session_id，去 /qa/resume/stream 恢复）或 done（复核后的权威答案）或 error（收尾帧）；/qa 慢入口返回 status=ok/interrupted 两形状；/health 顶层 = status/version/rag，llm_ready/degraded/weights 在 rag 子字典内，boot 失败 = 503 detail。CLAUDE.md 已在你树里自动加载。共享记忆（如需）：C:\Users\陈淼森\.claude\projects\d--projects-------agent\memory\。
任务：在你的树里新建仓根顶层 `frontend/`（登记为第七顶层，不得动六个 Python 包的任何文件）。技术栈 Vite + React + TypeScript（本机 node v24.19.0、npm 11.17.0 已就绪，registry 已配 npmmirror，直连可用）。MVP 四件：①流式问答（delta 草稿流式渲染，done 到达后整体替换为权威答案，step 轨迹可见）；②地区澄清中断 → 界面出地区选择 → /qa/resume/stream 恢复；③依据 / 条文展示（法名 + 条号 + 摘录；复核降级要有提示）；④健康 / 降级状态条。/qa 的 400/409/429/503 与 error 帧都要有像样的界面态。M2（本轮后段再做）：⑤多轮会话历史；⑥材料 / 附件（接口文档已有 /documents 三个端点）。
开发口径：mock 先行——从接口文档手搓 SSE 事件夹具，覆盖上述全部帧与状态码，不调真实服务、0 次 LLM 外呼。等协调者的「联调信号」（用户转达）之后，才连 http://127.0.0.1:8001 真实容器联调；届时任何批量真实调用先向用户报量（几次 / 多久 / 量什么，并给一个不跑也能收口的备选）。认证头可选：只有 VITE_AGENT_API_KEY 设了才发 X-API-Key。
边界：只改 frontend/**；禁碰六包后端、deploy/、任何 .md 文档、共享记忆；需要后端口径或后端改动（比如 CORS 细节不符）→ 停下把问题列给用户转达协调者，不要自己发明。
文案：中文界面；不出现「demo」字样，用「服务」「示例」。
验收：npm run build 通过 + 演示脚本（计划文件 §4，可细化但保留四条覆盖）逐条人工过一遍。收工 = 报告（做了什么 / 怎么验的 / 遗留）+ 分支就绪；merge 与推送等用户明说。

## 6. 协作协议

- 转达点：① 前端遇后端口径 / 改动需求 → 列问题 → 用户转协调者裁定；② 审查报告完成 → 用户转协调者做行动清单；③ M2 联调信号由协调者发、用户转。
- 会话间不直连；协调介质 = 共享记忆 + git + 用户转达。
- merge / 推送 / 删除动作一律等用户明说；本文件暂未提交（协调者以小分支 `round2-plan` 落库）。
- 环境事实（2026-10-10 核）：node v24.19.0 / npm 11.17.0；registry = npmmirror（直连通）；两 app 现无 CORS；agent 鉴权 = `X-API-Key`、默认关；`/qa/stream` 为 POST；HEAD=`5bbb210`，领先 origin 45。
