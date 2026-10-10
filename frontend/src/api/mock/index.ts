import type { Settings } from "../../config";
import type { Transport } from "../transport";
import {
  ApiError,
  type AgentHealth,
  type AskOk,
  type AskResponse,
  type DocumentList,
  type DocumentRecord,
  type DonePayload,
  type QaRequest,
  type ResumeRequest,
  type StepEvent,
  type StreamEvent,
} from "../types";
import {
  CLARIFY_MESSAGE,
  contentOf,
  LAW_COUNT,
  LOCAL_LAWS,
  splitDeltas,
  stepAgent,
  stepClarify,
  stepFinalize,
  stepRegion,
  stepReview,
  stepTools,
  type CaseId,
} from "./corpus";

export const USAGE =
  "示例数据源：按《接口文档》手搓的 SSE 帧（step / delta / interrupt / done / error 全覆盖）与非 200 状态码，不连服务、不外呼。事件序照真实节点顺序：region → agent → tools → delta… → finalize → review → done。";

const DEGRADED_RESUME_DETAIL = "规划模型暂不可用，无法恢复被中断的会话；请直接重新提问。";

type Failure =
  | { mode: "http"; status: number; detail: string; retryAfter?: number }
  | { mode: "frame"; message: string; onceStatus: number; onceDetail: string };

export type Scenario = {
  id: string;
  label: string;
  hint: string;
  caseId: CaseId | "auto";
  clarify?: boolean;
  failure?: Failure;
  resumeFailure?: Failure;
  pace?: number;
};

export const SCENARIOS: Scenario[] = [
  { id: "auto", label: "自动识别", hint: "按问题关键词选场景；认不出时走「检索为空」", caseId: "auto" },
  { id: "plain", label: "普通问答", hint: "闯红灯记多少分：流式草稿 → 复核 → 权威答案", caseId: "red_light" },
  {
    id: "clarify",
    label: "地区澄清中断",
    hint: "深圳产的电动自行车上牌：判不出地区 → interrupt 帧 → 地区选择 → 恢复流",
    caseId: "shenzhen",
    clarify: true,
  },
  { id: "long", label: "长答案流式", hint: "醉驾处罚：连续 delta，done 帧整篇替换", caseId: "drunk", pace: 0.75 },
  {
    id: "mismatch",
    label: "复核未过降级",
    hint: "复核判「依据不足」：草稿被整篇换成降级答复，原稿收在可展开的详情里",
    caseId: "mismatch",
  },
  { id: "empty", label: "检索为空", hint: "没有命中法条：无依据、不生成、无复核", caseId: "empty" },
  {
    id: "degraded",
    label: "未配置规划模型",
    hint: "提问降级为单轮检索；先制造一个待澄清会话，再切到本场景点恢复，会收到 503 文案",
    caseId: "degraded",
    resumeFailure: { mode: "frame", message: DEGRADED_RESUME_DETAIL, onceStatus: 503, onceDetail: DEGRADED_RESUME_DETAIL },
  },
  {
    id: "stream_error",
    label: "流中途出错",
    hint: "已吐出的 delta 留在界面上，error 帧收尾",
    caseId: "red_light",
    failure: {
      mode: "frame",
      message: "Agent 未完成：检索服务超时（示例）",
      onceStatus: 500,
      onceDetail: "服务内部错误：检索服务超时（示例）",
    },
  },
  {
    id: "conflict409",
    label: "会话无等待中的澄清",
    hint: "流式面落在 error 帧，非流式面是 409",
    caseId: "red_light",
    failure: {
      mode: "frame",
      message: "Agent 未完成：这条会话没有等待澄清的地区问题（可能已经答完）",
      onceStatus: 409,
      onceDetail: "这条会话没有等待澄清的地区问题（可能已经答完）",
    },
  },
  {
    id: "ratelimit429",
    label: "限流",
    hint: "429 + Retry-After，开场前就拦下",
    caseId: "red_light",
    failure: { mode: "http", status: 429, detail: "请求过于频繁：每分钟上限 30 次", retryAfter: 42 },
  },
  {
    id: "quota429",
    label: "日额度用完",
    hint: "429 + Retry-After 到次日 UTC 0 点",
    caseId: "red_light",
    failure: {
      mode: "http",
      status: 429,
      detail: "日额度已用完：预算 500000 tokens，今日已用 500123 tokens；按 UTC 0 点重置",
      retryAfter: 3600,
    },
  },
  {
    id: "unready503",
    label: "服务未就绪",
    hint: "图形装配失败时 /qa 全 503",
    caseId: "red_light",
    failure: { mode: "http", status: 503, detail: "服务未就绪" },
  },
  {
    id: "unauthorized401",
    label: "鉴权失败",
    hint: "配了 key 但请求头不对",
    caseId: "red_light",
    failure: { mode: "http", status: 401, detail: "API key 缺失或无效" },
  },
  {
    id: "badrequest400",
    label: "会话存储不可用",
    hint: "带了 session_id 但服务没接会话存储",
    caseId: "red_light",
    failure: { mode: "http", status: 400, detail: "这个 runner 没接会话存储（sessions=None），不能带 session_id" },
  },
  {
    id: "malformed",
    label: "事件帧损坏",
    hint: "流里出现解析不了的帧",
    caseId: "red_light",
    failure: { mode: "frame", message: "__MALFORMED__", onceStatus: 500, onceDetail: "服务内部错误：示例" },
  },
];

export const HEALTH_SCENARIOS: { id: string; label: string; hint: string }[] = [
  { id: "ok", label: "正常", hint: "两路通道在走，生成层可用" },
  { id: "channel_degraded", label: "通道降级", hint: "稠密通道降级为纯 BM25，服务照常在答" },
  { id: "weights_mismatch", label: "权重不符", hint: "weights=mismatch，degraded=true" },
  { id: "no_llm", label: "生成层不可用", hint: "llm_ready=false，答问会拒答" },
  { id: "rag_unreachable", label: "知识库不可达", hint: "rag.status=unreachable" },
  { id: "boot_failed", label: "服务未就绪", hint: "/health 直接 503" },
  { id: "offline", label: "连不上", hint: "请求本身发不出去" },
];

function scenarioOf(id: string): Scenario {
  return SCENARIOS.find((row) => row.id === id) ?? SCENARIOS[0];
}

export function activeScenario(settings: Settings): Scenario {
  return scenarioOf(settings.scenario);
}

function caseOf(question: string): CaseId {
  if (/深圳|电动自行车|载人|地方/.test(question)) return "shenzhen";
  if (/醉驾|醉酒|酒驾|饮酒/.test(question)) return "drunk";
  if (/闯红灯|信号灯|记分|多少分|记多少/.test(question)) return "red_light";
  return "empty";
}

export function resolveCase(settings: Settings, question: string): CaseId {
  const scenario = activeScenario(settings);
  return scenario.caseId === "auto" ? caseOf(question) : scenario.caseId;
}

function hex32(): string {
  const bytes = new Uint32Array(4);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => value.toString(16).padStart(8, "0")).join("");
}

type MockEvent = StreamEvent | { type: "parse_failure"; data: { raw: string } };
type Beat = { delay: number; event: MockEvent };

const interrupted = new Map<string, { caseId: CaseId; question: string; place: string }>();

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("已取消", "AbortError"));
      return;
    }
    function onAbort() {
      window.clearTimeout(timer);
      reject(new DOMException("已取消", "AbortError"));
    }
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function* play(beats: Beat[], signal?: AbortSignal): AsyncGenerator<MockEvent> {
  for (const beat of beats) {
    await sleep(beat.delay, signal);
    yield beat.event;
  }
}

function donePayload(caseId: CaseId, question: string, sessionId: string): DonePayload {
  const content = contentOf(caseId);
  const answer = content.final;
  const skipModel = content.model === "(unavailable)" || content.model === "(skip)";
  return {
    question,
    answer,
    model: content.model,
    elapsed_ms: 4210.5,
    usage: skipModel ? {} : { prompt_tokens: 1820, completion_tokens: answer.length, total_tokens: 1820 + answer.length },
    notes: content.notes,
    evidences: content.evidences,
    citations: content.evidences.map((row) => row.citation),
    retrieval: {
      query: question,
      matched_text: question,
      used_vector: true,
      used_bm25: true,
      elapsed_ms: 212.4,
      notes: [],
      articles: content.articles,
    },
    review:
      content.review === null
        ? null
        : {
            score: content.review.score,
            threshold: content.review.threshold,
            total: content.review.total,
            supported: content.review.supported,
            unsupported: content.review.unsupported,
            original_text: content.draft,
            passed: content.review.passed,
            model: "qwen-flash",
          },
    timeliness: [],
    materials: [],
    session_id: sessionId,
    request_ms: 4380.2,
  };
}

function pacing(pace: number) {
  return (delay: number): number => Math.round(delay * pace);
}

function askBeats(settings: Settings, question: string, sessionId: string): Beat[] {
  const scenario = activeScenario(settings);
  const pace = pacing(scenario.pace ?? 1);
  const caseId = resolveCase(settings, question);
  const local = caseId === "shenzhen" && !scenario.clarify;
  const beats: Beat[] = [
    { delay: pace(340), event: { type: "step", data: stepRegion(local ? "深圳" : null, local ? LOCAL_LAWS.length : 0) } },
  ];
  if (scenario.clarify) {
    interrupted.set(sessionId, { caseId: "shenzhen", question, place: "深圳" });
    beats.push({
      delay: pace(680),
      event: {
        type: "interrupt",
        data: {
          session_id: sessionId,
          interrupt: {
            id: `interrupt-${sessionId.slice(0, 8)}`,
            value: { type: "region_clarify", place: "深圳", message: CLARIFY_MESSAGE, laws: LOCAL_LAWS },
          },
        },
      },
    });
    return beats;
  }
  if (scenario.failure?.mode === "frame") {
    const pieces = splitDeltas(contentOf(caseId).draft);
    const cut = Math.max(1, Math.floor(pieces.length / 3));
    for (const piece of pieces.slice(0, cut)) {
      beats.push({ delay: pace(22), event: { type: "delta", data: { text: piece } } });
    }
    beats.push({
      delay: pace(500),
      event:
        scenario.failure.message === "__MALFORMED__"
          ? { type: "parse_failure", data: { raw: "event: delta  data: {oops" } }
          : { type: "error", data: { message: scenario.failure.message, partial: pieces.slice(0, cut).join("") } },
    });
    return beats;
  }
  beats.push(...answerBeats(caseId, question, sessionId, pace));
  return beats;
}

function answerBeats(caseId: CaseId, question: string, sessionId: string, pace: (ms: number) => number): Beat[] {
  const content = contentOf(caseId);
  const beats: Beat[] = [];
  const push = (delay: number, event: MockEvent) => beats.push({ delay: pace(delay), event });
  const step = (delay: number, data: StepEvent) => push(delay, { type: "step", data });
  step(420, stepAgent(1, 6, ["search_law"]));
  step(520, stepTools(1, Math.max(content.articles.length, 1), 0, 0));
  if (content.articles.length > 1) {
    step(380, stepAgent(2, 6, []));
  }
  for (const piece of splitDeltas(content.draft)) {
    push(22, { type: "delta", data: { text: piece } });
  }
  step(460, stepFinalize(content.evidences.length, content.fallback));
  step(520, stepReview(content.review));
  push(200, { type: "done", data: donePayload(caseId, question, sessionId) });
  return beats;
}

function resumeBeats(settings: Settings, req: ResumeRequest): Beat[] {
  const scenario = activeScenario(settings);
  const pace = pacing(scenario.pace ?? 1);
  const known = interrupted.get(req.session_id);
  const failure = scenario.resumeFailure ?? scenario.failure;
  if (failure?.mode === "frame") {
    return [
      {
        delay: pace(420),
        event:
          failure.message === "__MALFORMED__"
            ? { type: "parse_failure", data: { raw: "event: done  data: {oops" } }
            : { type: "error", data: { message: failure.message } },
      },
    ];
  }
  const caseId: CaseId = /深圳/.test(req.value.region) ? "shenzhen" : "shenzhen_national";
  const question = known?.question ?? "（示例：被中断的问题）";
  interrupted.delete(req.session_id);
  const content = contentOf(caseId);
  const beats: Beat[] = [];
  const push = (delay: number, event: MockEvent) => beats.push({ delay: pace(delay), event });
  const step = (delay: number, data: StepEvent) => push(delay, { type: "step", data });
  step(360, stepClarify(req.value.region, LOCAL_LAWS.length));
  step(420, stepAgent(1, 6, ["search_law"]));
  step(520, stepTools(1, content.articles.length, 0, 0));
  step(380, stepAgent(2, 6, []));
  for (const piece of splitDeltas(content.draft)) {
    push(22, { type: "delta", data: { text: piece } });
  }
  step(460, stepFinalize(content.evidences.length, content.fallback));
  step(520, stepReview(content.review));
  push(200, { type: "done", data: donePayload(caseId, question, req.session_id) });
  return beats;
}

function healthFixture(id: string): AgentHealth {
  const ragBase = {
    version: "0.1.0",
    boot_ms: 812.3,
    milvus: "v2.6.24",
    laws: LAW_COUNT,
    articles: 656,
    rows: 966,
    rerank: "bge-reranker-v2-m3",
    dense_built: true,
    index: "reuse",
  };
  if (id === "channel_degraded") {
    return {
      status: "ok",
      version: "0.1.0",
      rag: { ...ragBase, status: "ok", channels: "纯 BM25", weights: "ok", degraded: false, llm_ready: true },
    };
  }
  if (id === "weights_mismatch") {
    return {
      status: "ok",
      version: "0.1.0",
      rag: { ...ragBase, status: "ok", channels: "稠密+BM25", weights: "mismatch", degraded: true, llm_ready: true },
    };
  }
  if (id === "no_llm") {
    return {
      status: "ok",
      version: "0.1.0",
      rag: { ...ragBase, status: "ok", channels: "稠密+BM25", weights: "ok", degraded: false, llm_ready: false },
    };
  }
  if (id === "rag_unreachable") {
    return {
      status: "degraded",
      version: "0.1.0",
      rag: { status: "unreachable", detail: "无法连接知识库服务：Connection refused（示例）" },
    };
  }
  return {
    status: "ok",
    version: "0.1.0",
    rag: { ...ragBase, status: "ok", channels: "稠密+BM25", weights: "ok", degraded: false, llm_ready: true },
  };
}

export function createMockTransport(getSettings: () => Settings): Transport {
  const documents: DocumentRecord[] = [
    { doc_id: "示例材料-1", mode: "session", display_name: "现场记录.txt", note: "会话材料：仅本次会话可检索" },
    { doc_id: "示例材料-2", mode: "permanent", display_name: "地方补充规定.docx", note: "永久入库：重建索引后可进依据链" },
  ];

  function httpFailure(failure: Failure | undefined): ApiError | null {
    if (!failure) return null;
    if (failure.mode === "http") {
      return new ApiError({
        kind: "http",
        status: failure.status,
        detail: failure.detail,
        retryAfter: failure.retryAfter ?? null,
      });
    }
    return null;
  }

  async function* stream(beats: Beat[], failure: Failure | undefined, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
    const http = httpFailure(failure);
    if (http) {
      await sleep(240, signal);
      throw http;
    }
    await sleep(160, signal);
    for await (const event of play(beats, signal)) {
      if (event.type === "parse_failure") {
        throw new ApiError({
          kind: "parse",
          detail: `事件帧不是合法 JSON（示例）：${event.data.raw}`,
          body: event.data.raw,
        });
      }
      yield event;
    }
  }

  function onceFailure(failure: Failure | undefined): ApiError | null {
    if (!failure) return null;
    if (failure.mode === "http") {
      return new ApiError({
        kind: "http",
        status: failure.status,
        detail: failure.detail,
        retryAfter: failure.retryAfter ?? null,
      });
    }
    return new ApiError({ kind: "http", status: failure.onceStatus, detail: failure.onceDetail });
  }

  return {
    async *ask(req: QaRequest, signal?: AbortSignal) {
      const settings = getSettings();
      const sessionId = req.session_id || hex32();
      yield* stream(askBeats(settings, req.question, sessionId), activeScenario(settings).failure, signal);
    },
    async *resume(req: ResumeRequest, signal?: AbortSignal) {
      const settings = getSettings();
      const scenario = activeScenario(settings);
      yield* stream(resumeBeats(settings, req), scenario.resumeFailure ?? scenario.failure, signal);
    },
    async askOnce(req: QaRequest, signal?: AbortSignal) {
      const settings = getSettings();
      const scenario = activeScenario(settings);
      const failure = onceFailure(scenario.failure);
      await sleep(900, signal);
      if (failure) throw failure;
      const sessionId = req.session_id || hex32();
      if (scenario.clarify) {
        interrupted.set(sessionId, { caseId: "shenzhen", question: req.question, place: "深圳" });
        const response: AskResponse = {
          status: "interrupted",
          session_id: sessionId,
          interrupt: {
            id: `interrupt-${sessionId.slice(0, 8)}`,
            value: { type: "region_clarify", place: "深圳", message: CLARIFY_MESSAGE, laws: LOCAL_LAWS },
          },
          request_ms: 2140.6,
        };
        return response;
      }
      return { status: "ok", ...donePayload(resolveCase(settings, req.question), req.question, sessionId) } as AskOk;
    },
    async resumeOnce(req: ResumeRequest, signal?: AbortSignal) {
      const settings = getSettings();
      const scenario = activeScenario(settings);
      const failure = onceFailure(scenario.resumeFailure ?? scenario.failure);
      await sleep(900, signal);
      if (failure) throw failure;
      const known = interrupted.get(req.session_id);
      if (!known) {
        throw new ApiError({ kind: "http", status: 409, detail: "这条会话没有等待澄清的地区问题（可能已经答完）" });
      }
      interrupted.delete(req.session_id);
      const caseId: CaseId = /深圳/.test(req.value.region) ? "shenzhen" : "shenzhen_national";
      return { status: "ok", ...donePayload(caseId, known.question, req.session_id) } as AskOk;
    },
    async health(signal?: AbortSignal) {
      const id = getSettings().healthScenario || "ok";
      await sleep(180, signal);
      if (id === "offline") {
        throw new ApiError({ kind: "network", detail: "无法连接服务：Failed to fetch（示例）" });
      }
      if (id === "boot_failed") {
        throw new ApiError({
          kind: "http",
          status: 503,
          detail: "启动失败：无法连接知识库服务：Connection refused（示例）",
        });
      }
      return healthFixture(id);
    },
    async listDocuments(mode?: string | null, signal?: AbortSignal) {
      await sleep(220, signal);
      const rows = mode ? documents.filter((row) => row.mode === mode) : [...documents];
      const list: DocumentList = { count: rows.length, documents: rows, modes: { session: 1, permanent: 1 } };
      return list;
    },
    async uploadDocument(file: File, mode: string, signal?: AbortSignal) {
      await sleep(420, signal);
      const name = file.name || "未命名";
      const suffix = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1).toLowerCase() : "";
      if (!["docx", "md", "txt", "pdf"].includes(suffix)) {
        throw new ApiError({
          kind: "http",
          status: 400,
          detail: "文件类型不支持（示例数据源同样按这条口径拒收）",
          body: { accepted: false, note: "只收 docx / md / txt / pdf" },
        });
      }
      if (mode === "permanent") {
        throw new ApiError({
          kind: "http",
          status: 400,
          detail: "示例数据源不接收永久入库",
          body: { accepted: false, note: "永久入库要重建索引，请在真实服务上操作" },
        });
      }
      const record: DocumentRecord = {
        accepted: true,
        doc_id: `${name}-${hex32().slice(0, 6)}`,
        mode: "session",
        display_name: name,
        note: "会话材料：仅本次会话可检索，引用标 [材料N]",
      };
      documents.push(record);
      return record;
    },
    async deleteDocument(docId: string, signal?: AbortSignal) {
      await sleep(240, signal);
      const index = documents.findIndex((row) => row.doc_id === docId);
      if (index < 0) {
        throw new ApiError({ kind: "http", status: 404, detail: "没有这个 doc_id" });
      }
      if (documents[index].mode === "permanent") {
        throw new ApiError({
          kind: "http",
          status: 400,
          detail: "永久入库的法规不能从这里删",
          body: { note: "手工步骤：移出 法规知识库/docx/ 后重建索引" },
        });
      }
      documents.splice(index, 1);
      return { ok: true, doc_id: docId };
    },
  };
}
