export const USAGE =
  "契约类型镜像：只按《接口文档》（docs/04_接口文档.md）§1.7 / §3.1 / §3.2 的形状声明，字段可缺省处一律 optional。";

export type Usage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type Evidence = {
  label: string;
  citation: string;
  text: string;
  score?: number | null;
};

export type ParentChunk = {
  parent_id?: string;
  law_id?: string;
  law_name?: string;
  version?: string;
  citation?: string;
  article_no?: string;
  article_index?: number;
  chapter?: string | null;
  section?: string | null;
  text?: string;
  refs?: string[];
};

export type RetrievedArticle = {
  parent_id?: string;
  citation?: string;
  article?: ParentChunk;
  score?: number | null;
  vector_rank?: number | null;
  bm25_rank?: number | null;
  vector_score?: number | null;
  bm25_score?: number | null;
  hit_chunks?: string[];
  law_hint?: string | null;
};

export type RetrievalResult = {
  query?: string;
  matched_text?: string;
  used_vector?: boolean;
  used_bm25?: boolean;
  elapsed_ms?: number;
  notes?: string[];
  articles?: RetrievedArticle[];
};

export type Review = {
  score?: number | null;
  threshold?: number;
  total?: number;
  supported?: number;
  unsupported?: string[];
  original_text?: string;
  model?: string;
  passed?: boolean;
};

export type AnswerPayload = {
  question?: string;
  answer?: string;
  model?: string;
  elapsed_ms?: number;
  usage?: Usage;
  notes?: string[];
  evidences?: Evidence[];
  citations?: string[];
  retrieval?: RetrievalResult | null;
  review?: Review | null;
  timeliness?: unknown[];
  materials?: unknown[];
};

export type DonePayload = AnswerPayload & {
  session_id?: string | null;
  request_ms?: number;
};

export type InterruptValue = {
  type?: string;
  place?: string;
  message?: string;
  laws?: string[];
};

export type InterruptPayload = {
  id?: string | null;
  value?: InterruptValue | null;
};

export type InterruptEvent = {
  session_id?: string | null;
  interrupt?: InterruptPayload | null;
};

export type StepEvent = {
  node: string;
  [key: string]: unknown;
};

export type DeltaEvent = { text?: string };

export type ErrorEvent = { message?: string; partial?: string };

export type StreamEvent =
  | { type: "step"; data: StepEvent }
  | { type: "delta"; data: DeltaEvent }
  | { type: "interrupt"; data: InterruptEvent }
  | { type: "done"; data: DonePayload }
  | { type: "error"; data: ErrorEvent };

export type QaRequest = {
  question: string;
  top_k?: number | null;
  doc_ids?: string[];
  session_id?: string | null;
};

export type ResumeRequest = {
  session_id: string;
  value: { region: string };
};

export type AskOk = AnswerPayload & {
  status: "ok";
  session_id?: string | null;
  request_ms?: number;
};

export type AskInterrupted = {
  status: "interrupted";
  session_id?: string | null;
  interrupt?: InterruptPayload | null;
  request_ms?: number;
};

export type AskResponse = AskOk | AskInterrupted;

export type RagHealth = {
  status?: string;
  version?: string;
  boot_ms?: number;
  milvus?: string;
  laws?: number;
  articles?: number;
  rows?: number;
  channels?: string;
  rerank?: string | false;
  dense_built?: boolean;
  index?: string;
  weights?: string;
  degraded?: boolean;
  llm_ready?: boolean;
  detail?: string;
};

export type AgentHealth = {
  status?: string;
  version?: string;
  rag?: RagHealth;
};

export type LawInfo = {
  law_id: string;
  law_name: string;
  version?: string;
  articles?: number;
  local?: boolean;
};

export type DocumentRecord = {
  doc_id: string;
  mode?: string;
  display_name?: string;
  note?: string;
  accepted?: boolean;
  [key: string]: unknown;
};

export type DocumentList = {
  count?: number;
  documents?: DocumentRecord[];
  modes?: Record<string, number> | string[];
};

export type ApiFailureKind = "http" | "network" | "parse";

export class ApiError extends Error {
  kind: ApiFailureKind;
  status: number;
  detail: string;
  retryAfter: number | null;
  body: unknown;

  constructor(init: {
    kind: ApiFailureKind;
    status?: number;
    detail: string;
    retryAfter?: number | null;
    body?: unknown;
  }) {
    super(init.detail);
    this.name = "ApiError";
    this.kind = init.kind;
    this.status = init.status ?? 0;
    this.detail = init.detail;
    this.retryAfter = init.retryAfter ?? null;
    this.body = init.body ?? null;
  }
}
