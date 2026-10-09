import type { DonePayload, Evidence, Review, StepEvent } from "./api/types";

export const USAGE =
  "展示口径：答案正文里的 [依据N] / [时效N] / [材料N] 切成可点的引用片；复核三态（通过 / 未通过 / 未复核）与步骤中文名都从这里出。";

export type Segment = { kind: "text"; text: string } | { kind: "ref"; label: string; group: string; index: number };

const REF_PATTERN = /(\[|【)(依据|时效|材料)(\d+)(\]|】)/g;

export function segmentsOf(text: string): Segment[] {
  const segments: Segment[] = [];
  let cursor = 0;
  for (const match of text.matchAll(REF_PATTERN)) {
    const [open, group, digits, close] = [match[1], match[2], match[3], match[4]];
    if ((open === "[" && close !== "]") || (open === "【" && close !== "】")) continue;
    const start = match.index ?? 0;
    if (start > cursor) segments.push({ kind: "text", text: text.slice(cursor, start) });
    segments.push({ kind: "ref", label: `[${group}${digits}]`, group, index: Number(digits) });
    cursor = start + match[0].length;
  }
  if (cursor < text.length) segments.push({ kind: "text", text: text.slice(cursor) });
  return segments;
}

export function paragraphsOf(text: string): string[] {
  return text.split(/\n{2,}/).filter((row) => row.trim() !== "");
}

export type ReviewState = {
  tier: "passed" | "failed" | "unfinished" | "unscored" | "none";
  label: string;
  detail: string;
};

export function reviewStateOf(answer: DonePayload | null): ReviewState {
  if (!answer) return { tier: "none", label: "未复核", detail: "" };
  const review: Review | null | undefined = answer.review;
  if (review) {
    const score = typeof review.score === "number" ? review.score.toFixed(2) : "—";
    const threshold = typeof review.threshold === "number" ? review.threshold.toFixed(2) : "—";
    const head = `复核 ${score} / 阈值 ${threshold}`;
    if (review.passed === false) {
      const unsupported = review.unsupported ?? [];
      const detail =
        unsupported.length > 0 ? `${head} · 复核指出的问题：${unsupported.join("；")}` : `${head} · 复核给了否定结论`;
      return { tier: "failed", label: "复核未过", detail };
    }
    return {
      tier: "passed",
      label: "已复核",
      detail: `${head} · ${review.supported ?? 0}/${review.total ?? 0} 条依据在答案里有支撑`,
    };
  }
  const notes = answer.notes ?? [];
  const unfinished = notes.find((row) => row.startsWith("复核未完成"));
  if (unfinished) return { tier: "unfinished", label: "本次未复核", detail: unfinished };
  const unscored = notes.find((row) => row.startsWith("复核未打分"));
  if (unscored) return { tier: "unscored", label: "本次未打分", detail: unscored };
  return { tier: "none", label: "未复核", detail: "答案里没有引用依据，复核没有可判的据" };
}

const NODE_LABELS: Record<string, string> = {
  region: "识别地区",
  clarify: "地区澄清",
  agent: "规划",
  tools: "检索",
  finalize: "生成",
  review: "复核",
};

export function nodeLabel(node: string): string {
  return NODE_LABELS[node] ?? node;
}

function number(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function stepDetail(step: StepEvent): string {
  const row = step as Record<string, unknown>;
  if (step.node === "region") {
    const region = typeof row.region === "string" && row.region ? row.region : null;
    const laws = number(row.laws);
    return region ? `地区：${region} · 库内地方性法规 ${laws} 部` : `未涉及具体地区 · 地方性法规 ${laws} 部`;
  }
  if (step.node === "clarify") {
    const region = typeof row.region === "string" && row.region ? row.region : "national";
    const label = region === "national" ? "只按全国法" : `按「${region}」`;
    return `${label} · 纳入地方性法规 ${number(row.laws)} 部`;
  }
  if (step.node === "agent") {
    const tools = Array.isArray(row.tools) ? (row.tools as unknown[]) : [];
    const named = tools.filter((row) => typeof row === "string" && row !== "");
    const turn = number(row.turn);
    const maxSteps = number(row.max_steps);
    const head = maxSteps > 0 ? `第 ${turn}/${maxSteps} 轮` : `第 ${turn} 轮`;
    const tail = named.length > 0 ? `计划调用 ${named.join("、")}` : "收口";
    const truncated = row.truncated === true ? " · 上下文已截断" : "";
    return `${head} · ${tail}${truncated}`;
  }
  if (step.node === "tools") {
    return `检索 ${number(row.retrievals)} 次 · 命中 ${number(row.hits)} 条 · 材料 ${number(row.materials)} · 网搜 ${number(row.web)}`;
  }
  if (step.node === "finalize") {
    const fallback = row.fallback === true ? " · 走了检索兜底" : "";
    return `依据 ${number(row.evidence)} 条${fallback}`;
  }
  if (step.node === "review") {
    if (row.passed === undefined && row.score === undefined) return "本次没有可判的复核（空对象）";
    const score = typeof row.score === "number" ? row.score.toFixed(2) : "—";
    return `${row.passed === true ? "通过" : "未通过"} · 得分 ${score}`;
  }
  return "";
}

export function evidenceOf(answer: DonePayload | null): Evidence[] {
  return answer?.evidences ?? [];
}

export function foundLabel(group: string): string {
  if (group === "时效") return "时效";
  if (group === "材料") return "材料";
  return "依据";
}

export function groupClass(group: string): string {
  if (group === "时效") return "timeliness";
  if (group === "材料") return "material";
  return "basis";
}
