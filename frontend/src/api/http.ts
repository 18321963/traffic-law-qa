import { ApiError } from "./types";

export const USAGE =
  "HTTP 失败归一：detail 可能是字符串（HTTPException）或 pydantic 的校验数组（422），Retry-After 头按秒解析。";

type PydanticIssue = { loc?: (string | number)[]; msg?: string; type?: string };

export function detailOf(body: unknown, fallback: string): string {
  if (typeof body === "string" && body.trim() !== "") return body.trim();
  if (body && typeof body === "object") {
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim() !== "") return detail.trim();
    if (Array.isArray(detail)) {
      const rows = detail
        .filter((row): row is PydanticIssue => Boolean(row) && typeof row === "object")
        .map((row) => {
          const where = Array.isArray(row.loc) ? row.loc.filter((x) => x !== "body").join(".") : "";
          const what = row.msg || row.type || "不合法";
          return where ? `${where}：${what}` : what;
        });
      if (rows.length > 0) return rows.join("；");
      return "请求体不合法";
    }
  }
  return fallback;
}

export function retryAfterOf(headers: Headers): number | null {
  const raw = headers.get("Retry-After");
  if (!raw) return null;
  const seconds = Number(raw.trim());
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

export async function readFailure(response: Response): Promise<ApiError> {
  const text = await response.text().catch(() => "");
  let body: unknown = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  return new ApiError({
    kind: "http",
    status: response.status,
    detail: detailOf(body, `服务返回 ${response.status}`),
    retryAfter: retryAfterOf(response.headers),
    body,
  });
}

export function isAbort(exc: unknown): boolean {
  return exc instanceof Error && exc.name === "AbortError";
}

export function networkFailure(exc: unknown): ApiError {
  if (exc instanceof ApiError) return exc;
  if (isAbort(exc)) throw exc;
  const message = exc instanceof Error ? exc.message : String(exc);
  return new ApiError({ kind: "network", detail: `无法连接服务：${message}` });
}
