import { ApiError, type StreamEvent } from "./types";

export const USAGE =
  "帧格式：event: <名>\\ndata: <JSON>\\n\\n（服务端 ensure_ascii=False）。解析器按块喂入，跨块半行与 CRLF 都在这里收敛。";

type RawFrame = { event: string; data: string };

export class SseDecoder {
  private buffer = "";
  private pending: RawFrame = { event: "", data: "" };

  private line(rawLine: string, frames: RawFrame[]): void {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") {
      if (this.pending.data !== "" || this.pending.event !== "") {
        frames.push(this.pending);
      }
      this.pending = { event: "", data: "" };
      return;
    }
    if (line.startsWith(":")) {
      return;
    }
    if (line.startsWith("event:")) {
      this.pending.event = line.slice(6).trim();
      return;
    }
    if (line.startsWith("data:")) {
      const piece = line.slice(5).replace(/^ /, "");
      this.pending.data = this.pending.data === "" ? piece : `${this.pending.data}\n${piece}`;
    }
  }

  push(chunk: string): RawFrame[] {
    this.buffer += chunk;
    const frames: RawFrame[] = [];
    for (;;) {
      const index = this.buffer.indexOf("\n");
      if (index < 0) break;
      const rawLine = this.buffer.slice(0, index);
      this.buffer = this.buffer.slice(index + 1);
      this.line(rawLine, frames);
    }
    return frames;
  }

  flush(): RawFrame[] {
    const frames: RawFrame[] = [];
    const rest = this.buffer;
    this.buffer = "";
    if (rest !== "") {
      this.line(rest, frames);
    }
    if (this.pending.data !== "" || this.pending.event !== "") {
      frames.push(this.pending);
    }
    this.pending = { event: "", data: "" };
    return frames;
  }
}

export function toStreamEvent(frame: RawFrame): StreamEvent | null {
  let payload: unknown = null;
  try {
    payload = frame.data ? JSON.parse(frame.data) : {};
  } catch (exc) {
    throw new ApiError({
      kind: "parse",
      detail: `事件帧不是合法 JSON：${frame.event || "(无名)"} ${String(exc)}`,
      body: frame.data,
    });
  }
  const data = (payload ?? {}) as Record<string, unknown>;
  if (frame.event === "step") return { type: "step", data: data as { node: string } };
  if (frame.event === "delta") return { type: "delta", data };
  if (frame.event === "interrupt") return { type: "interrupt", data };
  if (frame.event === "done") return { type: "done", data };
  if (frame.event === "error") return { type: "error", data };
  return null;
}
