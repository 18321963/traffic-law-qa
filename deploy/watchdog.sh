#!/usr/bin/env bash
# 探活告警：探 agent /health，判据 status==ok && rag.status==ok（与 compose healthcheck 同）。
# 用法：watchdog.sh          单次探测（给宿主计划任务，失败退出码 1）
#       watchdog.sh --loop N 每 N 秒一轮、长驻
# env：WATCHDOG_URL（默认 http://127.0.0.1:8001）、WATCHDOG_WEBHOOK_URL（非空时失败
#      POST 告警 JSON，空则只打日志）、WATCHDOG_PYTHON（指定解释器）。
set -euo pipefail

BASE="${WATCHDOG_URL:-http://127.0.0.1:8001}"
WEBHOOK="${WATCHDOG_WEBHOOK_URL:-}"
INTERVAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop)
      INTERVAL="${2:?--loop 需要秒数}"
      shift 2
      ;;
    -h | --help)
      echo "用法: watchdog.sh [--loop 秒数]"
      echo "env: WATCHDOG_URL / WATCHDOG_WEBHOOK_URL / WATCHDOG_PYTHON"
      exit 0
      ;;
    *)
      echo "watchdog: 未知参数 $1" >&2
      exit 2
      ;;
  esac
done
if [[ -n "$INTERVAL" && ! "$INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "watchdog: --loop 需要整数秒" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${WATCHDOG_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in python3 python "$REPO/.venv/Scripts/python.exe"; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "pass" >/dev/null 2>&1; then
      PY="$cand"
      break
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "watchdog: 找不到可用解释器（设 WATCHDOG_PYTHON 指定）" >&2
  exit 2
fi

probe() {
  "$PY" - "$BASE" <<'PY'
import json
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(sys.argv[1] + "/health", timeout=5) as resp:
        data = json.load(resp)
except Exception as exc:
    print(f"fail: 请求失败 {exc}")
    raise SystemExit(1)
if data.get("status") == "ok" and data.get("rag", {}).get("status") == "ok":
    print(f"ok: rag_laws={data.get('rag', {}).get('laws')}")
    raise SystemExit(0)
print(f"fail: status={data.get('status')!r} rag.status={data.get('rag', {}).get('status')!r}")
raise SystemExit(1)
PY
}

run_once() {
  local detail rc=0 stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  detail="$(probe 2>&1)" || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[watchdog] ok $stamp url=$BASE $detail"
    return 0
  fi
  echo "[watchdog] FAIL $stamp url=$BASE $detail"
  if [[ -n "$WEBHOOK" ]]; then
    local payload
    payload="$(
      "$PY" - "$BASE" "$stamp" "$detail" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "service": "traffic-law-qa-agent",
            "status": "fail",
            "url": sys.argv[1],
            "time": sys.argv[2],
            "reason": sys.argv[3][:200],
        }
    )
)
PY
    )"
    if printf '%s' "$payload" | curl -fsS -m 10 -X POST -H "Content-Type: application/json" --data-binary @- "$WEBHOOK" >/dev/null; then
      echo "[watchdog] webhook 已发送"
    else
      echo "[watchdog] webhook 发送失败（不改变退出码）" >&2
    fi
  fi
  return 1
}

if [[ -n "$INTERVAL" ]]; then
  while true; do
    run_once || true
    sleep "$INTERVAL"
  done
else
  run_once
fi
