#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/Scripts/python.exe"
PROXY="${PROXY:-http://127.0.0.1:7890}"
EXTRAS="${1:-all}"

echo "== Clash 探活 $PROXY"
if ! curl -sS -o /dev/null -m 12 -r 0-1023 -x "$PROXY" https://pypi.org/simple/; then
  echo "代理不通：先起 Clash（或 PROXY=... $0 $EXTRAS）" >&2
  exit 1
fi

echo "== pip install -e .[$EXTRAS]"
cd "$REPO"
"$PY" -m pip install -e ".[$EXTRAS]" --no-build-isolation --proxy "$PROXY"

echo "== 从仓外验证安装（cwd=/tmp，仓根不在 sys.path）"
cd /tmp
"$PY" -P -c "import agent_service, api_contracts, eval, mcp_server, rag_contracts, rag_service; print('agent_service ->', agent_service.__file__); print('api_contracts ->', api_contracts.__file__); print('eval ->', eval.__file__); print('mcp_server ->', mcp_server.__file__); print('rag_contracts ->', rag_contracts.__file__); print('rag_service ->', rag_service.__file__)"
"$PY" -m pip list 2>/dev/null | grep -i "traffic-law" || echo "（未找到 traffic-law-qa 分发，见下）"
