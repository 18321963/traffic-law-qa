#!/usr/bin/env bash
# 重建三个示例服务镜像并滚动上线（直连 buildx 配方）。
# web 镜像只 COPY frontend/dist —— 源码比 dist 新就在这里先重建 dist，不靠人记。
# 禁用 `docker compose up --build`：本机 compose v5.4.0 内嵌 bake 失败（gRPC 头错），
# 报错表面像网络问题，实际是 builder 选择坏；也禁用 `docker builder prune`。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ ! -d frontend/src ]; then
  echo "找不到 frontend/src：这里不是仓库根？" >&2
  exit 1
fi

FRONTEND_STAMP=frontend/dist/index.html
FRONTEND_INPUTS=(
  frontend/src
  frontend/index.html
  frontend/package.json
  frontend/package-lock.json
  frontend/tsconfig.json
  frontend/vite.config.ts
)

ACTION=""
if [ ! -f "$FRONTEND_STAMP" ]; then
  ACTION="dist 不存在"
else
  NEWER="$(find "${FRONTEND_INPUTS[@]}" -newer "$FRONTEND_STAMP" -print -quit 2>/dev/null || true)"
  if [ -n "$NEWER" ]; then
    ACTION="$NEWER 比 dist 新"
  fi
fi

if [ -n "$ACTION" ]; then
  echo "web 镜像只 COPY frontend/dist：$ACTION，先重建前端（cd frontend && npm ci && npm run build）" >&2
  (cd frontend && npm ci && npm run build)
else
  echo "web 镜像只 COPY frontend/dist：dist 不比源码旧，跳过 npm run build（要强制重建就先删 frontend/dist）"
fi

docker buildx build --load -t traffic-law-qa:0.1.0 -f rag_service/Dockerfile .
docker buildx build --load -t traffic-law-qa-agent:0.1.0 -f agent_service/Dockerfile .
docker buildx build --load -t traffic-law-qa-web:0.1.0 -f frontend/Dockerfile .

docker compose -f deploy/docker-compose.yml up -d --force-recreate --no-deps app agent web
