#!/usr/bin/env bash
# 重建三个示例服务镜像并滚动上线（直连 buildx 配方）。
# web 镜像只 COPY frontend/dist —— 改前端先 `cd frontend && npm ci && npm run build`。
# 禁用 `docker compose up --build`：本机 compose v5.4.0 内嵌 bake 失败（gRPC 头错），
# 报错表面像网络问题，实际是 builder 选择坏；也禁用 `docker builder prune`。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ ! -d frontend/dist ]; then
  echo "frontend/dist 不存在：先 cd frontend && npm ci && npm run build" >&2
  exit 1
fi

docker buildx build --load -t traffic-law-qa:0.1.0 -f rag_service/Dockerfile .
docker buildx build --load -t traffic-law-qa-agent:0.1.0 -f agent_service/Dockerfile .
docker buildx build --load -t traffic-law-qa-web:0.1.0 -f frontend/Dockerfile .

docker compose -f deploy/docker-compose.yml up -d --force-recreate --no-deps app agent web
