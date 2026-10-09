#!/usr/bin/env bash
# 冷备与恢复（R8）：停写 → 打包 → 起栈。
# 备份面 = volumes/{etcd,minio,milvus}（宿主 tar）+ 命名卷 agent_tlr_sessions（helper 容器 tar，
# 走 stdin/stdout，不往容器挂宿主目录，绕开 Git Bash 的中文路径 argv）；不备份可重建物
# （chunks/index 卷，重建配方见 README/手册）。etcd/minio 不停：唯一写者 milvus 已停，二者静止。
# 用法：
#   backup.sh                    打一份冷备到 deploy/backup/<YYYYmmdd-HHMMSS>/
#   backup.sh --keep N           同上，打完只保留最近 N 份
#   backup.sh --restore <目录>   反向恢复（同名卷目录改名留存 .replaced-<stamp>，命名卷清空后解包），起栈
# env：BACKUP_HELPER_IMAGE（tar 辅助镜像；默认挑本机已有的 alpine → 服务镜像 → 兜底 alpine:3.20）
set -euo pipefail

MODE="backup"
KEEP=""
RESTORE_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep)
      KEEP="${2:?--keep 需要份数}"
      shift 2
      ;;
    --restore)
      MODE="restore"
      RESTORE_DIR="${2:?--restore 需要产物目录}"
      shift 2
      ;;
    -h | --help)
      echo "用法: backup.sh [--keep 份数] [--restore 产物目录]"
      echo "  无参数          打一份冷备到 deploy/backup/<时间戳>/"
      echo "  --keep N        打完只保留最近 N 份"
      echo "  --restore DIR   反向恢复该产物（缺卷自动建；结束起栈）"
      echo "env: BACKUP_HELPER_IMAGE（tar 辅助镜像，默认自动挑本机已有的）"
      exit 0
      ;;
    *)
      echo "backup: 未知参数 $1" >&2
      exit 2
      ;;
  esac
done
if [[ -n "$KEEP" && ! "$KEEP" =~ ^[0-9]+$ ]]; then
  echo "backup: --keep 需要整数份数" >&2
  exit 2
fi
if [[ -n "$KEEP" && "$KEEP" -lt 1 ]]; then
  echo "backup: --keep 至少 1（0 会把所有产物删掉）" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="$REPO/deploy/docker-compose.yml"
VOLUMES="$REPO/volumes"
BACKUP_ROOT="$REPO/deploy/backup"
BIND_DIRS="etcd minio milvus"
WRITERS="app agent standalone"
SESSION_VOLUME="agent_tlr_sessions"
STAMP="$(date +%Y%m%d-%H%M%S)"

HELPER=""
STOPPED=""

resolve_helper() {
  if [[ -n "${BACKUP_HELPER_IMAGE:-}" ]]; then
    HELPER="$BACKUP_HELPER_IMAGE"
    return 0
  fi
  local cand
  for cand in alpine:3.20 alpine traffic-law-qa:0.1.0 milvusdb/milvus:v2.6.24; do
    if docker image inspect "$cand" >/dev/null 2>&1; then
      HELPER="$cand"
      return 0
    fi
  done
  HELPER="alpine:3.20"
}

running_writers() {
  docker compose -f "$COMPOSE" ps --status running --services $WRITERS 2>/dev/null | tr -d '\r' | tr '\n' ' '
}

stop_writers() {
  local list="$1"
  if [[ -z "${list// /}" ]]; then
    echo "[backup] 写侧未在运行，跳过停机"
    return 0
  fi
  echo "[backup] 停写：$list"
  docker compose -f "$COMPOSE" stop $list >/dev/null
  STOPPED="$list"
}

start_writers() {
  local list="$1"
  if [[ -z "${list// /}" ]]; then
    return 0
  fi
  echo "[backup] 起栈：$list"
  STOPPED=""
  docker compose -f "$COMPOSE" start $list
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && -n "$STOPPED" ]]; then
    echo "[backup] 失败退出（rc=$rc），先把写侧起回来：$STOPPED" >&2
    docker compose -f "$COMPOSE" start $STOPPED >/dev/null 2>&1 || true
  fi
}
trap on_exit EXIT

pack_volume() {
  MSYS_NO_PATHCONV=1 docker run --rm -v "$SESSION_VOLUME":/src:ro "$HELPER" \
    tar -czf - -C /src . >"$1"
}

unpack_volume() {
  MSYS_NO_PATHCONV=1 docker run --rm -i -v "$SESSION_VOLUME":/src "$HELPER" \
    sh -c 'rm -rf /src/* /src/.[!.]* /src/..?* 2>/dev/null; tar -xzf - -C /src' <"$1"
}

do_backup() {
  resolve_helper
  local dest="$BACKUP_ROOT/$STAMP" name
  for name in $BIND_DIRS; do
    if [[ ! -d "$VOLUMES/$name" ]]; then
      echo "backup: 找不到 $VOLUMES/$name" >&2
      exit 1
    fi
  done
  if ! docker volume inspect "$SESSION_VOLUME" >/dev/null 2>&1; then
    echo "backup: 找不到命名卷 $SESSION_VOLUME（先起过一次栈才有）" >&2
    exit 1
  fi
  mkdir -p "$dest"

  local running
  running="$(running_writers)"
  stop_writers "$running"
  for name in $BIND_DIRS; do
    tar -czf "$dest/$name.tar.gz" -C "$VOLUMES" "$name"
  done
  pack_volume "$dest/tlr_sessions.tar.gz"
  start_writers "$running"

  for name in $BIND_DIRS tlr_sessions; do
    if [[ ! -s "$dest/$name.tar.gz" ]]; then
      echo "backup: $dest/$name.tar.gz 空产物" >&2
      exit 1
    fi
  done
  echo "[backup] 完成 $dest（helper=$HELPER）"
  du -h "$dest"/*.tar.gz | sed 's/^/[backup]   /'

  if [[ -n "$KEEP" ]]; then
    prune "$KEEP"
  fi
}

prune() {
  local keep="$1" index=0 dir
  while IFS= read -r dir; do
    index=$((index + 1))
    if [[ $index -le "$keep" ]]; then
      continue
    fi
    echo "[backup] --keep $keep：清理旧份 $dir"
    rm -rf "$dir"
  done < <(find "$BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d | sort -r)
}

do_restore() {
  resolve_helper
  local src="$RESTORE_DIR" name
  if [[ ! -d "$src" ]]; then
    echo "backup: 产物目录不存在：$src" >&2
    exit 1
  fi
  for name in $BIND_DIRS tlr_sessions; do
    if [[ ! -s "$src/$name.tar.gz" ]]; then
      echo "backup: $src/$name.tar.gz 缺失或为空" >&2
      exit 1
    fi
  done

  echo "[backup] 恢复自 $src：停写 $WRITERS；同名卷目录改名 .replaced-$STAMP；命名卷内容清空后解包"
  local running
  running="$(running_writers)"
  stop_writers "$running"
  mkdir -p "$VOLUMES"
  for name in $BIND_DIRS; do
    if [[ -e "$VOLUMES/$name" ]]; then
      mv "$VOLUMES/$name" "$VOLUMES/$name.replaced-$STAMP"
      echo "[backup] 旧卷留存：$VOLUMES/$name.replaced-$STAMP（确认后可删）"
    fi
    tar -xzf "$src/$name.tar.gz" -C "$VOLUMES"
  done
  unpack_volume "$src/tlr_sessions.tar.gz"
  STOPPED=""
  echo "[backup] 起栈：全部已存在的服务"
  docker compose -f "$COMPOSE" start
  echo "[backup] 恢复完成，已起栈"
}

if [[ "$MODE" == "restore" ]]; then
  do_restore
else
  do_backup
fi
