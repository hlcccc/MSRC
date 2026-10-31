#!/usr/bin/env bash
# (1) 创建隔离环境并安装验收所需的全部依赖，最后打印机器与版本清单。
#   用法：
#     bash scripts/01_setup_envs.sh [--execute]                   # 联网：走 PyPI 镜像
#     bash scripts/01_setup_envs.sh [--execute] --offline-wheels ../env/wheels \
#         [--offline-python ../env/python/cpython-3.10.19-...tar.gz]   # 离线：只用网盘材料
#   不加 --execute 只打印计划；加了才真正创建 .venv 并安装。
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

BASE_PY="${MSRC_BASE_PYTHON:-python3.10}"
VENV="$ROOT/.venv"
REQ="${MSRC_REQUIREMENTS:-$ROOT/requirements-acceptance.txt}"
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
OFFLINE_WHEELS=""
OFFLINE_PYTHON=""
while [ $# -gt 0 ]; do
  case "$1" in
    --offline-wheels) OFFLINE_WHEELS="${2:-}"; shift 2 ;;
    --offline-python) OFFLINE_PYTHON="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
done
if [ -n "$OFFLINE_PYTHON" ]; then
  [ -f "$OFFLINE_PYTHON" ] || die "offline python archive not found: $OFFLINE_PYTHON"
  export MSRC_PYTHON_ARCHIVE="$OFFLINE_PYTHON"
fi
if [ -n "$OFFLINE_WHEELS" ]; then
  [ -d "$OFFLINE_WHEELS" ] || die "offline wheels directory not found: $OFFLINE_WHEELS"
  OFFLINE_WHEELS="$(cd "$OFFLINE_WHEELS" && pwd)"
  PIP_SOURCE=(--no-index --find-links "$OFFLINE_WHEELS")
  INSTALL_MODE="offline wheels ($OFFLINE_WHEELS)"
else
  PIP_SOURCE=(-i "$INDEX")
  INSTALL_MODE="online index ($INDEX)"
fi

log "project      : $ROOT"
log "base python  : $BASE_PY"
log "venv         : $VENV"
log "pin file     : $REQ"
log "install      : $INSTALL_MODE"
log "mode         : $MODE"

case "$(realpath -m "$VENV")" in "$ROOT"/*) ;; *) die "venv leaves project: $VENV" ;; esac

if [ "$MODE" = "check" ]; then
  log "inspecting the existing environment without installing anything"
  [ -x "$VENV/bin/python" ] || die "no $VENV yet; run this script with --execute first"
  "$VENV/bin/python" -m pipeline.env_check --out "$ROOT/assets/environment.json"
  exit $?
fi

if [ "$MODE" = execute ]; then
  if [ -z "${MSRC_BASE_PYTHON:-}" ] && { ! have "$BASE_PY" || ! "$BASE_PY" -c 'import ensurepip' 2>/dev/null; }; then
    have python3 || die "python3 is required to bootstrap the private Python 3.10 runtime"
    BASE_PY="$(python3 -B -m pipeline.bootstrap_python)"
  fi
  have "$BASE_PY" || die "$BASE_PY not found; set MSRC_BASE_PYTHON to a CPython 3.10 interpreter"
  "$BASE_PY" -c 'import ensurepip,sys; assert sys.version_info[:2] == (3, 10), "CPython 3.10 is required"'
fi

if [ ! -x "$VENV/bin/python" ]; then
  run "$BASE_PY" -m venv "$VENV"
else
  log "reusing the existing virtual environment"
fi

PIP=("$VENV/bin/python" -m pip)
run "${PIP[@]}" install pip==25.2 wheel==0.45.1 "${PIP_SOURCE[@]}"
TORCH_RUNTIME_ARGS=()
[ -z "$OFFLINE_WHEELS" ] || TORCH_RUNTIME_ARGS=(--offline-wheels "$OFFLINE_WHEELS")
run "$VENV/bin/python" -B -m pipeline.torch_runtime "${TORCH_RUNTIME_ARGS[@]}"
run "${PIP[@]}" install -r "$REQ" "${PIP_SOURCE[@]}"
run "${PIP[@]}" install -e "$ROOT" --no-deps "${PIP_SOURCE[@]}"

if [ "$MODE" = "execute" ]; then
  "$VENV/bin/python" -m pipeline.runtime_libraries
  log "verifying the interpreter, the pinned packages and the GPU"
  "$VENV/bin/python" -m pipeline.env_check --out "$ROOT/assets/environment.json"
  log "environment ready. Re-run with --check to inspect it without installing anything."
else
  printf '\n plan only. Re-run with --execute to install.\n'
fi
