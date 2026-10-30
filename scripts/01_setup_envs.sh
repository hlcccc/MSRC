#!/usr/bin/env bash
# (1) 创建隔离环境并安装验收所需的全部依赖，最后打印机器与版本清单。
#   用法： bash scripts/01_setup_envs.sh [--execute]
#   不加 --execute 只打印计划；加了才真正创建 .venv 并安装。
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

BASE_PY="${MSRC_BASE_PYTHON:-python3.10}"
VENV="$ROOT/.venv"
REQ="${MSRC_REQUIREMENTS:-$ROOT/requirements-acceptance.txt}"
INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

log "project      : $ROOT"
log "base python  : $BASE_PY"
log "venv         : $VENV"
log "pin file     : $REQ"
log "pypi index   : $INDEX"
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
run "${PIP[@]}" install pip==25.2 wheel==0.45.1 -i "$INDEX"
run "${PIP[@]}" install -r "$REQ" -i "$INDEX"
run "${PIP[@]}" install -e "$ROOT" --no-deps -i "$INDEX"

if [ "$MODE" = "execute" ]; then
  "$VENV/bin/python" -m pipeline.runtime_libraries
  log "verifying the interpreter, the pinned packages and the GPU"
  "$VENV/bin/python" -m pipeline.env_check --out "$ROOT/assets/environment.json"
  log "environment ready. Re-run with --check to inspect it without installing anything."
else
  printf '\n plan only. Re-run with --execute to install.\n'
fi
