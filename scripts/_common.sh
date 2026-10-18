#!/usr/bin/env bash
# Shared helpers for the acceptance scripts. Sourced, never executed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

PYTHON_BIN="${MSRC_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

require_python() {
  [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ] || die "no python interpreter; run scripts/01_setup_envs.sh first"
}

# Every script takes --execute; without it the plan is printed and nothing runs.
MODE="plan"
set_mode() {
  for arg in "$@"; do
    case "$arg" in
      --execute|--run) MODE="execute" ;;
      --check)         MODE="check" ;;
    esac
  done
}

run() {
  if [ "$MODE" = "execute" ] || [ "$MODE" = "check" ]; then
    log "RUN: $*"
    "$@"
  else
    printf '   plan: %s\n' "$*"
  fi
}

require_new_dir() {
  [ -e "$1" ] && die "output directory already exists and will not be reused: $1"
  return 0
}
