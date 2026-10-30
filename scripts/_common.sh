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
  if [ "$MODE" != "plan" ]; then
    for sub in tmp cache cache/pip cache/hf cache/hf/hub cache/torch; do
      dest="$(realpath -m "$ROOT/$sub")"
      case "$dest" in "$ROOT"/*) ;; *) die "cache path leaves project: $dest" ;; esac
      mkdir -p "$dest"
    done
    export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip"
    export HF_HOME="$ROOT/cache/hf" HF_HUB_CACHE="$ROOT/cache/hf/hub"
    export TORCH_HOME="$ROOT/cache/torch" XDG_CACHE_HOME="$ROOT/cache"
    export HF_DATASETS_CACHE="$ROOT/cache/hf/datasets" CUDA_CACHE_PATH="$ROOT/cache/cuda"
    export TRITON_CACHE_DIR="$ROOT/cache/triton" NUMBA_CACHE_DIR="$ROOT/cache/numba"
    export MPLCONFIGDIR="$ROOT/cache/matplotlib"
    export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    export ARROW_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
    export LD_LIBRARY_PATH="$ROOT/.venv/system-libs/usr/lib/x86_64-linux-gnu:$ROOT/.venv/system-libs/lib/x86_64-linux-gnu:$ROOT/.venv/system-libs/usr/lib/aarch64-linux-gnu:$ROOT/.venv/system-libs/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

validate_run_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid run ID: $1"
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
