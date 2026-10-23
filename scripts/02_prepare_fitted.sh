#!/usr/bin/env bash
# Only for supplied-state variants. Does not download public base model weights.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK=1 ;;
    --execute|--run) ;;
    *) die "unknown option: $arg" ;;
  esac
done
require_python
if [ "$MODE" = plan ]; then
  printf ' plan: %s -m pipeline.provided_assets\n' "$PYTHON_BIN"
  exit 0
fi
ARGS=()
[ "$CHECK" = 0 ] || ARGS=(--check)
"$PYTHON_BIN" -m pipeline.provided_assets "${ARGS[@]}"
