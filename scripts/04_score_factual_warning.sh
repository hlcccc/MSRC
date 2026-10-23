#!/usr/bin/env bash
# Factual indicator 2.3; not a harmful-content/streaming-token classifier.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
RUN_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --execute|--run) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
validate_run_id "$RUN_ID"
require_python
RUN_DIR="$("$PYTHON_BIN" -c 'from pipeline.common import runs_dir; print(runs_dir())')/$RUN_ID"
if [ "$MODE" != execute ]; then
  printf ' plan: %s -m pipeline.factual_warning --factual %s/factual --out %s/factual_warning\n' "$PYTHON_BIN" "$RUN_DIR" "$RUN_DIR"
  exit 0
fi
"$PYTHON_BIN" -m pipeline.factual_warning --factual "$RUN_DIR/factual" --out "$RUN_DIR/factual_warning"
