#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
RUN_ID=""; FACTUAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --factual-run-id) FACTUAL="$2"; shift 2 ;;
    --execute) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
validate_run_id "$RUN_ID"
FACTUAL="${FACTUAL:-$RUN_ID}"
validate_run_id "$FACTUAL"
require_python
export MSRC_SAFETY_BACKEND=plugguard CUDA_VISIBLE_DEVICES=""
run "$PYTHON_BIN" -B -m extensions.multilayer.runner --run-id "$RUN_ID" \
  --factual-run-id "$FACTUAL" --phase joint-report
