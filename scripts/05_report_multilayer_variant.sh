#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
VARIANT=""; RUN_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) VARIANT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --execute) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
case "$VARIANT" in fitted|no_fit) ;; *) die "--mode must be fitted or no_fit" ;; esac
validate_run_id "$RUN_ID"
require_python
export CUDA_VISIBLE_DEVICES="" MSRC_SAFETY_BACKEND=plugguard
run "$PYTHON_BIN" -B -m extensions.multilayer.variants --mode "$VARIANT" --run-id "$RUN_ID" --phase joint-report
