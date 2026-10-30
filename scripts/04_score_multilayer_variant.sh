#!/usr/bin/env bash
# Two isolated parameter modes; the historical hybrid entry remains unchanged.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
VARIANT=""; RUN_ID=""; GPU="0"; PHASE="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) VARIANT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus) GPU="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --execute|--check) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
case "$VARIANT" in fitted|no_fit) ;; *) die "--mode must be fitted or no_fit" ;; esac
validate_run_id "$RUN_ID"
require_python
export MSRC_GPU="$GPU" MSRC_SAFETY_BACKEND=plugguard
if [ "$MODE" = check ]; then PHASE=check; fi
run "$PYTHON_BIN" -B -m extensions.multilayer.variants --mode "$VARIANT" --run-id "$RUN_ID" --phase "$PHASE"
