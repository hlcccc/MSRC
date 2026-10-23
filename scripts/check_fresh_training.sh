#!/usr/bin/env bash
# Public acceptance must fit this run's project parameters, not import a package.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

RUN_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --execute|--run|--check) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
[ -n "$RUN_ID" ] || die "--run-id is required"
validate_run_id "$RUN_ID"
require_python
run "$PYTHON_BIN" -m pipeline.fresh_training --run-id "$RUN_ID"
