#!/usr/bin/env bash
# (5) 汇总指标2.1—2.3 并生成测试报告。
#   用法： bash scripts/05_score_report.sh --run-id thirdparty_001 [--stage milestone|completion] [--execute]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

RUN_ID=""; STAGE="milestone"; VARIANT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --stage)  STAGE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --execute|--run) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
[ -n "$RUN_ID" ] || die "--run-id is required, e.g. --run-id thirdparty_001"
validate_run_id "$RUN_ID"

require_python
ARGS=()
[ -z "$VARIANT" ] || ARGS=(--variant "$VARIANT")
RUN_DIR="$("$PYTHON_BIN" -c 'from pipeline.common import runs_dir; print(runs_dir())')/$RUN_ID"

log "run id       : $RUN_ID"
log "stage        : $STAGE"
log "run dir      : $RUN_DIR"
log "mode         : $MODE"

if [ "$MODE" != "execute" ]; then
  printf ' plan: %s -m pipeline.report --run-id %s --stage %s\n' "$PYTHON_BIN" "$RUN_ID" "$STAGE"
  [ -z "$VARIANT" ] || printf ' variant: %s\n' "$VARIANT"
  exit 0
fi

[ -d "$RUN_DIR" ] || die "no such run: $RUN_DIR"
"$PYTHON_BIN" -m pipeline.report --run-id "$RUN_ID" --stage "$STAGE" "${ARGS[@]}"
log "report ready: $RUN_DIR/report/report.html"
