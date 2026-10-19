#!/usr/bin/env bash
# (4) 指标1.13：PlugGuard在生成过程中逐token判定；训练侧冻结参数，
#     测试侧计算准确率与分来源统计，再独立核验实际输出。
#   用法： bash scripts/04_score_safety.sh --run-id thirdparty_001 [--gpus 0] [--mode full|smoke] [--execute]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

RUN_ID=""; GPU=""; RUN_MODE="full"; HEAD_PACKAGE=""; REFIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus)   GPU="$2"; shift 2 ;;
    --mode)   RUN_MODE="$2"; shift 2 ;;
    --head-package) HEAD_PACKAGE="$2"; shift 2 ;;
    --refit) REFIT=1; shift ;;
    --execute|--run|--check) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
[ -n "$RUN_ID" ] || die "--run-id is required, e.g. --run-id thirdparty_001"
validate_run_id "$RUN_ID"
case "$RUN_MODE" in full|smoke) ;; *) die "--mode must be full or smoke" ;; esac
[ "$REFIT" = 0 ] || [ -z "$HEAD_PACKAGE" ] || die "--refit and --head-package are mutually exclusive"

require_python
export MSRC_GPU="${GPU:-0}"
export MSRC_SAFETY_BACKEND="plugguard"
RUN_DIR="$("$PYTHON_BIN" -c 'import sys; from pipeline.common import inside, runs_dir; print(inside(runs_dir()/sys.argv[1]))' "$RUN_ID")"
OUT="$RUN_DIR/safety"

log "run id       : $RUN_ID"
log "run mode     : $RUN_MODE"
log "backend      : plugguard (this release never falls back to SingGuard)"
log "gpu          : $MSRC_GPU"
log "output       : $OUT"
log "mode         : $MODE"

if [ "$MODE" != "execute" ]; then
  printf ' plan: %s -m pipeline.plugguard --out %s --mode %s' "$PYTHON_BIN" "$OUT" "$RUN_MODE"
  [ "$REFIT" = 0 ] || printf ' --refit'
  [ -z "$HEAD_PACKAGE" ] || printf ' --head-package %s' "$HEAD_PACKAGE"
  printf '\n'
  exit 0
fi

[ -e "$OUT" ] && die "fresh output required: $OUT already exists"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" -m pipeline.preflight --out "$RUN_DIR/safety_preflight" --wait-for-gpu
ARGS=(--out "$OUT" --mode "$RUN_MODE")
[ "$REFIT" = 0 ] || ARGS+=(--refit)
[ -z "$HEAD_PACKAGE" ] || ARGS+=(--head-package "$HEAD_PACKAGE")
"$PYTHON_BIN" -m pipeline.plugguard "${ARGS[@]}"
log "safety scoring and threshold frozen"
if [ "$RUN_MODE" = smoke ]; then
  echo "SAFETY_SMOKE_COMPLETED_NOT_ACCEPTANCE $RUN_ID"
else
  echo "SAFETY_COMPLETED $RUN_ID"
fi
