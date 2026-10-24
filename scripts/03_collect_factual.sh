#!/usr/bin/env bash
# (3) 指标2.1 / 2.2：对三个被测模型重新生成回答、现场采集八类不确定性信号，
#     只在训练集上拟合融合校准器，再在测试集上计算校准性能提升。
#   用法： bash scripts/03_collect_factual.sh --run-id thirdparty_001 [--phase run|collect|fit] [--gpus 0] [--mode full|smoke] [--execute]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

RUN_ID=""; GPU=""; RUN_MODE="full"; PHASE="run"; PARAMETERS=""; SHARE_POLICY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus)   GPU="$2"; shift 2 ;;
    --mode)   RUN_MODE="$2"; shift 2 ;;
    --phase)  PHASE="$2"; shift 2 ;;
    --parameters) PARAMETERS="$2"; shift 2 ;;
    --shared-gpu-policy) SHARE_POLICY="$2"; shift 2 ;;
    --execute|--run|--check) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
[ -n "$RUN_ID" ] || die "--run-id is required, e.g. --run-id thirdparty_001"
validate_run_id "$RUN_ID"
case "$RUN_MODE" in full|smoke) ;; *) die "--mode must be full or smoke" ;; esac
case "$PHASE" in run|collect|fit|apply) ;; *) die "--phase must be run, collect, fit or apply" ;; esac
if [ "$PHASE" = fit ] || [ "$PHASE" = apply ]; then
  [ -z "$PARAMETERS" ] || die "set --parameters at collection, not evaluation"
fi
PARAMETER_ARGS=()
[ -z "$PARAMETERS" ] || PARAMETER_ARGS=(--parameters "$PARAMETERS")

require_python
export MSRC_GPU="${GPU:-0}"
[ -z "$SHARE_POLICY" ] || export MSRC_GPU_SHARE_POLICY="$SHARE_POLICY"
RUN_DIR="$("$PYTHON_BIN" -c 'import sys; from pipeline.common import inside, runs_dir; print(inside(runs_dir()/sys.argv[1]))' "$RUN_ID")"
OUT="$RUN_DIR/factual"

log "run id       : $RUN_ID"
log "run mode     : $RUN_MODE"
log "phase        : $PHASE"
log "gpu          : $MSRC_GPU"
log "output       : $OUT"
log "mode         : $MODE"

if [ "$MODE" != "execute" ]; then
  if [ "$PHASE" = fit ] || [ "$PHASE" = apply ]; then
    printf ' plan: %s -m pipeline.factual --out %s --mode %s --phase evaluate\n' "$PYTHON_BIN" "$OUT" "$RUN_MODE"
  else
    printf ' plan: mkdir -p %s\n' "$RUN_DIR"
    printf ' plan: %s -m pipeline.preflight --out %s\n' "$PYTHON_BIN" "$RUN_DIR"
    if [ "$PHASE" = collect ]; then
      printf ' plan: %s -m pipeline.factual --out %s --mode %s --phase prepare' "$PYTHON_BIN" "$OUT" "$RUN_MODE"
      [ -z "$PARAMETERS" ] || printf ' --parameters %s' "$PARAMETERS"
      printf '\n'
      printf ' plan: %s -m pipeline.factual --out %s --mode %s --phase collect\n' "$PYTHON_BIN" "$OUT" "$RUN_MODE"
    else
      printf ' plan: %s -m pipeline.factual --out %s --mode %s' "$PYTHON_BIN" "$OUT" "$RUN_MODE"
      [ -z "$PARAMETERS" ] || printf ' --parameters %s' "$PARAMETERS"
      printf '\n'
    fi
  fi
  exit 0
fi

if [ "$PHASE" = fit ] || [ "$PHASE" = apply ]; then
  [ -f "$OUT/protocol.json" ] && [ -d "$OUT/evidence" ] || die "collect this run's evidence first"
  [ ! -e "$OUT/calibrators" ] && [ ! -e "$OUT/report.json" ] || die "fitted output exists; use a new run ID"
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); expected="provided" if sys.argv[2]=="apply" else "fresh_fit"; assert p.get("parameter_mode","fresh_fit")==expected, "wrong parameter mode for phase"' "$OUT/protocol.json" "$PHASE"
  "$PYTHON_BIN" -m pipeline.factual --out "$OUT" --mode "$RUN_MODE" --phase evaluate
else
  [ -e "$OUT" ] && die "fresh output required: $OUT already exists"
  mkdir -p "$RUN_DIR"
  "$PYTHON_BIN" -m pipeline.preflight --out "$RUN_DIR" --wait-for-gpu
  if [ "$PHASE" = collect ]; then
    "$PYTHON_BIN" -m pipeline.factual --out "$OUT" --mode "$RUN_MODE" --phase prepare "${PARAMETER_ARGS[@]}"
    "$PYTHON_BIN" -m pipeline.factual --out "$OUT" --mode "$RUN_MODE" --phase collect
    echo "FACTUAL_COLLECTED $RUN_ID"
    exit 0
  fi
  "$PYTHON_BIN" -m pipeline.factual --out "$OUT" --mode "$RUN_MODE" "${PARAMETER_ARGS[@]}"
fi
log "factual evidence and calibration finished"
if [ "$RUN_MODE" = smoke ]; then
  echo "FACTUAL_SMOKE_COMPLETED_NOT_ACCEPTANCE $RUN_ID"
else
  echo "FACTUAL_COMPLETED $RUN_ID"
fi
