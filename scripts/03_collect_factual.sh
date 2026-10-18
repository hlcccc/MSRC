#!/usr/bin/env bash
# (3) 指标2.1 / 2.2：对三个被测模型重新生成回答、现场采集八类不确定性信号，
#     只在训练集上拟合融合校准器，再在测试集上计算校准性能提升。
#   用法： bash scripts/03_collect_factual.sh --run-id thirdparty_001 [--gpus 0] [--mode full|smoke] [--execute]
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

RUN_ID=""; GPU=""; RUN_MODE="full"
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus)   GPU="$2"; shift 2 ;;
    --mode)   RUN_MODE="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$RUN_ID" ] || die "--run-id is required, e.g. --run-id thirdparty_001"
validate_run_id "$RUN_ID"
case "$RUN_MODE" in full|smoke) ;; *) die "--mode must be full or smoke" ;; esac

require_python
export MSRC_GPU="${GPU:-0}"
RUN_DIR="$("$PYTHON_BIN" -c 'import sys; from pipeline.common import inside, runs_dir; print(inside(runs_dir()/sys.argv[1]))' "$RUN_ID")"
OUT="$RUN_DIR/factual"

log "run id       : $RUN_ID"
log "run mode     : $RUN_MODE"
log "gpu          : $MSRC_GPU"
log "output       : $OUT"
log "mode         : $MODE"

if [ "$MODE" != "execute" ]; then
  printf ' plan: mkdir -p %s\n' "$RUN_DIR"
  printf ' plan: %s -m pipeline.preflight --out %s\n' "$PYTHON_BIN" "$RUN_DIR"
  printf ' plan: %s -m pipeline.factual --out %s --mode %s\n' "$PYTHON_BIN" "$OUT" "$RUN_MODE"
  exit 0
fi

[ -e "$OUT" ] && die "fresh output required: $OUT already exists"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" -m pipeline.preflight --out "$RUN_DIR" --wait-for-gpu
"$PYTHON_BIN" -m pipeline.factual --out "$OUT" --mode "$RUN_MODE"
log "factual evidence and calibration finished"
echo "FACTUAL_COMPLETED $RUN_ID"
