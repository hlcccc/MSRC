#!/usr/bin/env bash
# (4) 指标2.3：护栏模型在生成回答前对"图像＋问题"逐条打分，训练集选阈值并冻结，
#     测试集算提前预警准确率，再对实际输出做答案核验（仅作补充记录）。
#   用法： bash scripts/04_score_safety.sh --run-id thirdparty_001 [--gpus 0] [--mode full|smoke] [--execute]
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
case "$RUN_MODE" in full|smoke) ;; *) die "--mode must be full or smoke" ;; esac

require_python
export MSRC_GPU="${GPU:-0}"
RUN_DIR="$("$PYTHON_BIN" -c 'from pipeline.common import runs_dir; print(runs_dir())')/$RUN_ID"
OUT="$RUN_DIR/safety"

log "run id       : $RUN_ID"
log "run mode     : $RUN_MODE"
log "gpu          : $MSRC_GPU"
log "output       : $OUT"
log "mode         : $MODE"

if [ "$MODE" != "execute" ]; then
  printf ' plan: %s -m pipeline.safety --out %s --phase run\n' "$PYTHON_BIN" "$OUT"
  printf ' plan: %s -m pipeline.delivery --out %s\n' "$PYTHON_BIN" "$OUT"
  exit 0
fi

[ -e "$OUT" ] && die "fresh output required: $OUT already exists"
mkdir -p "$RUN_DIR"
if [ "$RUN_MODE" = "smoke" ]; then
  "$PYTHON_BIN" -m pipeline.safety --out "$OUT" --phase run --delivery-limit 16
else
  for phase in prepare score evaluate; do
    "$PYTHON_BIN" -m pipeline.safety --out "$OUT" --phase "$phase"
  done
  "$PYTHON_BIN" -m pipeline.delivery --out "$OUT"
fi
log "safety scoring and threshold frozen"
echo "SAFETY_COMPLETED $RUN_ID"
