#!/usr/bin/env bash
# Full preparation -> fresh inference -> fit/load -> warning -> checked report.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
VARIANT=""; RUN_ID=""; GPU="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --variant) [ -z "$VARIANT" ] || die "duplicate --variant"; VARIANT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus) GPU="$2"; shift 2 ;;
    --execute|--run) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
case "$VARIANT" in factual_no_fit|factual_fitted|safety_no_fit|safety_fitted) ;; *) die "unknown variant" ;; esac
validate_run_id "$RUN_ID"
step() {
  if [ "$MODE" = execute ]; then
    "$@"
  else
    printf ' plan:'
    printf ' %q' "$@"
    printf '\n'
  fi
}
step bash scripts/01_setup_envs.sh --execute
step bash scripts/02_prepare_assets_data.sh --execute --download-data --download-models --gpus "$GPU"
step bash scripts/02_prepare_assets_data.sh --check --gpus "$GPU"
PARAMETERS=(); PHASE=fit
if [[ "$VARIANT" = *_fitted ]]; then
  step bash scripts/02_prepare_fitted.sh --execute
  step bash scripts/02_prepare_fitted.sh --check
  PARAMETERS=(--parameters assets/fitted/factual.json)
  PHASE=apply
fi
step bash scripts/03_collect_factual.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --phase collect "${PARAMETERS[@]}" --execute
step bash scripts/03_collect_factual.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --phase "$PHASE" --execute
if [[ "$VARIANT" = factual_* ]]; then
  step bash scripts/04_score_factual_warning.sh --run-id "$RUN_ID" --execute
else
  SAFETY=(--refit)
  [[ "$VARIANT" != *_fitted ]] || SAFETY=(--head-package assets/fitted/plugguard)
  step bash scripts/04_score_safety.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full "${SAFETY[@]}" --execute
fi
step bash scripts/05_score_report.sh --run-id "$RUN_ID" --variant "$VARIANT" --stage milestone --execute
