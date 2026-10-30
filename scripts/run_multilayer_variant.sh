#!/usr/bin/env bash
# Full collection/testing in both modes; only the parameter provenance differs.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
VARIANT=""; RUN_ID=""; GPU="0"; PREPARED_PACKAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) [ -z "$VARIANT" ] || die "duplicate --mode"; VARIANT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus) GPU="$2"; shift 2 ;;
    --prepared-package) PREPARED_PACKAGE="$2"; shift 2 ;;
    --execute) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
case "$VARIANT" in fitted|no_fit) ;; *) die "--mode must be fitted or no_fit" ;; esac
validate_run_id "$RUN_ID"
step() {
  if [ "$MODE" = execute ]; then "$@"; else printf ' plan:'; printf ' %q' "$@"; printf '\n'; fi
}
if [ "$MODE" = execute ]; then require_new_dir "$ROOT/runs/$RUN_ID"; fi
step bash scripts/01_setup_envs.sh --execute
PARAMETERS=(); PHASE=fit
if [ "$VARIANT" = fitted ]; then
  step bash scripts/02_prepare_fitted.sh --execute
  step bash scripts/02_prepare_fitted.sh --check
  PARAMETERS=(--parameters assets/fitted/factual.json)
  PHASE=apply
fi
DATA_ARGS=(--download-data)
[ -z "$PREPARED_PACKAGE" ] || DATA_ARGS=(--prepared-package "$PREPARED_PACKAGE")
step bash scripts/02_prepare_assets_data.sh --execute "${DATA_ARGS[@]}" --download-models --gpus "$GPU"
step bash scripts/02_prepare_assets_data.sh --check --gpus "$GPU"
step bash scripts/03_collect_factual.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --phase collect "${PARAMETERS[@]}" --execute
step bash scripts/03_collect_factual.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --phase "$PHASE" --execute
if [ "$VARIANT" = no_fit ]; then
  step bash scripts/04_score_safety.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --refit --execute
fi
step bash scripts/04_score_multilayer_variant.sh --mode "$VARIANT" --run-id "$RUN_ID" --gpus "$GPU" --phase prepare --execute
step bash scripts/04_score_multilayer_variant.sh --mode "$VARIANT" --run-id "$RUN_ID" --gpus "$GPU" --phase check --execute
step bash scripts/04_score_multilayer_variant.sh --mode "$VARIANT" --run-id "$RUN_ID" --gpus "$GPU" --phase all --execute
step bash scripts/05_report_multilayer_variant.sh --mode "$VARIANT" --run-id "$RUN_ID" --execute
