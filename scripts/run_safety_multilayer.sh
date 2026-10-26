#!/usr/bin/env bash
# Full three-indicator procedure without changing any original variant entry.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"
RUN_ID=""; GPU="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --gpus) GPU="$2"; shift 2 ;;
    --execute) shift ;;
    *) die "unknown option: $1" ;;
  esac
done
validate_run_id "$RUN_ID"
step() {
  if [ "$MODE" = execute ]; then "$@"; else printf ' plan:'; printf ' %q' "$@"; printf '\n'; fi
}
if [ "$MODE" = execute ]; then require_new_dir "$ROOT/runs/$RUN_ID"; fi
step bash scripts/01_setup_envs.sh --execute
step bash scripts/02_prepare_assets_data.sh --execute --download-data --download-models --gpus "$GPU"
step bash scripts/02_prepare_assets_data.sh --check --gpus "$GPU"
step bash scripts/02_prepare_fitted.sh --execute
step bash scripts/02_prepare_fitted.sh --check
step bash scripts/04_score_multilayer.sh --run-id "$RUN_ID" --gpus "$GPU" --phase prepare --execute
step bash scripts/03_collect_factual.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --phase collect --execute
step bash scripts/03_collect_factual.sh --run-id "$RUN_ID" --gpus "$GPU" --mode full --phase fit --execute
step bash scripts/04_score_multilayer.sh --run-id "$RUN_ID" --gpus "$GPU" --phase all --execute
step bash scripts/05_report_multilayer.sh --run-id "$RUN_ID" --execute
