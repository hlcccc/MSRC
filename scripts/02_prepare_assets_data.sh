#!/usr/bin/env bash
# (2) Download public raw data, build benign inputs, and verify public models.
#   用法：
#     bash scripts/02_prepare_assets_data.sh --execute --package /path/msrc_assets_20261018.tar.gz
#     bash scripts/02_prepare_assets_data.sh --execute --download-models [--hf-endpoint URL]
#     bash scripts/02_prepare_assets_data.sh --execute --download-data --download-models
#     bash scripts/02_prepare_assets_data.sh --check
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

ASSETS="${MSRC_ASSETS:-$ROOT/assets}"
PACKAGE=""
DOWNLOAD=0
DOWNLOAD_DATA=0
ENDPOINT="${HF_ENDPOINT:-}"
GPU="0"
SOURCES=""
LOCAL_MODELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --assets)        ASSETS="$2"; shift 2 ;;
    --package)       PACKAGE="$2"; shift 2 ;;
    --download-models) DOWNLOAD=1; shift ;;
    --download-data) DOWNLOAD_DATA=1; shift ;;
    --hf-endpoint)   ENDPOINT="$2"; shift 2 ;;
    --gpus)          GPU="$2"; shift 2 ;;
    --sources)       SOURCES="$2"; shift 2 ;;
    --local-models)  LOCAL_MODELS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

require_python
[ "$DOWNLOAD_DATA" = "0" ] || { [ -z "$PACKAGE" ] && [ -z "$SOURCES" ]; } || die "--download-data cannot be mixed with --package or --sources"
log "assets       : $ASSETS"
log "mode         : $MODE"

if [ "$MODE" = "check" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets write-config --assets "$ASSETS" --gpu "$GPU"
  run "$PYTHON_BIN" -m pipeline.prepare_assets verify --assets "$ASSETS"
  run "$PYTHON_BIN" -m pipeline.prepare_assets verify-models --assets "$ASSETS"
  run "$PYTHON_BIN" -c 'from pipeline import preflight; preflight.check_models(); preflight.check_data(); print("CONFIG_PATHS_OK")'
  exit 0
fi

if [ -n "$PACKAGE" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets unpack --package "$PACKAGE" --assets "$ASSETS" --gpu "$GPU"
fi
if [ "$DOWNLOAD_DATA" = "1" ]; then
  run "$PYTHON_BIN" -m pipeline.public_assets --assets "$ASSETS" \
      ${ENDPOINT:+--hf-endpoint "$ENDPOINT"}
fi
if [ -n "$SOURCES" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_data --sources "$SOURCES" --assets "$ASSETS"
fi
if [ -n "$LOCAL_MODELS" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets reuse-models --local-models "$LOCAL_MODELS" --assets "$ASSETS" --gpu "$GPU"
fi
if [ "$DOWNLOAD" = "1" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets download-models --assets "$ASSETS" --gpu "$GPU" \
      ${ENDPOINT:+--hf-endpoint "$ENDPOINT"}
fi
if [ -z "$PACKAGE" ] && [ "$DOWNLOAD" = "0" ] && [ "$DOWNLOAD_DATA" = "0" ] && [ -z "$SOURCES" ] && [ -z "$LOCAL_MODELS" ]; then
  log "nothing selected; writing configs/paths.json for the current asset tree"
  run "$PYTHON_BIN" -m pipeline.prepare_assets write-config --assets "$ASSETS" --gpu "$GPU"
fi

if [ "$MODE" = "execute" ]; then
  "$PYTHON_BIN" -m pipeline.prepare_assets write-config --assets "$ASSETS" --gpu "$GPU"
  log "verifying the delivered manifest and the asset layout"
  if [ -f "$ASSETS/MANIFEST.sha256" ]; then
    "$PYTHON_BIN" -m pipeline.prepare_assets verify --assets "$ASSETS"
    "$PYTHON_BIN" -c 'from pipeline import preflight; preflight.check_data(); print("DATA_PATHS_OK")'
  else
    log "DATA_PENDING: use --download-data (or --package/--sources), then run --check"
  fi
  if [ -f "$ASSETS/models.lock.json" ]; then
    "$PYTHON_BIN" -c 'from pipeline import preflight; preflight.check_models(); print("MODEL_PATHS_OK")'
  else
    log "MODELS_PENDING: use --download-models or --local-models, then run --check"
  fi
  if [ -f "$ASSETS/MANIFEST.sha256" ] && [ -f "$ASSETS/models.lock.json" ]; then
    echo "CONFIG_PATHS_OK"
  fi
fi
