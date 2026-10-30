#!/usr/bin/env bash
# (2) Download public raw data, build benign inputs, and verify public models.
#   用法：
#     bash scripts/02_prepare_assets_data.sh --execute --package /path/msrc_assets_20261018.tar.gz
#     bash scripts/02_prepare_assets_data.sh --execute --download-models [--hf-endpoint URL]
#     bash scripts/02_prepare_assets_data.sh --execute --local-models ../models/local-models.json
#     bash scripts/02_prepare_assets_data.sh --execute --download-data --download-models
#     bash scripts/02_prepare_assets_data.sh --check
#  离线包（网盘材料）用 --local-models；--skip-optional-roles 只下载多层交付需要的角色，
#  不下 SingGuard-4b。
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

ASSETS="${MSRC_ASSETS:-$ROOT/assets}"
PACKAGE=""
RAW_PACKAGE=""
PREPARED_PACKAGE=""
DOWNLOAD=0
DOWNLOAD_DATA=0
ENDPOINT="${HF_ENDPOINT:-}"
GPU="0"
SOURCES=""
LOCAL_MODELS=""
SKIP_OPTIONAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --assets)        ASSETS="$2"; shift 2 ;;
    --package)       PACKAGE="$2"; shift 2 ;;
    --raw-package)   RAW_PACKAGE="$2"; shift 2 ;;
    --prepared-package) PREPARED_PACKAGE="$2"; shift 2 ;;
    --download-models) DOWNLOAD=1; shift ;;
    --download-data) DOWNLOAD_DATA=1; shift ;;
    --hf-endpoint)   ENDPOINT="$2"; shift 2 ;;
    --gpus)          GPU="$2"; shift 2 ;;
    --sources)       SOURCES="$2"; shift 2 ;;
    --local-models)  LOCAL_MODELS="$2"; shift 2 ;;
    --skip-optional-roles) SKIP_OPTIONAL=1; shift ;;
    *) shift ;;
  esac
done

require_python
[ -z "$PREPARED_PACKAGE" ] || { [ -z "$PACKAGE" ] && [ -z "$RAW_PACKAGE" ] && [ -z "$SOURCES" ] && [ "$DOWNLOAD_DATA" = "0" ]; } || die "--prepared-package cannot be mixed with other data preparation options"
[ "$DOWNLOAD_DATA" = "0" ] || { [ -z "$PACKAGE" ] && [ -z "$SOURCES" ]; } || die "--download-data cannot be mixed with --package or --sources"
[ -z "$RAW_PACKAGE" ] || { [ -z "$PACKAGE" ] && [ -z "$SOURCES" ]; } || die "--raw-package cannot be mixed with --package or --sources"
log "assets       : $ASSETS"
log "mode         : $MODE"

if [ "$MODE" = "check" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets write-config --assets "$ASSETS" --gpu "$GPU"
  run "$PYTHON_BIN" -m pipeline.prepare_assets verify --assets "$ASSETS"
  run "$PYTHON_BIN" -m pipeline.prepare_assets verify-models --assets "$ASSETS"
  run "$PYTHON_BIN" -c 'from pipeline import preflight; preflight.check_models(); preflight.check_data(); print("CONFIG_PATHS_OK")'
  exit 0
fi

if [ -n "$PREPARED_PACKAGE" ]; then
  run "$PYTHON_BIN" -m pipeline.prepared_bundle restore --package "$PREPARED_PACKAGE" --assets "$ASSETS"
fi
if [ -n "$PACKAGE" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets unpack --package "$PACKAGE" --assets "$ASSETS" --gpu "$GPU"
fi
if [ "$DOWNLOAD_DATA" = "1" ] || [ -n "$RAW_PACKAGE" ]; then
  RAW_ARGS=()
  [ -z "$RAW_PACKAGE" ] || RAW_ARGS=(--raw-package "$RAW_PACKAGE")
  run "$PYTHON_BIN" -m pipeline.public_assets --assets "$ASSETS" \
      "${RAW_ARGS[@]}" ${ENDPOINT:+--hf-endpoint "$ENDPOINT"}
fi
if [ -n "$SOURCES" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_data --sources "$SOURCES" --assets "$ASSETS"
fi
if [ -n "$LOCAL_MODELS" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets reuse-models --local-models "$LOCAL_MODELS" --assets "$ASSETS" --gpu "$GPU"
fi
if [ "$DOWNLOAD" = "1" ]; then
  SKIP_ARGS=()
  [ "$SKIP_OPTIONAL" = "0" ] || SKIP_ARGS=(--skip-optional-roles)
  run "$PYTHON_BIN" -m pipeline.prepare_assets download-models --assets "$ASSETS" --gpu "$GPU" \
      "${SKIP_ARGS[@]}" ${ENDPOINT:+--hf-endpoint "$ENDPOINT"}
fi
if [ -z "$PACKAGE" ] && [ -z "$RAW_PACKAGE" ] && [ -z "$PREPARED_PACKAGE" ] && [ "$DOWNLOAD" = "0" ] && [ "$DOWNLOAD_DATA" = "0" ] && [ -z "$SOURCES" ] && [ -z "$LOCAL_MODELS" ]; then
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
