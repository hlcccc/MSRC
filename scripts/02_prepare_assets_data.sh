#!/usr/bin/env bash
# (2) 准备并校验模型与数据：交付包里的自建样本地集解包，公开权重按需下载。
#   用法：
#     bash scripts/02_prepare_assets_data.sh --execute --package /path/msrc_assets_20261018.tar.gz
#     bash scripts/02_prepare_assets_data.sh --execute --download-models [--hf-endpoint URL]
#     bash scripts/02_prepare_assets_data.sh --check
set -euo pipefail
source "$(dirname "$0")/_common.sh"
set_mode "$@"

ASSETS="${MSRC_ASSETS:-$ROOT/assets}"
PACKAGE=""
DOWNLOAD=0
ENDPOINT="${HF_ENDPOINT:-}"
GPU="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --assets)        ASSETS="$2"; shift 2 ;;
    --package)       PACKAGE="$2"; shift 2 ;;
    --download-models) DOWNLOAD=1; shift ;;
    --hf-endpoint)   ENDPOINT="$2"; shift 2 ;;
    --gpus)          GPU="$2"; shift 2 ;;
    *) shift ;;
  esac
done

require_python
log "assets       : $ASSETS"
log "mode         : $MODE"

if [ "$MODE" = "check" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets write-config --assets "$ASSETS" --gpu "$GPU"
  run "$PYTHON_BIN" -m pipeline.prepare_assets verify --assets "$ASSETS"
  run "$PYTHON_BIN" -c 'from pipeline import preflight; preflight.check_models(); preflight.check_data(); print("CONFIG_PATHS_OK")'
  exit 0
fi

if [ -n "$PACKAGE" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets unpack --package "$PACKAGE" --assets "$ASSETS" --gpu "$GPU"
fi
if [ "$DOWNLOAD" = "1" ]; then
  run "$PYTHON_BIN" -m pipeline.prepare_assets download-models --assets "$ASSETS" --gpu "$GPU" \
      ${ENDPOINT:+--hf-endpoint "$ENDPOINT"}
fi
if [ -z "$PACKAGE" ] && [ "$DOWNLOAD" = "0" ]; then
  log "nothing selected; writing configs/paths.json for the current asset tree"
  run "$PYTHON_BIN" -m pipeline.prepare_assets write-config --assets "$ASSETS" --gpu "$GPU"
fi

if [ "$MODE" = "execute" ]; then
  log "verifying the delivered manifest and the asset layout"
  "$PYTHON_BIN" -m pipeline.prepare_assets verify --assets "$ASSETS"
  "$PYTHON_BIN" -c 'from pipeline import preflight; preflight.check_models(); preflight.check_data(); print("CONFIG_PATHS_OK")'
fi
