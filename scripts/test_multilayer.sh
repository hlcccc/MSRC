#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$ROOT/extensions/multilayer/frozen:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"${MSRC_PYTHON:-python3}" -B -m unittest discover -s extensions/multilayer/tests -v
