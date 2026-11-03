#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${MSRC_PYTHON:-.venv/bin/python}"
exec "$PYTHON" -B -u -m pipeline.fast3 --phase local "$@"
