#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN="${MSRC_FAST_RUN:-fast_001}"
[[ "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid run ID" >&2; exit 2; }
.venv/bin/python -B -m fast_review.demo --out "runs/$RUN/demo"
