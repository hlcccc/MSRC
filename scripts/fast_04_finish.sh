#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN="${MSRC_FAST_RUN:-fast_001}"
[[ "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "invalid run ID" >&2; exit 2; }
.venv-review/bin/python -B -m fast_review.finish --run "runs/$RUN" "$@"
cat "runs/$RUN/report.md"
