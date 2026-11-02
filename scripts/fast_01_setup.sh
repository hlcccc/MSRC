#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/tmp/fast_review" PIP_CACHE_DIR="$ROOT/cache/pip-fast"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
PY="${MSRC_BASE_PYTHON:-python3}"
WHEELS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --offline-wheels) WHEELS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if ! "$PY" -c 'import ensurepip,sys; assert sys.version_info >= (3,10)' >/dev/null 2>&1; then
  PY="$(python3 -B -m pipeline.bootstrap_python)"
fi
"$PY" -m venv .venv-review
if [ -n "$WHEELS" ]; then
  .venv-review/bin/python -m pip install --no-index --find-links "$WHEELS" -r requirements-fast-review.txt
else
  .venv-review/bin/python -m pip install --timeout 20 --retries 1 \
    -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" -r requirements-fast-review.txt
fi
.venv-review/bin/python -B -m unittest discover -s tests_fast -v
echo FAST_REVIEW_ENV_OK
