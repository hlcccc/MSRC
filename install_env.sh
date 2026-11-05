#!/usr/bin/env bash
# Explicit offline install only; the check path never invokes this file.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$ROOT"
status=0
python3 -B check_env.py --environment-only || status=$?
if [ "$status" = 0 ]; then
  printf '%s\n' 'ENV_REUSE_OK: skip installation and installer downloads.'
  exit 0
fi
if [ "$status" != 2 ]; then
  printf '%s\n' 'Inspect the check error before installing.' >&2
  exit "$status"
fi
MATERIALS="${1:-$ROOT/..}"
MATERIALS="$(cd -- "$MATERIALS" && pwd -P)"
[ -d "$MATERIALS/env/wheels" ] || { printf '%s\n' 'NEED_DOWNLOAD: env/wheels' >&2; exit 2; }
[ -f "$MATERIALS/env_patch/SHA256SUMS.txt" ] || { printf '%s\n' 'NEED_DOWNLOAD: env_patch' >&2; exit 2; }
# Do not overwrite a material directory shared with another reproduction.
WHEELS="$(realpath -m -- "$ROOT/cache/install-wheels")"
case "$WHEELS" in "$ROOT"/*) ;; *) printf '%s\n' 'Wheel cache leaves project.' >&2; exit 4 ;; esac
(cd -- "$MATERIALS/env_patch" && sha256sum -c SHA256SUMS.txt)
args=(--execute --offline-wheels "$WHEELS")
base="${MSRC_BASE_PYTHON:-python3.10}"
if ! "$base" -B -c 'import ensurepip,sys; assert sys.version_info[:2] == (3,10)'; then
  archive="$MATERIALS/env/python/cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
  [ -f "$archive" ] || { printf 'NEED_DOWNLOAD: %s\n' "$archive" >&2; exit 2; }
  args+=(--offline-python "$archive")
fi
mkdir -p -- "$WHEELS"
cp --reflink=auto --no-preserve=ownership -- "$MATERIALS/env/wheels/"*.whl "$WHEELS/"
cp --reflink=auto --no-preserve=ownership -- "$MATERIALS/env_patch/wheels/"*.whl "$WHEELS/"
printf '%s\n' 'Running the original offline installer; its final ENV_OK check requires an allocated GPU.'
bash scripts/01_setup_envs.sh "${args[@]}"
unset MSRC_PYTHON
export MSRC_PYTHON="$ROOT/.venv/bin/python"
python3 -B check_env.py --environment-only
