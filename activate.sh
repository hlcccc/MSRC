#!/usr/bin/env bash
# Source this file after check_env.py has selected a qualified interpreter.
_msrc_activate() {
  local root python record key sub path
  root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
  record="$root/cache/check-first/python.path"
  case "$(realpath -m -- "$record")" in "$root"/*) ;; *) printf '%s\n' 'Python record leaves project.' >&2; return 1 ;; esac
  if [ ! -f "$record" ] || ! IFS= read -r python < "$record" || [ ! -x "$python" ]; then
    printf '%s\n' 'No checked Python. Run python3 -B check_env.py first.' >&2
    return 1
  fi
  for sub in tmp cache cache/pip cache/hf cache/hf/hub cache/hf/datasets \
      cache/torch cache/cuda cache/triton cache/numba cache/matplotlib; do
    path="$(realpath -m -- "$root/$sub")" || return 1
    case "$path" in "$root"/*) ;; *) printf 'Path leaves project: %s\n' "$path" >&2; return 1 ;; esac
    mkdir -p -- "$path" || return 1
  done
  export MSRC_PYTHON="$python" MSRC_CONFIG="$root/configs/paths.json"
  export TMPDIR="$root/tmp" PIP_CACHE_DIR="$root/cache/pip" XDG_CACHE_HOME="$root/cache"
  export HF_HOME="$root/cache/hf" HF_HUB_CACHE="$root/cache/hf/hub"
  export HF_DATASETS_CACHE="$root/cache/hf/datasets" TORCH_HOME="$root/cache/torch"
  export CUDA_CACHE_PATH="$root/cache/cuda" TRITON_CACHE_DIR="$root/cache/triton"
  export NUMBA_CACHE_DIR="$root/cache/numba" MPLCONFIGDIR="$root/cache/matplotlib"
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
  export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  export ARROW_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
  if [ "$(dirname -- "$python")" = "$root/.venv/bin" ]; then
    export LD_LIBRARY_PATH="$root/.venv/system-libs/usr/lib/x86_64-linux-gnu:$root/.venv/system-libs/lib/x86_64-linux-gnu:$root/.venv/system-libs/usr/lib/aarch64-linux-gnu:$root/.venv/system-libs/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  printf 'MSRC_PYTHON=%s\n' "$MSRC_PYTHON"
}
if _msrc_activate; then
  unset -f _msrc_activate
else
  unset -f _msrc_activate
  return 1 2>/dev/null || exit 1
fi
