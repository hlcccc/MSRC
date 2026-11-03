#!/usr/bin/env bash
#SBATCH --job-name=msrc-single5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
: "${SLURM_JOB_ID:?submit this script with sbatch}"
: "${SLURM_SUBMIT_DIR:?missing Slurm submit directory}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible GPU}"
cd "${MSRC_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[[ -f pipeline/fast_single.py && -f requirements-acceptance.txt ]] || {
  printf 'ERROR: submit from the extracted MSRC project directory\n' >&2
  exit 2
}
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "$CUDA_VISIBLE_DEVICES"
# All GPU smoke checks and inference run inside the allocation, never on the login node.
bash scripts/01_setup_envs.sh --execute \
  --offline-wheels ../env/wheels \
  --offline-python ../env/python/cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
mkdir -p cache/fitted
if [[ ! -f cache/fitted/parameters.tar.gz ]]; then
  cp ../fitted_only/parameters-20261023.tar.gz cache/fitted/parameters.tar.gz
fi
.venv/bin/python -B -m pipeline.provided_assets
.venv/bin/python -B -m pipeline.fast3_assets \
  --prepared-package ../prepared_data --local-models ../local-models.fast3.json
exec bash scripts/run_fast_single.sh "$@"
