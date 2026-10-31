#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH -C cpu
#SBATCH --account=m5021
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -J SR_Fig2
#SBATCH --output=logs/sr_sampling/%x_%j.out
#SBATCH --error=logs/sr_sampling/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
RUN_GLOB="${1:?Usage: sbatch scripts/plot_sr_figure2.slurm 'RUN_GLOB' OUTPUT.png [FRAME]}"
OUTPUT="${2:-outputs/figures/figure2.png}"
FRAME="${3:-0}"

cd "${REPO_DIR}"
mkdir -p "$(dirname "${OUTPUT}")"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

srun python scripts/plot_sr_figure2.py \
  --run-glob "${RUN_GLOB}" \
  --output "${OUTPUT}" \
  --frame "${FRAME}" \
  --expected-samples 8
