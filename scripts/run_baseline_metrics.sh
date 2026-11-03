#!/bin/bash -l
# Step 2 evaluation of the completed independent SR baseline ensemble.
#SBATCH --time=02:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J SR_metrics
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_metrics/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_metrics/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
cd "${REPO_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/era5-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

python scripts/verify_baseline_ensemble.py
python scripts/evaluate_baseline_ensemble.py \
  --root outputs/sr/baseline_independent \
  --data-root data \
  --output-dir outputs/metrics/baseline_independent

echo "Step 2 baseline evaluation completed."
echo "Summary: ${REPO_DIR}/outputs/metrics/baseline_independent/summary.json"
