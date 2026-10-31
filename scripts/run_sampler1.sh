#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J SR_E3c
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_sampling/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_sampling/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
# The downloaded demo bundle has one local dataset element, index 0. The
# original source-window index is retained inside that element's metadata.
WINDOW_INDEX="${1:-0}"

cd "${REPO_DIR}"

# Fail with download/setup instructions before starting any expensive model
# work. checkpoints/ and data/ are intentionally excluded from Git.
bash scripts/check_demo_assets.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Conda environment: ${CONDA_PREFIX}"
echo "Python: $(which python)"
echo "Window index: ${WINDOW_INDEX}"

srun python -c \
'import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print("Visible GPUs:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))'

srun bash scripts/run_sr.sh "${WINDOW_INDEX}"

echo "Sampling completed."
# echo "Output: ${REPO_DIR}/outputs/sr/E3c_lam0/sample_$(printf "%04d" "${WINDOW_INDEX}")"
SEED="${SR_SEED:-17}"
RUN_NAME="${SR_RUN_NAME:-E3c_lam0_seed${SEED}}"
echo "Output: ${REPO_DIR}/outputs/sr/${RUN_NAME}/sample_$(printf "%04d" "${WINDOW_INDEX}")"
