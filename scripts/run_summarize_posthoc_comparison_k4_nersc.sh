#!/bin/bash -l
# Aggregate the three successful post-hoc comparison array tasks.
# NERSC project policy requires one GPU allocation for this analysis job.
#SBATCH --time=00:20:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J SR_postSum
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_repulsion/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_repulsion/%x_%j.err

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

# Avoid inherited, mutually exclusive memory variables in child processes.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/summarize_posthoc_comparison_k4.py \
  --root outputs/metrics/posthoc_comparison_k4 \
  --output outputs/metrics/posthoc_comparison_k4/summary.json \
  --expected-windows 0 3 7

echo "Post-hoc comparison aggregation completed."
echo "Summary: ${REPO_DIR}/outputs/metrics/posthoc_comparison_k4/summary.json"
echo "Figure:  ${REPO_DIR}/outputs/metrics/posthoc_comparison_k4/posthoc_full_comparison.png"
