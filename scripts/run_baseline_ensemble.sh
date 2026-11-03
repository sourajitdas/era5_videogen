#!/bin/bash -l
# Independent SR baseline: 8 local windows x 8 seeds = 64 Slurm array tasks.
#
# Smoke test:
#   sbatch --array=0 --export=ALL,BASELINE_LABEL=baseline_smoke \
#     scripts/run_baseline_ensemble.sh
#
# Full baseline (the array directive below is 0-63 with at most 8 concurrent):
#   sbatch scripts/run_baseline_ensemble.sh
#
# Every task gets a unique output directory, so tasks never race while writing
# metadata.json. The local indices 0..7 map to source windows recorded in the
# compact bundle filenames (93, 244, 478, 554, 731, 1024, 1181, 1296).
#SBATCH --time=01:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=0-63%8
#SBATCH -J SR_base
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_sampling/%x_%A_%a.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_sampling/%x_%A_%a.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
cd "${REPO_DIR}"

WINDOWS=(0 1 2 3 4 5 6 7)
SEEDS=(17 18 19 20 21 22 23 24)
NUM_WINDOWS="${#WINDOWS[@]}"
NUM_SEEDS="${#SEEDS[@]}"
NUM_TASKS="$((NUM_WINDOWS * NUM_SEEDS))"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "ERROR: array task ${TASK_ID} is outside the valid range 0-$((NUM_TASKS - 1))" >&2
  exit 2
fi

WINDOW_INDEX="${WINDOWS[$((TASK_ID / NUM_SEEDS))]}"
SEED="${SEEDS[$((TASK_ID % NUM_SEEDS))]}"

bash scripts/check_main8_assets.sh data
bash scripts/check_demo_assets.sh

mapfile -t LATENT_FILES < <(
  find data/latents/test -maxdepth 1 -type f -name '*.h5' -printf '%f\n' | sort
)
if (( ${#LATENT_FILES[@]} != NUM_WINDOWS )); then
  echo "ERROR: expected ${NUM_WINDOWS} active latent files, found ${#LATENT_FILES[@]}" >&2
  exit 2
fi

SOURCE_FILE="${LATENT_FILES[${WINDOW_INDEX}]}"
SOURCE_INDEX="${SOURCE_FILE#2020_}"
SOURCE_INDEX="${SOURCE_INDEX%.h5}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline_independent}"

export SR_SEED="${SEED}"
export SR_RUN_NAME="${BASELINE_LABEL}/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_seed${SEED}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Array task:         ${TASK_ID}/${NUM_TASKS}"
echo "Host:               $(hostname)"
echo "Python:             $(which python)"
echo "Local window index: ${WINDOW_INDEX}"
echo "Source window:      ${SOURCE_INDEX}"
echo "Independent seed:   ${SEED}"
echo "Output run:         outputs/sr/${SR_RUN_NAME}"

srun python -c \
'import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))'

srun bash scripts/run_sr.sh "${WINDOW_INDEX}"

SAMPLE_DIR="outputs/sr/${SR_RUN_NAME}/sample_$(printf '%04d' "${WINDOW_INDEX}")"
for name in samples.npy ground_truth.npy observation_lowres.npy metrics.json; do
  [[ -s "${SAMPLE_DIR}/${name}" ]] || {
    echo "ERROR: expected output is missing or empty: ${SAMPLE_DIR}/${name}" >&2
    exit 2
  }
done
[[ -s "outputs/sr/${SR_RUN_NAME}/metadata.json" ]] || {
  echo "ERROR: run metadata is missing" >&2
  exit 2
}

echo "Baseline member completed: ${SAMPLE_DIR}"
