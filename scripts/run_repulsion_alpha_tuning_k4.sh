#!/bin/bash -l
# Step 3 strength tuning: matched K=4 control and alpha sweep on three windows.
#
# Each array task owns one window, so control files are never written by two
# jobs concurrently. Existing completed control/repulsion samples are reused.
# Override the default strengths with, for example:
#   sbatch --export=ALL,ALPHA_VALUES='0.10 0.15 0.20' \
#     scripts/run_repulsion_alpha_tuning_k4.sh
#SBATCH --time=04:00:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=0-2%3
#SBATCH -J SR_aTune
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_repulsion/%x_%A_%a.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_repulsion/%x_%A_%a.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/era5-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

WINDOWS=(0 3 7)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#WINDOWS[@]} )); then
  echo "ERROR: array task ${TASK_ID} is outside 0..$((${#WINDOWS[@]} - 1))" >&2
  exit 2
fi
WINDOW_INDEX="${WINDOWS[${TASK_ID}]}"
SEED="${SEED:-17}"
K=4
ALPHA_VALUES="${ALPHA_VALUES:-0.10 0.20}"
read -r -a ALPHAS <<< "${ALPHA_VALUES}"
REPULSION_MIN_SIGMA="${REPULSION_MIN_SIGMA:-0.05}"
REPULSION_MAX_SIGMA="${REPULSION_MAX_SIGMA:-5.0}"
REPULSION_SPATIAL_POOL="${REPULSION_SPATIAL_POOL:-8}"
FORCE_CONTROL="${FORCE_CONTROL:-0}"
FORCE_REPULSION="${FORCE_REPULSION:-0}"

bash scripts/check_main8_assets.sh data
bash scripts/check_demo_assets.sh
python scripts/verify_baseline_ensemble.py

mapfile -t LATENT_FILES < <(
  find data/latents/test -maxdepth 1 -type f -name '*.h5' -printf '%f\n' | sort
)
SOURCE_FILE="${LATENT_FILES[${WINDOW_INDEX}]}"
SOURCE_INDEX="${SOURCE_FILE#2020_}"
SOURCE_INDEX="${SOURCE_INDEX%.h5}"

CONTROL_NAME="repulsion_smoke_k4/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_control_seed${SEED}"
CONTROL_RUN="outputs/sr/${CONTROL_NAME}"
CONTROL_SAMPLE="${CONTROL_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"

CONFIG=checkpoints/prior/config.yaml
CKPT=checkpoints/prior/trigflow_adaptive_ema_019901.pt
AE=checkpoints/autoencoder/ae_model.pt
COMMON_ARGS=(
  --diffusion_config "${CONFIG}"
  --diffusion_checkpoint "${CKPT}"
  --ae_model_path "${AE}"
  --output_dir outputs/sr
  --sample_index "${WINDOW_INDEX}"
  --num_posterior_samples "${K}"
  --seed "${SEED}"
  --obs_frames every:4
  --obs_downscale 8
  --obs_mode subsample
  --guidance_scale 4.0
  --gamma 1e-1
  --meas_std 1e-2
  --num_steps 30
  --corrector_steps 1
  --corrector_snr 0.3
  --corrector_max_sigma 5
  --noise_scale 0
  --log_every 2
  --no_net_grad
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Alpha-tuning window: ${WINDOW_INDEX} (source ${SOURCE_INDEX})"
echo "Alpha values:        ${ALPHAS[*]}"
echo "Sigma band:          ${REPULSION_MIN_SIGMA}..${REPULSION_MAX_SIGMA}"

if [[ -s "${CONTROL_SAMPLE}" && "${FORCE_CONTROL}" == "0" ]]; then
  echo "Reusing K=4 control: ${CONTROL_SAMPLE}"
else
  echo "Generating matched K=4 control..."
  srun python -m era5_videogen.generate_posterior_repulsion \
    "${COMMON_ARGS[@]}" \
    --name "${CONTROL_NAME}"
fi

for ALPHA in "${ALPHAS[@]}"; do
  ALPHA_TAG="${ALPHA//./p}"
  REPULSION_NAME="repulsion_smoke_k4/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_alpha${ALPHA_TAG}_seed${SEED}"
  REPULSION_RUN="outputs/sr/${REPULSION_NAME}"
  REPULSION_SAMPLE="${REPULSION_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"
  METRICS_DIR="outputs/metrics/repulsion_tuning_k4/w$(printf '%02d' "${WINDOW_INDEX}")_alpha${ALPHA_TAG}"

  if [[ -s "${REPULSION_SAMPLE}" && "${FORCE_REPULSION}" == "0" ]]; then
    echo "Reusing alpha=${ALPHA} samples: ${REPULSION_SAMPLE}"
  else
    echo "Generating K=4 repulsion samples at alpha=${ALPHA}..."
    srun python -m era5_videogen.generate_posterior_repulsion \
      "${COMMON_ARGS[@]}" \
      --name "${REPULSION_NAME}" \
      --repulsion_strength "${ALPHA}" \
      --repulsion_bandwidth 0 \
      --repulsion_min_sigma "${REPULSION_MIN_SIGMA}" \
      --repulsion_max_sigma "${REPULSION_MAX_SIGMA}" \
      --repulsion_schedule sine \
      --repulsion_spatial_pool "${REPULSION_SPATIAL_POOL}"
  fi

  echo "Evaluating alpha=${ALPHA}..."
  srun python scripts/evaluate_repulsion_smoke.py \
    --repulsion-run "${REPULSION_RUN}" \
    --control-run "${CONTROL_RUN}" \
    --baseline-root outputs/sr/baseline_independent \
    --data-root data \
    --window "${WINDOW_INDEX}" \
    --output-dir "${METRICS_DIR}"
done

echo "K=4 alpha tuning completed for window ${WINDOW_INDEX}."
