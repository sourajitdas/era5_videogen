#!/bin/bash -l
# Final eight-window matched K=8 repulsion evaluation on NERSC Perlmutter.
# Each array task owns one window and runs a matched independent control plus
# the joint alpha=0.20 repulsion ensemble on one 80-GB A100.
#SBATCH --time=06:00:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=0-7%4
#SBATCH -J SR_repK8_full
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

# Avoid mutually exclusive inherited memory requests in subsequent srun steps.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

WINDOW_INDEX="${SLURM_ARRAY_TASK_ID:-${WINDOW_INDEX:-0}}"
if (( WINDOW_INDEX < 0 || WINDOW_INDEX > 7 )); then
  echo "ERROR: WINDOW_INDEX=${WINDOW_INDEX} is outside 0..7" >&2
  exit 2
fi

SEED="${SEED:-17}"
K=8
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.20}"
REPULSION_MIN_SIGMA="${REPULSION_MIN_SIGMA:-0.05}"
REPULSION_MAX_SIGMA="${REPULSION_MAX_SIGMA:-5.0}"
REPULSION_SPATIAL_POOL="${REPULSION_SPATIAL_POOL:-8}"
FORCE_CONTROL="${FORCE_CONTROL:-0}"
FORCE_REPULSION="${FORCE_REPULSION:-0}"

bash scripts/check_main8_assets.sh data
bash scripts/check_demo_assets.sh

GPU_MEMORY_MIB="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | head -n 1 | tr -d ' '
)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ ! "${GPU_MEMORY_MIB}" =~ ^[0-9]+$ ]] || (( GPU_MEMORY_MIB < 79000 )); then
  echo "ERROR: expected an 80-GB A100, got ${GPU_MEMORY_MIB} MiB." >&2
  exit 2
fi

mapfile -t LATENT_FILES < <(
  find data/latents/test -maxdepth 1 -type f -name '*.h5' -printf '%f\n' | sort
)
if (( ${#LATENT_FILES[@]} != 8 )); then
  echo "ERROR: expected 8 latent windows, found ${#LATENT_FILES[@]}" >&2
  exit 2
fi
SOURCE_FILE="${LATENT_FILES[${WINDOW_INDEX}]}"
SOURCE_INDEX="${SOURCE_FILE#2020_}"
SOURCE_INDEX="${SOURCE_INDEX%.h5}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"

CONTROL_NAME="repulsion_full_k8/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_control_seed${SEED}"
REPULSION_NAME="repulsion_full_k8/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_alpha${STRENGTH_TAG}_seed${SEED}"
CONTROL_RUN="outputs/sr/${CONTROL_NAME}"
REPULSION_RUN="outputs/sr/${REPULSION_NAME}"
CONTROL_SAMPLE="${CONTROL_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"
REPULSION_SAMPLE="${REPULSION_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"
METRICS_DIR="outputs/metrics/repulsion_full_k8/w$(printf '%02d' "${WINDOW_INDEX}")_alpha${STRENGTH_TAG}"

# Reuse the already-completed window-0 matched K=8 smoke control when present;
# it used exactly the same control sampler settings and seed.
SMOKE_CONTROL_RUN="outputs/sr/repulsion_smoke_k8/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_control_seed${SEED}"
SMOKE_CONTROL_SAMPLE="${SMOKE_CONTROL_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"
if [[ "${FORCE_CONTROL}" == "0" && -s "${SMOKE_CONTROL_SAMPLE}" && -s "${SMOKE_CONTROL_RUN}/metadata.json" ]]; then
  CONTROL_RUN="${SMOKE_CONTROL_RUN}"
  CONTROL_SAMPLE="${SMOKE_CONTROL_SAMPLE}"
  echo "Reusing prior matched K=8 smoke control: ${CONTROL_SAMPLE}"
fi

CONFIG="${DIFFUSION_CONFIG:-checkpoints/prior/config.yaml}"
CKPT="${PRIOR_CHECKPOINT:-checkpoints/prior/trigflow_adaptive_ema_019901.pt}"
AE="${AE_MODEL_PATH:-checkpoints/autoencoder/ae_model.pt}"
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

echo "Window:              ${WINDOW_INDEX} (source ${SOURCE_INDEX})"
echo "K:                   ${K}"
echo "Seed:                ${SEED}"
echo "Repulsion strength:  ${REPULSION_STRENGTH}"
echo "Repulsion sigma band:${REPULSION_MIN_SIGMA}..${REPULSION_MAX_SIGMA}"
echo "Particle execution:  full joint K=8 batch"

if [[ -s "${CONTROL_SAMPLE}" && -s "${CONTROL_RUN}/metadata.json" && "${FORCE_CONTROL}" == "0" ]]; then
  echo "Using matched K=8 control: ${CONTROL_SAMPLE}"
else
  echo "Generating matched independent K=8 control..."
  CONTROL_RUN="outputs/sr/${CONTROL_NAME}"
  CONTROL_SAMPLE="${CONTROL_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"
  srun python -m era5_videogen.generate_posterior_repulsion \
    "${COMMON_ARGS[@]}" \
    --name "${CONTROL_NAME}"
fi

if [[ -s "${REPULSION_SAMPLE}" && -s "${REPULSION_RUN}/metadata.json" && "${FORCE_REPULSION}" == "0" ]]; then
  echo "Reusing joint K=8 repulsion samples: ${REPULSION_SAMPLE}"
else
  echo "Generating joint K=8 repulsion-guided particles..."
  srun python -m era5_videogen.generate_posterior_repulsion \
    "${COMMON_ARGS[@]}" \
    --name "${REPULSION_NAME}" \
    --repulsion_strength "${REPULSION_STRENGTH}" \
    --repulsion_bandwidth 0 \
    --repulsion_min_sigma "${REPULSION_MIN_SIGMA}" \
    --repulsion_max_sigma "${REPULSION_MAX_SIGMA}" \
    --repulsion_schedule sine \
    --repulsion_spatial_pool "${REPULSION_SPATIAL_POOL}"
fi

echo "Evaluating matched K=8 ensembles..."
srun python scripts/evaluate_repulsion_full_k8.py \
  --repulsion-run "${REPULSION_RUN}" \
  --control-run "${CONTROL_RUN}" \
  --data-root data \
  --window "${WINDOW_INDEX}" \
  --output-dir "${METRICS_DIR}"

echo "Final K=8 window ${WINDOW_INDEX} completed."
echo "Summary: ${REPO_DIR}/${METRICS_DIR}/summary.json"
