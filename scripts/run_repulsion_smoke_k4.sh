#!/bin/bash -l
# Step 3 smoke test: matched independent and repulsion-guided K=4 ensembles.
#
# Optional overrides at submission time:
#   sbatch --export=ALL,REPULSION_STRENGTH=0.10 scripts/run_repulsion_smoke_k4.sh
#   sbatch --export=ALL,WINDOW_INDEX=1 scripts/run_repulsion_smoke_k4.sh
#SBATCH --time=04:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J SR_repK4
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_repulsion/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/sr_repulsion/%x_%j.err

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

# NERSC login environments can carry more than one mutually exclusive Slurm
# memory default into the batch job.  The allocation is already established;
# remove those inherited request variables before launching job steps.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

WINDOW_INDEX="${WINDOW_INDEX:-0}"
SEED="${SEED:-17}"
K="${K:-4}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.05}"
REPULSION_MIN_SIGMA="${REPULSION_MIN_SIGMA:-0.05}"
REPULSION_MAX_SIGMA="${REPULSION_MAX_SIGMA:-5.0}"
REPULSION_SPATIAL_POOL="${REPULSION_SPATIAL_POOL:-8}"
NO_NET_GRAD="${NO_NET_GRAD:-1}"
FORCE_CONTROL="${FORCE_CONTROL:-0}"

if (( K != 4 )); then
  echo "ERROR: this smoke evaluator expects K=4, got ${K}" >&2
  exit 2
fi

bash scripts/check_main8_assets.sh data
bash scripts/check_demo_assets.sh
python scripts/verify_baseline_ensemble.py

mapfile -t LATENT_FILES < <(
  find data/latents/test -maxdepth 1 -type f -name '*.h5' -printf '%f\n' | sort
)
if (( WINDOW_INDEX < 0 || WINDOW_INDEX >= ${#LATENT_FILES[@]} )); then
  echo "ERROR: WINDOW_INDEX=${WINDOW_INDEX} is outside 0..$((${#LATENT_FILES[@]} - 1))" >&2
  exit 2
fi
SOURCE_FILE="${LATENT_FILES[${WINDOW_INDEX}]}"
SOURCE_INDEX="${SOURCE_FILE#2020_}"
SOURCE_INDEX="${SOURCE_INDEX%.h5}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"

CONTROL_NAME="repulsion_smoke_k4/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_control_seed${SEED}"
REPULSION_NAME="repulsion_smoke_k4/w$(printf '%02d' "${WINDOW_INDEX}")_src${SOURCE_INDEX}_alpha${STRENGTH_TAG}_seed${SEED}"
CONTROL_RUN="outputs/sr/${CONTROL_NAME}"
REPULSION_RUN="outputs/sr/${REPULSION_NAME}"
METRICS_DIR="outputs/metrics/repulsion_smoke_k4/w$(printf '%02d' "${WINDOW_INDEX}")_alpha${STRENGTH_TAG}"

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
)
if (( NO_NET_GRAD == 1 )); then
  # K=4 full DPS retains four denoiser graphs and is unlikely to fit a 40-GB
  # A100. Decoder-only guidance retains the frozen prior velocity and is used
  # identically in the control and repulsion arms of this smoke comparison.
  COMMON_ARGS+=(--no_net_grad)
fi

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

echo "Window:              ${WINDOW_INDEX} (source ${SOURCE_INDEX})"
echo "K:                   ${K}"
echo "Seed:                ${SEED}"
echo "Repulsion strength:  ${REPULSION_STRENGTH}"
echo "Repulsion sigma band:${REPULSION_MIN_SIGMA}..${REPULSION_MAX_SIGMA}"
echo "Decoder-only guide:  ${NO_NET_GRAD}"

CONTROL_SAMPLE="${CONTROL_RUN}/sample_$(printf '%04d' "${WINDOW_INDEX}")/samples.npy"
if [[ -s "${CONTROL_SAMPLE}" && "${FORCE_CONTROL}" == "0" ]]; then
  echo "Reusing matched K=4 independent control: ${CONTROL_SAMPLE}"
else
  echo "Running matched K=4 independent control..."
  srun python -m era5_videogen.generate_posterior_repulsion \
    "${COMMON_ARGS[@]}" \
    --name "${CONTROL_NAME}"
fi

echo "Running K=4 repulsion-guided particles..."
srun python -m era5_videogen.generate_posterior_repulsion \
  "${COMMON_ARGS[@]}" \
  --name "${REPULSION_NAME}" \
  --repulsion_strength "${REPULSION_STRENGTH}" \
  --repulsion_bandwidth 0 \
  --repulsion_min_sigma "${REPULSION_MIN_SIGMA}" \
  --repulsion_max_sigma "${REPULSION_MAX_SIGMA}" \
  --repulsion_schedule sine \
  --repulsion_spatial_pool "${REPULSION_SPATIAL_POOL}"

echo "Evaluating the matched smoke ensembles..."
srun python scripts/evaluate_repulsion_smoke.py \
  --repulsion-run "${REPULSION_RUN}" \
  --control-run "${CONTROL_RUN}" \
  --baseline-root outputs/sr/baseline_independent \
  --data-root data \
  --window "${WINDOW_INDEX}" \
  --output-dir "${METRICS_DIR}"

echo "Step 3 K=4 smoke experiment completed."
echo "Summary: ${REPO_DIR}/${METRICS_DIR}/summary.json"
echo "Plot:    ${REPO_DIR}/${METRICS_DIR}/smoke_comparison.png"
