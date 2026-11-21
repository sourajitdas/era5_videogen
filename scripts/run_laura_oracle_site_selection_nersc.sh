#!/bin/bash -l
# Retrospective two-day Laura observation-site selection and posterior rerun.
# The job requires an A100-80GB because K=8 joint decoder guidance exceeds 40GB.
#SBATCH --time=08:00:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J LauraOracleObs
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
BUNDLE="${LAURA_FORECAST_BUNDLE:-${REPO_DIR}/data/hurricane_laura/laura_w941_forecast_bundle_v1}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-oracle-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

SOURCE_INDEX=941
DATASET_INDEX=0
OBS_DAYS=2
OBS_FRAMES="0-7"
K="${K:-8}"
NUM_STEPS="${NUM_STEPS:-50}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.005}"
HURRICANE_DESCRIPTOR_HORIZON="${HURRICANE_DESCRIPTOR_HORIZON:-9}"
SITE_BUDGET="${SITE_BUDGET:-64}"
SELECTION_MODE="${SELECTION_MODE:-oracle}"
RANDOM_SEED="${RANDOM_SEED:-20261111}"
SITE_ROUND_DECIMALS="${SITE_ROUND_DECIMALS:-3}"
MIN_SITE_RECORDS="${MIN_SITE_RECORDS:-4}"
MIN_SITE_SEPARATION_DEG="${MIN_SITE_SEPARATION_DEG:-0.5}"
TARGET_SPREAD_RATIO="${TARGET_SPREAD_RATIO:-0.75}"
SPREAD_PENALTY="${SPREAD_PENALTY:-4.0}"
WEIGHT_TEMPERATURE="${WEIGHT_TEMPERATURE:-1.0}"
KEEP_OUTSIDE_REGION="${KEEP_OUTSIDE_REGION:-1}"
FORCE_SELECTION="${FORCE_SELECTION:-0}"
FORCE_SAMPLE="${FORCE_SAMPLE:-0}"

STRENGTH_TAG="${REPULSION_STRENGTH//./p}"
BASELINE_RUN="${ORACLE_BASELINE_RUN:-${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion/laura_w941_forecast_obs2d_repulsion_k8_alpha0p005_hurricane_h9}"
OUTPUT_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
case "${SELECTION_MODE}" in
  oracle) SELECTION_TAG="oracle_sites${SITE_BUDGET}" ;;
  random) SELECTION_TAG="random_sites${SITE_BUDGET}_seed${RANDOM_SEED}" ;;
  *) echo "SELECTION_MODE must be oracle or random" >&2; exit 2 ;;
esac
EXPERIMENT_TAG="${SELECTION_TAG}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}"
RUN_NAME="laura_w941_forecast_obs2d_${EXPERIMENT_TAG}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
ANALYSIS_DIR="${OUTPUT_ROOT}/${EXPERIMENT_TAG}"
MANIFEST="${ANALYSIS_DIR}/oracle_observation_selection.json"
RUNTIME_CONFIG="${RUN_DIR}/runtime_config.yaml"
SAMPLE_FILE="${RUN_DIR}/sample_0941/samples.npy"

CKPT="${BUNDLE}/models/dj_checkpoint-019901.pt"
AE="${REPO_DIR}/checkpoints/autoencoder/ae_model.pt"
OBS="${BUNDLE}/data/observations/global_joint_2020_year_superob.pkl"
mkdir -p "${RUN_DIR}" "${ANALYSIS_DIR}"

python scripts/prepare_laura_forecast_repulsion.py \
  --bundle "${BUNDLE}" \
  --autoencoder "${AE}" \
  --output-config "${RUNTIME_CONFIG}" \
  --source-index "${SOURCE_INDEX}"

[[ -s "${BASELINE_RUN}/sample_0941/samples.npy" ]] || {
  echo "missing completed alpha=0.005 baseline: ${BASELINE_RUN}" >&2
  exit 2
}

if [[ ! -s "${MANIFEST}" || "${FORCE_SELECTION}" == "1" ]]; then
  SELECT_ARGS=(
    --bundle "${BUNDLE}"
    --baseline-run "${BASELINE_RUN}"
    --observations-pkl "${OBS}"
    --output-manifest "${MANIFEST}"
    --source-index "${SOURCE_INDEX}"
    --obs-days "${OBS_DAYS}"
    --selection-mode "${SELECTION_MODE}"
    --random-seed "${RANDOM_SEED}"
    --budget "${SITE_BUDGET}"
    --selection-region 15 35 -100 -65
    --site-round-decimals "${SITE_ROUND_DECIMALS}"
    --min-site-records "${MIN_SITE_RECORDS}"
    --min-site-separation-deg "${MIN_SITE_SEPARATION_DEG}"
    --target-spread-ratio "${TARGET_SPREAD_RATIO}"
    --spread-penalty "${SPREAD_PENALTY}"
    --weight-temperature "${WEIGHT_TEMPERATURE}"
  )
  if [[ "${KEEP_OUTSIDE_REGION}" == "1" ]]; then
    SELECT_ARGS+=(--keep-outside-region)
  else
    SELECT_ARGS+=(--no-keep-outside-region)
  fi
  echo "Selecting an oracle subset of the first-two-day observation sites..."
  srun python scripts/select_laura_oracle_observations.py "${SELECT_ARGS[@]}" \
    2>&1 | tee "${ANALYSIS_DIR}/selection.log"
else
  echo "Reusing observation-selection manifest: ${MANIFEST}"
fi

echo "Oracle selection manifest: ${MANIFEST}"
echo "This is a retrospective diagnostic; future ERA5 is not passed to the sampler."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OBS_VARS="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,geopotential_500,geopotential_850,u_component_of_wind_500,u_component_of_wind_850,v_component_of_wind_500,v_component_of_wind_850,temperature_500,temperature_850,specific_humidity_500,specific_humidity_850"
SAMPLER_ARGS=(
  --diffusion_config "${RUNTIME_CONFIG}"
  --diffusion_checkpoint "${CKPT}"
  --ae_model_path "${AE}"
  --igra_pkl "${OBS}"
  --observation_selection_manifest "${MANIFEST}"
  --obs_variables "${OBS_VARS}"
  --output_dir "${OUTPUT_ROOT}"
  --name "${RUN_NAME}"
  --sample_index "${DATASET_INDEX}"
  --num_posterior_samples "${K}"
  --seed 17
  --obs_frames "${OBS_FRAMES}"
  --num_steps "${NUM_STEPS}"
  --guidance_scale 0.4
  --guide_norm dsg
  --gamma 2e-6
  --meas_std 5e-4
  --corrector_steps 1
  --corrector_snr 0.3
  --corrector_max_sigma 3
  --noise_scale 0.0
  --repulsion_strength "${REPULSION_STRENGTH}"
  --repulsion_bandwidth 0
  --repulsion_min_sigma 0.05
  --repulsion_max_sigma 5.0
  --repulsion_schedule sine
  --repulsion_spatial_pool 8
  --repulsion_mask_mode hurricane
  --hurricane_descriptor_horizon "${HURRICANE_DESCRIPTOR_HORIZON}"
  --hurricane_region_lat 15 35
  --hurricane_region_lon -100 -65
  --no_net_grad
  --log_every 2
)

if [[ ! -s "${SAMPLE_FILE}" || ! -s "${RUN_DIR}/metadata.json" || "${FORCE_SAMPLE}" == "1" ]]; then
  echo "Running K=${K} posterior sampling with the selected observations..."
  srun python -m era5_videogen.generate_posterior_igra_repulsion \
    "${SAMPLER_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/sampling.log"
else
  echo "Reusing completed selected-observation run: ${SAMPLE_FILE}"
fi

echo "Plotting selected/dropped sites and posterior tracks..."
srun python scripts/plot_laura_oracle_selection.py \
  --bundle "${BUNDLE}" \
  --baseline-run "${BASELINE_RUN}" \
  --selected-run "${RUN_DIR}" \
  --selection-manifest "${MANIFEST}" \
  --output-dir "${ANALYSIS_DIR}"

echo "Completed retrospective oracle site-selection experiment."
echo "Figure:  ${ANALYSIS_DIR}/oracle_site_selection_tracks.png"
echo "Metrics: ${ANALYSIS_DIR}/oracle_site_selection_metrics.json"
