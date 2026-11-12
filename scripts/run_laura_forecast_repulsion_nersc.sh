#!/bin/bash -l
# Reproduce the three Figure-17/19 forecast conditions with joint repulsion.
# Submit as a Slurm array: task 2/3/4 assimilates 2/3/4 days, respectively.
#SBATCH --time=06:00:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=2-4
#SBATCH -J LauraFcRep
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%A_%a.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%A_%a.err

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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-fc-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

OBS_DAYS="${OBS_DAYS:-${SLURM_ARRAY_TASK_ID:-${1:-}}}"
case "${OBS_DAYS}" in
  2|3|4) ;;
  *) echo "OBS_DAYS must be 2, 3, or 4 (received '${OBS_DAYS}')" >&2; exit 2 ;;
esac

SOURCE_INDEX=941
DATASET_INDEX=0
K="${K:-8}"
NUM_STEPS="${NUM_STEPS:-50}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.10}"
REPULSION_MIN_SIGMA="${REPULSION_MIN_SIGMA:-0.05}"
REPULSION_MAX_SIGMA="${REPULSION_MAX_SIGMA:-5.0}"
REPULSION_SPATIAL_POOL="${REPULSION_SPATIAL_POOL:-8}"
REPULSION_MASK_MODE="${REPULSION_MASK_MODE:-frames}"
REPULSION_OBSERVATION_BUFFER="${REPULSION_OBSERVATION_BUFFER:-0}"
REPULSION_SPLIT_BETA="${REPULSION_SPLIT_BETA:-1.0}"
HURRICANE_DESCRIPTOR_HORIZON="${HURRICANE_DESCRIPTOR_HORIZON:-9}"
HURRICANE_REGION_LAT_MIN="${HURRICANE_REGION_LAT_MIN:-15}"
HURRICANE_REGION_LAT_MAX="${HURRICANE_REGION_LAT_MAX:-35}"
HURRICANE_REGION_LON_MIN="${HURRICANE_REGION_LON_MIN:--100}"
HURRICANE_REGION_LON_MAX="${HURRICANE_REGION_LON_MAX:--65}"
NO_NET_GRAD="${NO_NET_GRAD:-1}"
FORCE="${FORCE:-0}"
VERIFY_MODEL_HASHES="${VERIFY_MODEL_HASHES:-0}"

OBS_COUNT=$((OBS_DAYS * 4))
OBS_LAST=$((OBS_COUNT - 1))
OBS_FRAMES="0-${OBS_LAST}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"
case "${REPULSION_MASK_MODE}" in
  frames|spatial|split|hurricane|all) ;;
  *) echo "REPULSION_MASK_MODE must be frames, spatial, split, hurricane, or all" >&2; exit 2 ;;
esac
if (( REPULSION_OBSERVATION_BUFFER < 0 )); then
  echo "REPULSION_OBSERVATION_BUFFER must be non-negative" >&2
  exit 2
fi
MASK_TAG=""
if [[ "${REPULSION_MASK_MODE}" != "frames" ]]; then
  MASK_TAG="_mask${REPULSION_MASK_MODE}_buf${REPULSION_OBSERVATION_BUFFER}"
fi
if [[ "${REPULSION_MASK_MODE}" == "split" ]]; then
  BETA_TAG="${REPULSION_SPLIT_BETA//./p}"
  MASK_TAG="${MASK_TAG}_beta${BETA_TAG}"
fi
if [[ "${REPULSION_MASK_MODE}" == "hurricane" ]]; then
  MASK_TAG="_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}"
fi

CKPT="${BUNDLE}/models/dj_checkpoint-019901.pt"
AE="${REPO_DIR}/checkpoints/autoencoder/ae_model.pt"
OBS="${BUNDLE}/data/observations/global_joint_2020_year_superob.pkl"
OUTPUT_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
RUN_NAME="laura_w941_forecast_obs${OBS_DAYS}d_repulsion_k${K}_alpha${STRENGTH_TAG}${MASK_TAG}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RUNTIME_CONFIG="${RUN_DIR}/runtime_config.yaml"
mkdir -p "${RUN_DIR}"

PREP_ARGS=(
  --bundle "${BUNDLE}"
  --autoencoder "${AE}"
  --output-config "${RUNTIME_CONFIG}"
  --source-index "${SOURCE_INDEX}"
)
if (( VERIFY_MODEL_HASHES == 1 )); then
  PREP_ARGS+=(--verify-model-hashes)
fi
python scripts/prepare_laura_forecast_repulsion.py "${PREP_ARGS[@]}"

echo "Source index:          ${SOURCE_INDEX} (2020-08-23 06Z)"
echo "Observed days/frames:  ${OBS_DAYS} / ${OBS_FRAMES}"
echo "Joint particle count:  ${K}"
echo "Repulsion strength:    ${REPULSION_STRENGTH}"
echo "Repulsion sigma band:  ${REPULSION_MIN_SIGMA}..${REPULSION_MAX_SIGMA}"
echo "Repulsion mask:        ${REPULSION_MASK_MODE} (buffer ${REPULSION_OBSERVATION_BUFFER})"
echo "Split objective beta:  ${REPULSION_SPLIT_BETA}"
echo "Hurricane descriptor:  horizon ${HURRICANE_DESCRIPTOR_HORIZON}; lat ${HURRICANE_REGION_LAT_MIN}..${HURRICANE_REGION_LAT_MAX}; lon ${HURRICANE_REGION_LON_MIN}..${HURRICANE_REGION_LON_MAX}"
echo "Decoder-only guidance: ${NO_NET_GRAD}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OBS_VARS="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,geopotential_500,geopotential_850,u_component_of_wind_500,u_component_of_wind_850,v_component_of_wind_500,v_component_of_wind_850,temperature_500,temperature_850,specific_humidity_500,specific_humidity_850"
SAMPLER_ARGS=(
  --diffusion_config "${RUNTIME_CONFIG}"
  --diffusion_checkpoint "${CKPT}"
  --ae_model_path "${AE}"
  --igra_pkl "${OBS}"
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
  --repulsion_min_sigma "${REPULSION_MIN_SIGMA}"
  --repulsion_max_sigma "${REPULSION_MAX_SIGMA}"
  --repulsion_schedule sine
  --repulsion_spatial_pool "${REPULSION_SPATIAL_POOL}"
  --repulsion_mask_mode "${REPULSION_MASK_MODE}"
  --repulsion_observation_buffer "${REPULSION_OBSERVATION_BUFFER}"
  --repulsion_split_beta "${REPULSION_SPLIT_BETA}"
  --hurricane_descriptor_horizon "${HURRICANE_DESCRIPTOR_HORIZON}"
  --hurricane_region_lat "${HURRICANE_REGION_LAT_MIN}" "${HURRICANE_REGION_LAT_MAX}"
  --hurricane_region_lon "${HURRICANE_REGION_LON_MIN}" "${HURRICANE_REGION_LON_MAX}"
  --log_every 2
)
if (( NO_NET_GRAD == 1 )); then
  SAMPLER_ARGS+=(--no_net_grad)
fi

SAMPLE_FILE="${RUN_DIR}/sample_0941/samples.npy"
if [[ -s "${SAMPLE_FILE}" && -s "${RUN_DIR}/metadata.json" && "${FORCE}" == "0" ]]; then
  echo "Reusing completed forecast sample: ${SAMPLE_FILE}"
else
  echo "Running ${OBS_DAYS}-day-observed joint-repulsion forecast..."
  srun python -m era5_videogen.generate_posterior_igra_repulsion \
    "${SAMPLER_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/sampling.log"
fi

echo "Completed ${OBS_DAYS}-day-observed Laura forecast."
echo "Samples: ${SAMPLE_FILE}"
