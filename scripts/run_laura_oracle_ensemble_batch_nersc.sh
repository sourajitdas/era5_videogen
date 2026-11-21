#!/bin/bash -l
# Generate one K=8 batch for the 100-member Laura oracle-mask diagnostic.
# Submit as --array=1-12; seed 17 is reused from the original oracle run.
#SBATCH --time=06:00:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=1-12%4
#SBATCH -J LauraOracle100
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-oracle100-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

BATCH_INDEX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
case "${BATCH_INDEX}" in
  ''|*[!0-9]*) echo "batch index must be an integer from 1 through 12" >&2; exit 2 ;;
esac
if (( BATCH_INDEX < 1 || BATCH_INDEX > 12 )); then
  echo "batch index must be from 1 through 12" >&2
  exit 2
fi

SOURCE_INDEX=941
DATASET_INDEX=0
K=8
NUM_STEPS="${NUM_STEPS:-50}"
SEED=$((17 + BATCH_INDEX))
SEED_TAG="$(printf '%04d' "${SEED}")"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.005}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"
HURRICANE_DESCRIPTOR_HORIZON="${HURRICANE_DESCRIPTOR_HORIZON:-9}"
SITE_BUDGET="${SITE_BUDGET:-192}"
FORCE="${FORCE:-0}"

OUTPUT_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
SELECTION_DIR="${OUTPUT_ROOT}/oracle_sites${SITE_BUDGET}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}"
MANIFEST="${ORACLE_MANIFEST:-${SELECTION_DIR}/oracle_observation_selection.json}"
RUN_NAME="laura_w941_forecast_obs2d_oracle_sites${SITE_BUDGET}_k${K}_seed${SEED_TAG}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RUNTIME_CONFIG="${RUN_DIR}/runtime_config.yaml"
SAMPLE_FILE="${RUN_DIR}/sample_0941/samples.npy"

CKPT="${BUNDLE}/models/dj_checkpoint-019901.pt"
AE="${REPO_DIR}/checkpoints/autoencoder/ae_model.pt"
OBS="${BUNDLE}/data/observations/global_joint_2020_year_superob.pkl"
mkdir -p "${RUN_DIR}"

[[ -s "${MANIFEST}" ]] || { echo "missing oracle manifest: ${MANIFEST}" >&2; exit 2; }
python scripts/prepare_laura_forecast_repulsion.py \
  --bundle "${BUNDLE}" \
  --autoencoder "${AE}" \
  --output-config "${RUNTIME_CONFIG}" \
  --source-index "${SOURCE_INDEX}"

echo "Oracle-mask ensemble batch ${BATCH_INDEX}/12; seed=${SEED}; K=${K}"
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
  --seed "${SEED}"
  --obs_frames 0-7
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

if [[ -s "${SAMPLE_FILE}" && -s "${RUN_DIR}/metadata.json" && "${FORCE}" == "0" ]]; then
  echo "Reusing completed batch: ${SAMPLE_FILE}"
else
  srun python -m era5_videogen.generate_posterior_igra_repulsion \
    "${SAMPLER_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/sampling.log"
fi

echo "Completed seed ${SEED}: ${SAMPLE_FILE}"
