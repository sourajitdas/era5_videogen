#!/bin/bash -l
# Hurricane Laura w932: reference real-observation hindcast with joint K=8 repulsion.
# The transferred reference files are read-only inputs; all new products are written
# under outputs/hurricane_laura and carry distinct repulsion names.
#SBATCH --time=04:00:00
#SBATCH -C gpu&hbm80g
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J LauraRepK8
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_repulsion/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_repulsion/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
BUNDLE="${LAURA_BUNDLE:-${REPO_DIR}/data/hurricane_laura/laura_w932_k8_bundle_v1}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export SLURM_CPU_BIND=cores
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

SOURCE_INDEX=932
DATASET_INDEX=0
SMOKE="${SMOKE:-0}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.10}"
REPULSION_MIN_SIGMA="${REPULSION_MIN_SIGMA:-0.05}"
REPULSION_MAX_SIGMA="${REPULSION_MAX_SIGMA:-5.0}"
REPULSION_SPATIAL_POOL="${REPULSION_SPATIAL_POOL:-8}"
REPULSION_MASK_MODE="${REPULSION_MASK_MODE:-frames}"
REPULSION_OBSERVATION_BUFFER="${REPULSION_OBSERVATION_BUFFER:-0}"
REPULSION_SPLIT_BETA="${REPULSION_SPLIT_BETA:-1.0}"
HURRICANE_DESCRIPTOR_HORIZON="${HURRICANE_DESCRIPTOR_HORIZON:-0}"
HURRICANE_REGION_LAT_MIN="${HURRICANE_REGION_LAT_MIN:-15}"
HURRICANE_REGION_LAT_MAX="${HURRICANE_REGION_LAT_MAX:-35}"
HURRICANE_REGION_LON_MIN="${HURRICANE_REGION_LON_MIN:--100}"
HURRICANE_REGION_LON_MAX="${HURRICANE_REGION_LON_MAX:--65}"
NO_NET_GRAD="${NO_NET_GRAD:-1}"
FORCE="${FORCE:-0}"
VERIFY_MODEL_HASHES="${VERIFY_MODEL_HASHES:-0}"
PLAIN_MAP="${PLAIN_MAP:-0}"

if (( SMOKE == 1 )); then
  K=2
  NUM_STEPS=3
  CORRECTOR_STEPS=0
  RUN_KIND="smoke_k2"
else
  K=8
  NUM_STEPS=50
  CORRECTOR_STEPS=1
  RUN_KIND="k8"
fi

CKPT="${BUNDLE}/models/dj_checkpoint-019901.pt"
AE="${REPO_DIR}/checkpoints/autoencoder/ae_model.pt"
OBS="${BUNDLE}/data/observations/global_joint_2020_year_superob.pkl"
OUTPUT_ROOT="${REPO_DIR}/outputs/hurricane_laura"
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
RUN_NAME="laura_w932_last16_repulsion_${RUN_KIND}_alpha${STRENGTH_TAG}${MASK_TAG}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
FIG_DIR="${RUN_DIR}/comparison_with_independent_control"
RUNTIME_CONFIG="${RUN_DIR}/runtime_config.yaml"

mkdir -p "${RUN_DIR}" "${FIG_DIR}"
PREP_ARGS=(
  --bundle "${BUNDLE}"
  --autoencoder "${AE}"
  --output-config "${RUNTIME_CONFIG}"
  --source-index "${SOURCE_INDEX}"
)
if (( VERIFY_MODEL_HASHES == 1 )); then
  PREP_ARGS+=(--verify-model-hashes)
fi
python scripts/prepare_laura_repulsion.py "${PREP_ARGS[@]}"

echo "Laura bundle:          ${BUNDLE}"
echo "Dataset/source index:  ${DATASET_INDEX}/${SOURCE_INDEX}"
echo "K:                     ${K}"
echo "Observation frames:    16-31"
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
  --obs_frames 16-31
  --num_steps "${NUM_STEPS}"
  --guidance_scale 0.6
  --guide_norm dsg
  --gamma 2e-6
  --meas_std 5e-4
  --corrector_steps "${CORRECTOR_STEPS}"
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

SAMPLE_FILE="${RUN_DIR}/sample_0932/samples.npy"
if [[ -s "${SAMPLE_FILE}" && -s "${RUN_DIR}/metadata.json" && "${FORCE}" == "0" ]]; then
  echo "Reusing completed repulsion sample: ${SAMPLE_FILE}"
else
  echo "Running joint K=${K} Laura repulsion sampler..."
  srun python -m era5_videogen.generate_posterior_igra_repulsion \
    "${SAMPLER_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/sampling.log"
fi

if (( SMOKE == 1 )); then
  echo "Laura K=2 smoke test completed: ${SAMPLE_FILE}"
  exit 0
fi

FIG_ARGS=(
  --bundle "${BUNDLE}"
  --repulsion-run "${RUN_DIR}"
  --output-dir "${FIG_DIR}"
  --source-index "${SOURCE_INDEX}"
  --observations-pkl "${OBS}"
)
if (( PLAIN_MAP == 1 )); then
  FIG_ARGS+=(--plain-map)
fi
srun python scripts/laura_hindcast_repulsion_fig.py "${FIG_ARGS[@]}"

echo "Laura repulsion experiment completed."
echo "Samples:    ${SAMPLE_FILE}"
echo "Track plot: ${FIG_DIR}/laura_track_comparison.png"
echo "Metrics:    ${FIG_DIR}/laura_comparison.json"
