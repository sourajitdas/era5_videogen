#!/bin/bash -l
# Aggregate the seed-17 oracle run plus seeds 18--29 into 100 members.
#SBATCH --time=02:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J LauraOracleAgg
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-oracle-aggregate-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

SITE_BUDGET="${SITE_BUDGET:-192}"
TARGET_MEMBERS="${TARGET_MEMBERS:-100}"
TRACK_DENSITY_BANDWIDTH_KM="${TRACK_DENSITY_BANDWIDTH_KM:-0}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.005}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"
HURRICANE_DESCRIPTOR_HORIZON="${HURRICANE_DESCRIPTOR_HORIZON:-9}"
OUTPUT_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
SELECTION_DIR="${OUTPUT_ROOT}/oracle_sites${SITE_BUDGET}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}"
MANIFEST="${ORACLE_MANIFEST:-${SELECTION_DIR}/oracle_observation_selection.json}"
REFERENCE_RUN="${ORACLE_REFERENCE_RUN:-${OUTPUT_ROOT}/laura_w941_forecast_obs2d_oracle_sites${SITE_BUDGET}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}}"
ANALYSIS_DIR="${OUTPUT_ROOT}/oracle_sites${SITE_BUDGET}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}_ensemble${TARGET_MEMBERS}"

[[ -s "${MANIFEST}" ]] || { echo "missing oracle manifest: ${MANIFEST}" >&2; exit 2; }
[[ -s "${REFERENCE_RUN}/sample_0941/samples.npy" ]] || {
  echo "missing seed-17 oracle reference run: ${REFERENCE_RUN}" >&2
  exit 2
}

BATCH_ARGS=()
for SEED in $(seq 18 29); do
  SEED_TAG="$(printf '%04d' "${SEED}")"
  RUN="${OUTPUT_ROOT}/laura_w941_forecast_obs2d_oracle_sites${SITE_BUDGET}_k8_seed${SEED_TAG}_alpha${STRENGTH_TAG}_hurricane_h${HURRICANE_DESCRIPTOR_HORIZON}"
  SAMPLE="${RUN}/sample_0941/samples.npy"
  [[ -s "${SAMPLE}" ]] || { echo "missing completed batch: ${SAMPLE}" >&2; exit 2; }
  BATCH_ARGS+=(--batch-run "${RUN}")
done

mkdir -p "${ANALYSIS_DIR}"
srun python scripts/aggregate_laura_oracle_ensemble.py \
  --bundle "${BUNDLE}" \
  --reference-run "${REFERENCE_RUN}" \
  "${BATCH_ARGS[@]}" \
  --selection-manifest "${MANIFEST}" \
  --target-members "${TARGET_MEMBERS}" \
  --track-density-bandwidth-km "${TRACK_DENSITY_BANDWIDTH_KM}" \
  --output-dir "${ANALYSIS_DIR}"

echo "Completed ${TARGET_MEMBERS}-member oracle-mask aggregation."
echo "Figure:  ${ANALYSIS_DIR}/oracle_ensemble${TARGET_MEMBERS}_density_convergence.png"
echo "Metrics: ${ANALYSIS_DIR}/oracle_ensemble${TARGET_MEMBERS}_density_metrics.json"
