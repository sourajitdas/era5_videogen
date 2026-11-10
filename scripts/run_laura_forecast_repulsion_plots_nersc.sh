#!/bin/bash -l
# Plot the completed 2/3/4-day Laura repulsion forecast experiments.
#SBATCH --time=01:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J LauraFcFig
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
BUNDLE="${LAURA_FORECAST_BUNDLE:-${REPO_DIR}/data/hurricane_laura/laura_w941_forecast_bundle_v1}"
K="${K:-8}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.10}"
REPULSION_MASK_MODE="${REPULSION_MASK_MODE:-frames}"
REPULSION_OBSERVATION_BUFFER="${REPULSION_OBSERVATION_BUFFER:-0}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"
MASK_TAG=""
if [[ "${REPULSION_MASK_MODE}" != "frames" ]]; then
  MASK_TAG="_mask${REPULSION_MASK_MODE}_buf${REPULSION_OBSERVATION_BUFFER}"
fi
RUN_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
OUTPUT_DIR="${RUN_ROOT}/figures_k${K}_alpha${STRENGTH_TAG}${MASK_TAG}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-fc-fig-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

run_dir() {
  local days="$1"
  echo "${RUN_ROOT}/laura_w941_forecast_obs${days}d_repulsion_k${K}_alpha${STRENGTH_TAG}${MASK_TAG}"
}

for days in 2 3 4; do
  sample="$(run_dir "${days}")/sample_0941/samples.npy"
  [[ -s "${sample}" ]] || { echo "missing completed run: ${sample}" >&2; exit 2; }
done

OBS_RAW="${BUNDLE}/data/observations/global_joint_2020_year.pkl"
srun python scripts/laura_forecast_repulsion_figs.py \
  --bundle "${BUNDLE}" \
  --run-2d "$(run_dir 2)" \
  --run-3d "$(run_dir 3)" \
  --run-4d "$(run_dir 4)" \
  --observations-pkl "${OBS_RAW}" \
  --output-dir "${OUTPUT_DIR}"

echo "Figure-17 analogue: ${OUTPUT_DIR}/figure17_repulsion_tracks.png"
echo "Figure-19 analogue: ${OUTPUT_DIR}/figure19_repulsion_u10_distribution.png"
echo "Provenance:         ${OUTPUT_DIR}/laura_forecast_repulsion_figures.json"
