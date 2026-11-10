#!/bin/bash -l
# Compare the original frame-only mask with the spatial observation-complement mask.
#SBATCH --time=01:00:00
#SBATCH -C gpu
#SBATCH --account=m5021_g
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -J LauraMaskCmp
#SBATCH --output=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%j.out
#SBATCH --error=/global/cfs/cdirs/m5021/sjdas/era5_videogen/logs/laura_forecast_repulsion/%x_%j.err

set -euo pipefail

module load conda
conda activate "${CONDA_ENV_NAME:-era5_env}"

REPO_DIR="${ERA5_VIDEOGEN_REPO_DIR:-/global/cfs/cdirs/m5021/sjdas/era5_videogen}"
BUNDLE="${LAURA_FORECAST_BUNDLE:-${REPO_DIR}/data/hurricane_laura/laura_w941_forecast_bundle_v1}"
OBS_DAYS="${OBS_DAYS:-4}"
K="${K:-8}"
REPULSION_STRENGTH="${REPULSION_STRENGTH:-0.10}"
REPULSION_OBSERVATION_BUFFER="${REPULSION_OBSERVATION_BUFFER:-0}"
STRENGTH_TAG="${REPULSION_STRENGTH//./p}"
MASK_TAG="_maskspatial_buf${REPULSION_OBSERVATION_BUFFER}"
RUN_ROOT="${REPO_DIR}/outputs/hurricane_laura/forecast_repulsion"
FRAME_RUN="${RUN_ROOT}/laura_w941_forecast_obs${OBS_DAYS}d_repulsion_k${K}_alpha${STRENGTH_TAG}"
SPATIAL_RUN="${RUN_ROOT}/laura_w941_forecast_obs${OBS_DAYS}d_repulsion_k${K}_alpha${STRENGTH_TAG}${MASK_TAG}"
OUTPUT_DIR="${RUN_ROOT}/mask_comparison_obs${OBS_DAYS}d_k${K}_alpha${STRENGTH_TAG}_buf${REPULSION_OBSERVATION_BUFFER}"

case "${OBS_DAYS}" in
  2|3|4) ;;
  *) echo "OBS_DAYS must be 2, 3, or 4" >&2; exit 2 ;;
esac

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/laura-mask-comparison-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

for run in "${FRAME_RUN}" "${SPATIAL_RUN}"; do
  sample="${run}/sample_0941/samples.npy"
  [[ -s "${sample}" ]] || { echo "missing completed run: ${sample}" >&2; exit 2; }
done

OBS_RAW="${BUNDLE}/data/observations/global_joint_2020_year.pkl"
srun python scripts/compare_laura_repulsion_masks.py \
  --bundle "${BUNDLE}" \
  --frame-run "${FRAME_RUN}" \
  --spatial-run "${SPATIAL_RUN}" \
  --obs-days "${OBS_DAYS}" \
  --observations-pkl "${OBS_RAW}" \
  --output-dir "${OUTPUT_DIR}"

echo "Comparison figure: ${OUTPUT_DIR}/repulsion_mask_comparison_obs${OBS_DAYS}d.png"
echo "Comparison metrics: ${OUTPUT_DIR}/repulsion_mask_comparison_obs${OBS_DAYS}d.json"
