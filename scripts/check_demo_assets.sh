#!/usr/bin/env bash
# Validate the non-Git assets required by the released sampling demos.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_ROOT="${ERA5_VIDEOGEN_CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"

CONFIG="${DIFFUSION_CONFIG:-${CHECKPOINT_ROOT}/prior/config.yaml}"
PRIOR_CHECKPOINT="${PRIOR_CHECKPOINT:-${CHECKPOINT_ROOT}/prior/trigflow_adaptive_ema_019901.pt}"
AE_MODEL_PATH="${AE_MODEL_PATH:-${CHECKPOINT_ROOT}/autoencoder/ae_model.pt}"

missing=()
for path in "${CONFIG}" "${PRIOR_CHECKPOINT}" "${AE_MODEL_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    missing+=("${path}")
  fi
done

for path in \
  "${REPO_ROOT}/data/latents/metadata.json" \
  "${REPO_ROOT}/data/latents/stats.yaml" \
  "${REPO_ROOT}/data/era5/normalize_mean.npz" \
  "${REPO_ROOT}/data/era5/normalize_std.npz"; do
  if [[ ! -f "${path}" ]]; then
    missing+=("${path}")
  fi
done

if ! compgen -G "${REPO_ROOT}/data/latents/test/*.h5" >/dev/null; then
  missing+=("${REPO_ROOT}/data/latents/test/*.h5")
fi
if ! compgen -G "${REPO_ROOT}/data/era5/test/*.h5" >/dev/null; then
  missing+=("${REPO_ROOT}/data/era5/test/*.h5")
fi

if (( ${#missing[@]} > 0 )); then
  {
    echo "ERROR: the released sampling assets are incomplete."
    echo
    echo "Missing:"
    printf '  - %s\n' "${missing[@]}"
    echo
    echo "Download the checkpoints and demo-data folders from:"
    echo "  https://app.box.com/folder/407292865654"
    echo
    echo "Place them at:"
    echo "  ${REPO_ROOT}/checkpoints/"
    echo "  ${REPO_ROOT}/data/"
    echo
    echo "If checkpoints live elsewhere, set:"
    echo "  export ERA5_VIDEOGEN_CHECKPOINT_ROOT=/path/to/checkpoints"
    echo
    echo "Then validate before submitting a GPU job:"
    echo "  bash scripts/check_demo_assets.sh"
  } >&2
  exit 2
fi

echo "Sampling assets found:"
echo "  diffusion config: ${CONFIG}"
echo "  prior checkpoint: ${PRIOR_CHECKPOINT}"
echo "  autoencoder:      ${AE_MODEL_PATH}"
echo "  demo data:        ${REPO_ROOT}/data"
