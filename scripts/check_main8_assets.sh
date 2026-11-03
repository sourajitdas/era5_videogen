#!/usr/bin/env bash
# Validate the self-contained eight-window paper-evaluation bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_ROOT="${1:-${REPO_ROOT}/era5_main8_all8}"

EXPECTED_LATENTS=(
  2020_0093.h5
  2020_0244.h5
  2020_0478.h5
  2020_0554.h5
  2020_0731.h5
  2020_1024.h5
  2020_1181.h5
  2020_1296.h5
)
EXPECTED_OBS=(
  igra_2020_demo.pkl
  igra_2020_all_channels.json
  isd_2020_demo.pkl
  icoads_2020_demo.pkl
  joint_superob_2020_demo.pkl
)

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -d "${BUNDLE_ROOT}" ]] || die "bundle not found: ${BUNDLE_ROOT}"

for path in \
  "${BUNDLE_ROOT}/latents/metadata.json" \
  "${BUNDLE_ROOT}/latents/stats.yaml" \
  "${BUNDLE_ROOT}/latents/manifest.json" \
  "${BUNDLE_ROOT}/era5/normalize_mean.npz" \
  "${BUNDLE_ROOT}/era5/normalize_std.npz" \
  "${BUNDLE_ROOT}/era5/lat.npy" \
  "${BUNDLE_ROOT}/era5/lon.npy"; do
  [[ -f "${path}" ]] || die "required file missing: ${path}"
done

for name in "${EXPECTED_LATENTS[@]}"; do
  [[ -f "${BUNDLE_ROOT}/latents/test/${name}" ]] || \
    die "latent window missing: ${name}"
done

latent_count="$(find "${BUNDLE_ROOT}/latents/test" -maxdepth 1 -type f -name '*.h5' | wc -l)"
era5_count="$(find "${BUNDLE_ROOT}/era5/test" -maxdepth 1 -type f -name '*.h5' | wc -l)"
[[ "${latent_count}" -eq 8 ]] || \
  die "expected 8 latent windows, found ${latent_count}"
[[ "${era5_count}" -eq 256 ]] || \
  die "expected 256 ERA5 source frames, found ${era5_count}"

for name in "${EXPECTED_OBS[@]}"; do
  [[ -s "${BUNDLE_ROOT}/obs/${name}" ]] || \
    die "observation file missing or empty: ${name}"
done

echo "Eight-window evaluation bundle is complete:"
echo "  root:        ${BUNDLE_ROOT}"
echo "  latents:     ${latent_count}"
echo "  ERA5 frames: ${era5_count}"
echo "  observations:${#EXPECTED_OBS[@]} files"
printf '  windows:     '
printf '%s ' "${EXPECTED_LATENTS[@]%.h5}"
echo
