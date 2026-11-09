#!/usr/bin/env bash
# Assemble the minimal source-index-941 Laura forecast bundle on NERSC.
#
# The existing w932 bundle already contains the model, observations, dataset
# metadata, and ERA5 frames 0941--0963.  Only the 0941 latent and ERA5 frames
# 0964--0972 need to be transferred into INBOX first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_BUNDLE="${1:-${REPO_ROOT}/data/hurricane_laura/laura_w932_k8_bundle_v1}"
INBOX="${2:-${REPO_ROOT}/data/hurricane_laura/laura_w941_inbox}"
OUTPUT_BUNDLE="${3:-${REPO_ROOT}/data/hurricane_laura/laura_w941_forecast_bundle_v1}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -d "${BASE_BUNDLE}" ]] || die "base bundle not found: ${BASE_BUNDLE}"
[[ -f "${INBOX}/latent/2020_0941.h5" ]] || \
  die "missing forecast latent: ${INBOX}/latent/2020_0941.h5"
for source_index in $(seq 964 972); do
  [[ -f "${INBOX}/source/2020_$(printf '%04d' "${source_index}").h5" ]] || \
    die "missing ERA5 source frame ${source_index} in ${INBOX}/source"
done
[[ ! -e "${OUTPUT_BUNDLE}" ]] || \
  die "output already exists; refusing to modify it: ${OUTPUT_BUNDLE}"

mkdir -p \
  "${OUTPUT_BUNDLE}/data/latent_root/test" \
  "${OUTPUT_BUNDLE}/data/source_root/test"

# Reuse immutable assets from the verified w932 transfer.
ln -s "$(cd "${BASE_BUNDLE}/models" && pwd)" "${OUTPUT_BUNDLE}/models"
ln -s "$(cd "${BASE_BUNDLE}/data/observations" && pwd)" \
  "${OUTPUT_BUNDLE}/data/observations"

for item in "${BASE_BUNDLE}/data/latent_root"/*; do
  [[ "$(basename "${item}")" == "test" ]] && continue
  ln -s "$(cd "$(dirname "${item}")" && pwd)/$(basename "${item}")" \
    "${OUTPUT_BUNDLE}/data/latent_root/$(basename "${item}")"
done
for item in "${BASE_BUNDLE}/data/source_root"/*; do
  [[ "$(basename "${item}")" == "test" ]] && continue
  ln -s "$(cd "$(dirname "${item}")" && pwd)/$(basename "${item}")" \
    "${OUTPUT_BUNDLE}/data/source_root/$(basename "${item}")"
done

ln -s "$(cd "${INBOX}/latent" && pwd)/2020_0941.h5" \
  "${OUTPUT_BUNDLE}/data/latent_root/test/2020_0941.h5"

# Frames 0941--0963 overlap the existing hindcast bundle.
for source_index in $(seq 941 963); do
  name="2020_$(printf '%04d' "${source_index}").h5"
  source_path="${BASE_BUNDLE}/data/source_root/test/${name}"
  [[ -f "${source_path}" ]] || die "overlap frame missing from base bundle: ${source_path}"
  ln -s "$(cd "$(dirname "${source_path}")" && pwd)/${name}" \
    "${OUTPUT_BUNDLE}/data/source_root/test/${name}"
done

# Frames 0964--0972 are the only new raw ERA5 files.
for source_index in $(seq 964 972); do
  name="2020_$(printf '%04d' "${source_index}").h5"
  ln -s "$(cd "${INBOX}/source" && pwd)/${name}" \
    "${OUTPUT_BUNDLE}/data/source_root/test/${name}"
done

echo "Assembled Laura forecast bundle: ${OUTPUT_BUNDLE}"
echo "  latent: $(find "${OUTPUT_BUNDLE}/data/latent_root/test" -name '*.h5' | wc -l)"
echo "  ERA5 frames: $(find "${OUTPUT_BUNDLE}/data/source_root/test" -name '*.h5' | wc -l)"
echo "Run scripts/prepare_laura_forecast_repulsion.py to validate it."
