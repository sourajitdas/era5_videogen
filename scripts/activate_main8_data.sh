#!/usr/bin/env bash
# Activate era5_main8_all8 as data/ without copying the 4+ GB bundle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_ROOT="${1:-${REPO_ROOT}/era5_main8_all8}"
ACTIVE_DATA="${REPO_ROOT}/data"
BACKUP_DATA="${REPO_ROOT}/data_two_window_backup"

bash "${SCRIPT_DIR}/check_main8_assets.sh" "${BUNDLE_ROOT}"

if [[ -L "${ACTIVE_DATA}" ]]; then
  active_target="$(cd "$(dirname "${ACTIVE_DATA}")" && cd "$(readlink "${ACTIVE_DATA}")" && pwd)"
  bundle_target="$(cd "${BUNDLE_ROOT}" && pwd)"
  if [[ "${active_target}" == "${bundle_target}" ]]; then
    echo "Already active: ${ACTIVE_DATA} -> $(readlink "${ACTIVE_DATA}")"
    bash "${SCRIPT_DIR}/check_demo_assets.sh"
    exit 0
  fi
  echo "ERROR: ${ACTIVE_DATA} is already a symlink to ${active_target}" >&2
  exit 2
fi

if [[ -e "${ACTIVE_DATA}" ]]; then
  if [[ -e "${BACKUP_DATA}" ]]; then
    echo "ERROR: refusing to replace data/: backup already exists at ${BACKUP_DATA}" >&2
    exit 2
  fi
  mv "${ACTIVE_DATA}" "${BACKUP_DATA}"
  echo "Preserved the previous data directory at ${BACKUP_DATA}"
fi

# Resolve the target once so an explicitly supplied bundle path also works.
BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}" && pwd)"
ln -s "${BUNDLE_ROOT}" "${ACTIVE_DATA}"
echo "Activated: ${ACTIVE_DATA} -> $(readlink "${ACTIVE_DATA}")"

if ! bash "${SCRIPT_DIR}/check_demo_assets.sh"; then
  echo "Activation validation failed; restoring the previous data directory." >&2
  rm "${ACTIVE_DATA}"
  if [[ -e "${BACKUP_DATA}" ]]; then
    mv "${BACKUP_DATA}" "${ACTIVE_DATA}"
  fi
  exit 2
fi

echo "The eight-window bundle is ready for sampling."
