#!/bin/bash
# Stage 2: encode the ERA5 corpus into autoencoder latents and write the
# deployable bundle (ae_model.pt + metadata.json) plus the per-channel
# normalization statistics (stats.yaml) the prior trains against.
#
# Usage: scripts/encode_latents.sh <checkpoint.pt> <output_dir>
set -euo pipefail
CKPT="${1:?usage: encode_latents.sh <checkpoint.pt> <output_dir>}"
OUT="${2:?usage: encode_latents.sh <checkpoint.pt> <output_dir>}"
cd "$(dirname "$0")/../src/ae"
export ERA5_DATA_ROOT="${ERA5_DATA_ROOT:?set ERA5_DATA_ROOT to your ERA5 directory}"
python save_latents.py --config config1.yaml --checkpoint "$CKPT" --output-root "$OUT"

# The prior standardizes each latent channel, so compute the statistics once.
cd "$(dirname "$0")/.."
python scripts/compute_latent_stats.py --root "$OUT"
