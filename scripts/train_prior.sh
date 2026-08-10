#!/bin/bash
# Stage 3: train the TrigFlow latent video prior (DiT3D, 525M parameters) on the
# encoded latents. Hydra config; override anything on the command line, e.g.
#
#   scripts/train_prior.sh data.dataset.root=/path/to/latents trainer.total_kimg=20000
#
# The paper's checkpoint is EMA at 19,901 kimg (8 A100s, ~5 days).
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m era5_videogen.train --config-name train \
  experiment=era5_ae_dit3d_trigflow_adaptive "$@"
