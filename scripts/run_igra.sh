#!/bin/bash
# Real-observation data assimilation: reconstruct the full state from sparse
# IGRA radiosonde soundings (69 observed channels) on the middle 8 frames.
# Sampler dsg_lam0: 50 steps + low-sigma corrector, DSG-normalized guidance,
# lambda=0 (76 NFE) -- the paper's station configuration.
#
# Usage: scripts/run_igra.sh [window ...]
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
CONFIG=checkpoints/prior/config.yaml
CKPT=checkpoints/prior/trigflow_adaptive_ema_019901.pt
AE=checkpoints/autoencoder/ae_model.pt
PKL=data/obs/igra_2020_demo.pkl
OBSV=$(python3 -c "import json;print(json.load(open('data/obs/igra_2020_all_channels.json'))['obs_variables_flag'])")
WINDOWS="${@:-93}"

python -m era5_videogen.generate_posterior_igra \
  --diffusion_config "$CONFIG" --diffusion_checkpoint "$CKPT" --ae_model_path "$AE" \
  --igra_pkl "$PKL" --obs_variables "$OBSV" \
  --output_dir outputs/igra --name dsg_lam0 \
  --sample_index $WINDOWS --num_posterior_samples 1 --seed 17 \
  --obs_frames middle:8 --gamma 2e-6 --meas_std 5e-4 \
  --num_steps 50 --guidance_scale 0.15 --guide_norm dsg \
  --corrector_steps 1 --corrector_snr 0.3 --corrector_max_sigma 3 \
  --noise_scale 0.0 --log_every 100
