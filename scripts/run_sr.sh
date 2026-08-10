#!/bin/bash
# Super-resolution: reconstruct the full 69-channel state from 8x-coarsened
# observations of every 4th frame (the paper's SR protocol).
#
#   E3c_lam0  = 30 solver steps + 1 low-sigma corrector step, lambda=0  (47 NFE)
#   dps_corr1 = 50 solver steps + corrector at every level              (99 NFE)
#
# Usage: scripts/run_sr.sh [window ...]     (default: window 93)
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
CONFIG=checkpoints/prior/config.yaml
CKPT=checkpoints/prior/trigflow_adaptive_ema_019901.pt
AE=checkpoints/autoencoder/ae_model.pt
WINDOWS="${@:-93}"

python -m era5_videogen.generate_posterior \
  --diffusion_config "$CONFIG" --diffusion_checkpoint "$CKPT" --ae_model_path "$AE" \
  --output_dir outputs/sr --name E3c_lam0 \
  --sample_index $WINDOWS --num_posterior_samples 1 --seed 17 \
  --obs_frames every:4 --obs_downscale 8 --obs_mode subsample \
  --guidance_scale 4.0 --gamma 1e-1 --meas_std 1e-2 \
  --num_steps 30 --corrector_steps 1 --corrector_snr 0.3 --corrector_max_sigma 5 \
  --noise_scale 0 --log_every 100
