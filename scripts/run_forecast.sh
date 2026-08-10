#!/bin/bash
# Observations to forecast: assimilate two days of station observations
# (frames 0-7, IGRA + ISD + ICOADS super-obbed) and let the prior carry the
# trajectory forward over the remaining 24 frames (6 days).
#
# Pass --chain N to continue autoregressively for N windows, each initialised
# from the last frame of the previous one (the paper's three-week forecast).
#
# Usage: scripts/run_forecast.sh [window] [--chain N]   (default: window 92)
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src:${PYTHONPATH:-}"
CONFIG=checkpoints/prior/config.yaml
CKPT=checkpoints/prior/trigflow_adaptive_ema_019901.pt
AE=checkpoints/autoencoder/ae_model.pt
PKL=data/obs/joint_superob_2020_demo.pkl
OBSV=$(python3 -c "import json;print(json.load(open('data/obs/igra_2020_all_channels.json'))['obs_variables_flag'])")
WINDOW="${1:-92}"; shift || true
CHAIN=1
if [ "${1:-}" = "--chain" ]; then CHAIN="$2"; fi
SEED=17

python -m era5_videogen.generate_posterior_igra \
  --diffusion_config "$CONFIG" --diffusion_checkpoint "$CKPT" --ae_model_path "$AE" \
  --igra_pkl "$PKL" --obs_variables "$OBSV" \
  --output_dir outputs/forecast --name "fc_seed${SEED}" \
  --sample_index "$WINDOW" --num_posterior_samples 1 --seed "$SEED" \
  --obs_frames 0-7 --num_steps 50 --guidance_scale 0.4 --guide_norm dsg \
  --gamma 2e-6 --meas_std 5e-4 --corrector_steps 1 --corrector_snr 0.3 \
  --corrector_max_sigma 3 --noise_scale 0.0 --log_every 50

if [ "$CHAIN" -gt 1 ]; then
  python -m era5_videogen.generate_forecast_chain \
    --diffusion_config "$CONFIG" --diffusion_checkpoint "$CKPT" --ae_model_path "$AE" \
    --window1_dir "outputs/forecast/fc_seed${SEED}" --sample_index "$WINDOW" \
    --output_dir outputs/forecast_chain --name "chain_seed${SEED}" \
    --num_windows "$CHAIN" --ic_frames 1 --seed "$SEED" \
    --num_steps 50 --guidance_scale 0.4 --guide_norm dsg \
    --gamma 2e-6 --meas_std 5e-4 --corrector_steps 1 --corrector_snr 0.3 \
    --corrector_max_sigma 3 --noise_scale 0.0 --log_every 50
fi
