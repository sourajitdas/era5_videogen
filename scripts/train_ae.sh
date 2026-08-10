#!/bin/bash
# Stage 1: train the 3D autoencoder that maps 69-channel ERA5 windows
# (32 frames, 128x256) to (128, 8, 32, 64) latents.
#
# Point ERA5_DATA_ROOT at your ERA5 dataset (see README, "Data layout").
# Single process; for multi-GPU use torchrun / your scheduler's DDP launcher.
set -euo pipefail
cd "$(dirname "$0")/../src/ae"
export ERA5_DATA_ROOT="${ERA5_DATA_ROOT:?set ERA5_DATA_ROOT to your ERA5 directory}"
python train.py --config config1.yaml
