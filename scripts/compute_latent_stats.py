#!/usr/bin/env python
"""Compute per-channel mean/std of the ERA5 AE latents and write ``stats.yaml``.

The deterministic autoencoder latents (``input/latent``, shape ``(C, T, H, W)``)
are not normalized at save time. This script streams the *training* split,
accumulates per-channel sums / sums-of-squares in float64, and writes the
resulting per-channel statistics next to the latents as ``<root>/stats.yaml``.

``era5_videogen.data.ae_latents.ERA5AELatentDataset`` reads this file to
standardize each latent channel to zero mean / unit variance during training,
and ``generate.py`` uses it to denormalize sampled latents before saving/decoding.

Run it once after encoding the latents:

    python scripts/compute_latent_stats.py --root /path/to/latents

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from era5_videogen.data.ae_latents import ERA5AELatentDataset

STATS_FORMAT_VERSION = "era5_ae_latent_stats_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="data/latents",
        help="Root directory containing train/val/test latent HDF5 folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Split to compute statistics over (training data by default).",
    )
    parser.add_argument("--latent-key", type=str, default="input/latent")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for the per-batch reduction (accumulation is float64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of files (for a quick sanity run).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path. Defaults to <root>/stats.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing stats file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output) if args.output else Path(args.root) / "stats.yaml"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to regenerate it."
        )

    # normalize=False -> raw latents (we are computing the normalization itself).
    dataset = ERA5AELatentDataset(
        root=args.root,
        split=args.split,
        normalize=False,
        latent_key=args.latent_key,
    )
    files = dataset.files
    if args.limit is not None:
        dataset.files = files[: args.limit]
        files = dataset.files

    n_channels = dataset.n_channels
    c_idx, t, h, w = dataset.sample_shape
    print(f"root={args.root} split={args.split}")
    print(f"num_files={len(files)} latent_shape={dataset.sample_shape}")
    print(f"device={args.device}")

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    device = torch.device(args.device)
    channel_sum = torch.zeros(n_channels, dtype=torch.float64, device=device)
    channel_sumsq = torch.zeros(n_channels, dtype=torch.float64, device=device)
    channel_min = torch.full((n_channels,), float("inf"), dtype=torch.float64, device=device)
    channel_max = torch.full((n_channels,), float("-inf"), dtype=torch.float64, device=device)
    total_count = 0  # number of (sample * T * H * W) elements per channel

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.split} latents", unit="batch"):
            # batch: (B, C, T, H, W)
            batch = batch.to(device=device, dtype=torch.float64, non_blocking=True)
            b = batch.shape[0]
            reduce_dims = (0, 2, 3, 4)
            channel_sum += batch.sum(dim=reduce_dims)
            channel_sumsq += batch.pow(2).sum(dim=reduce_dims)
            channel_min = torch.minimum(channel_min, batch.amin(dim=reduce_dims))
            channel_max = torch.maximum(channel_max, batch.amax(dim=reduce_dims))
            total_count += b * t * h * w

    if total_count == 0:
        raise RuntimeError("No latents were read; cannot compute statistics.")

    mean = channel_sum / total_count
    var = channel_sumsq / total_count - mean.pow(2)
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)

    # Guard against degenerate (constant) channels that would divide by ~0.
    eps = 1e-8
    n_degenerate = int((std < eps).sum().item())
    if n_degenerate:
        print(f"[warn] {n_degenerate} channel(s) have std < {eps}; flooring to {eps}.")
    std = torch.clamp(std, min=eps)

    mean_list = mean.detach().cpu().numpy().astype(np.float64).tolist()
    std_list = std.detach().cpu().numpy().astype(np.float64).tolist()
    min_list = channel_min.detach().cpu().numpy().astype(np.float64).tolist()
    max_list = channel_max.detach().cpu().numpy().astype(np.float64).tolist()

    stats = {
        "format": STATS_FORMAT_VERSION,
        "root": str(args.root),
        "split": args.split,
        "latent_key": args.latent_key,
        "latent_shape": [int(v) for v in dataset.sample_shape],
        "channels": int(n_channels),
        "num_files": int(len(files)),
        "count_per_channel": int(total_count),
        "reduce_over": "samples, time, height, width (per-channel)",
        "global_mean": float(mean.mean().item()),
        "global_std": float(std.mean().item()),
        "mean": mean_list,
        "std": std_list,
        "min": min_list,
        "max": max_list,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp_path, output_path)

    print(f"\nWrote {output_path}")
    print(f"  channels={n_channels} count_per_channel={total_count:,}")
    print(f"  mean[:5]={mean_list[:5]}")
    print(f"  std[:5] ={std_list[:5]}")
    print(f"  global mean={stats['global_mean']:.6f} global std={stats['global_std']:.6f}")


if __name__ == "__main__":
    main()
