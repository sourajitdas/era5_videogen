#!/usr/bin/env python3
"""Verify that the 8-window x 8-seed independent SR baseline is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/sr/baseline_independent")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(17, 25)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latent_files = sorted((args.data_root / "latents" / "test").glob("*.h5"))
    if len(latent_files) != 8:
        raise SystemExit(f"expected 8 active latent files, found {len(latent_files)}")
    invalid_windows = [index for index in args.windows if not 0 <= index < 8]
    if invalid_windows:
        raise SystemExit(f"window indices must be in 0..7, got {invalid_windows}")

    failures: list[str] = []
    completed = 0
    expected = len(args.windows) * len(args.seeds)

    for window_index in args.windows:
        latent_path = latent_files[window_index]
        source_index = latent_path.stem.removeprefix("2020_")
        expected_truth_shape = None
        for seed in args.seeds:
            run_dir = (
                args.root
                / f"w{window_index:02d}_src{source_index}_seed{seed}"
            )
            sample_dir = run_dir / f"sample_{window_index:04d}"
            required = [
                run_dir / "metadata.json",
                sample_dir / "samples.npy",
                sample_dir / "ground_truth.npy",
                sample_dir / "observation_lowres.npy",
                sample_dir / "metrics.json",
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                failures.append(f"window {window_index}, seed {seed}: missing {missing}")
                continue

            try:
                metadata = json.loads((run_dir / "metadata.json").read_text())
                if metadata.get("seed") != seed:
                    raise ValueError(f"metadata seed is {metadata.get('seed')}")
                if metadata.get("sample_index") != [window_index]:
                    raise ValueError(
                        f"metadata sample_index is {metadata.get('sample_index')}"
                    )

                samples = np.load(sample_dir / "samples.npy", mmap_mode="r")
                truth = np.load(sample_dir / "ground_truth.npy", mmap_mode="r")
                if samples.ndim != 5 or samples.shape[0] != 1:
                    raise ValueError(f"unexpected samples shape {samples.shape}")
                if tuple(samples.shape[1:]) != tuple(truth.shape):
                    raise ValueError(
                        f"sample/truth shape mismatch: {samples.shape} vs {truth.shape}"
                    )
                if expected_truth_shape is None:
                    expected_truth_shape = tuple(truth.shape)
                elif tuple(truth.shape) != expected_truth_shape:
                    raise ValueError(
                        f"truth shape changed across seeds: {truth.shape} vs "
                        f"{expected_truth_shape}"
                    )
            except Exception as exc:  # report every failed member in one pass
                failures.append(f"window {window_index}, seed {seed}: {exc}")
                continue

            completed += 1

    print(f"Validated {completed}/{expected} independent posterior members.")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(2)

    print("Step 1 baseline ensemble is complete.")


if __name__ == "__main__":
    main()