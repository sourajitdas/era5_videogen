#!/usr/bin/env python3
"""Build the paper's Figure 2-style super-resolution comparison.

The inputs are posterior output directories produced by ``generate_posterior``.
Each directory contributes every draw in its ``samples.npy`` file.  Only the
requested channels and frame are read, so eight full 69-channel trajectories
can be combined without loading all of them into memory at once.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIABLE_ALIASES = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}

VARIABLE_LABELS = {
    "2m_temperature": "t2m (K)",
    "10m_u_component_of_wind": "u10m (m/s)",
    "10m_v_component_of_wind": "v10m (m/s)",
    "mean_sea_level_pressure": "msl (Pa)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot low-resolution observation, ERA5 truth, and posterior ensemble mean."
    )
    parser.add_argument(
        "--run-glob",
        required=True,
        help=(
            "Quoted glob matching sample directories, for example "
            "'outputs/sr/E3c_lam0_seed*/sample_0000'."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Observed frame to show (default: first observed frame from metadata).",
    )
    parser.add_argument(
        "--variables",
        default="t2m,u10m,v10m,msl",
        help="Comma-separated aliases or ERA5 variable names.",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=8,
        help="Fail unless this many posterior draws are found (0 disables the check).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_metadata(sample_dir: Path) -> dict:
    path = sample_dir.parent / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_variables(spec: str, available: list[str]) -> tuple[list[str], list[int]]:
    names = []
    for token in (part.strip() for part in spec.split(",")):
        if token:
            names.append(VARIABLE_ALIASES.get(token, token))
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Variables absent from run metadata: {missing}")
    return names, [available.index(name) for name in names]


def selected_sample_fields(path: Path, channels: list[int], frame: int) -> np.ndarray:
    samples = np.load(path, mmap_mode="r")
    if samples.ndim == 4:
        samples = samples[None, ...]
    if samples.ndim != 5:
        raise ValueError(f"Expected (B,C,T,H,W) in {path}; found {samples.shape}")
    if not 0 <= frame < samples.shape[2]:
        raise IndexError(f"Frame {frame} is outside [0, {samples.shape[2] - 1}] in {path}")
    return np.asarray(samples[:, channels, frame], dtype=np.float64)


def color_limits(fields: list[np.ndarray], variable: str) -> tuple[float, float]:
    values = np.concatenate([np.asarray(field).ravel() for field in fields])
    if variable in {"10m_u_component_of_wind", "10m_v_component_of_wind"}:
        bound = float(np.nanpercentile(np.abs(values), 99.5))
        return -bound, bound
    return tuple(float(v) for v in np.nanpercentile(values, [0.5, 99.5]))


def main() -> None:
    args = parse_args()
    sample_dirs = [Path(path) for path in sorted(glob.glob(args.run_glob))]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories matched: {args.run_glob}")

    first_meta = load_metadata(sample_dirs[0])
    variables = list(first_meta["variables"])
    selected_names, channels = resolve_variables(args.variables, variables)
    obs_frames = [int(frame) for frame in first_meta.get("obs_frames", [])]
    frame = args.frame if args.frame is not None else (obs_frames[0] if obs_frames else 0)
    if obs_frames and frame not in obs_frames:
        raise ValueError(
            f"Frame {frame} is not observed. Figure 2 uses an observed frame; choose one of {obs_frames}."
        )
    downscale = int(first_meta.get("obs_downscale", 8))
    obs_mode = first_meta.get("obs_mode", "subsample")
    if obs_mode != "subsample":
        raise ValueError(f"This plotter currently expects obs_mode='subsample', found {obs_mode!r}.")

    truth_path = sample_dirs[0] / "ground_truth.npy"
    if not truth_path.is_file():
        raise FileNotFoundError(f"Missing ground truth: {truth_path}")
    truth_all = np.load(truth_path, mmap_mode="r")
    if truth_all.ndim != 4:
        raise ValueError(f"Expected (C,T,H,W) in {truth_path}; found {truth_all.shape}")
    truth = np.asarray(truth_all[channels, frame], dtype=np.float64)

    sample_sum = np.zeros_like(truth, dtype=np.float64)
    sample_count = 0
    seeds = []
    for sample_dir in sample_dirs:
        meta = load_metadata(sample_dir)
        if list(meta.get("variables", [])) != variables:
            raise ValueError(f"Variable order differs in {sample_dir.parent / 'metadata.json'}")
        if int(meta.get("obs_downscale", downscale)) != downscale:
            raise ValueError(f"Observation downscale differs in {sample_dir}")

        other_truth = np.load(sample_dir / "ground_truth.npy", mmap_mode="r")
        other_selected = np.asarray(other_truth[channels, frame])
        if other_selected.shape != truth.shape or not np.allclose(
            other_selected, truth, rtol=1e-6, atol=1e-5, equal_nan=True
        ):
            raise ValueError(
                f"Ground truth differs in {sample_dir}; do not mix posterior draws from different windows."
            )

        selected = selected_sample_fields(sample_dir / "samples.npy", channels, frame)
        sample_sum += selected.sum(axis=0)
        sample_count += selected.shape[0]
        seeds.append(meta.get("seed"))

    if args.expected_samples and sample_count != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} posterior draws, found {sample_count} "
            f"across {len(sample_dirs)} directories."
        )

    ensemble_mean = sample_sum / sample_count
    low_resolution = truth[:, ::downscale, ::downscale]

    fig, axes = plt.subplots(
        len(selected_names), 3, figsize=(11.2, 2.35 * len(selected_names)), squeeze=False
    )
    column_titles = ["low-res observation", "ERA5", "ensemble mean"]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11)

    for row, variable in enumerate(selected_names):
        panels = [low_resolution[row], truth[row], ensemble_mean[row]]
        vmin, vmax = color_limits(panels, variable)
        for column, field in enumerate(panels):
            image = axes[row, column].imshow(
                field,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest" if column == 0 else "bilinear",
                aspect="auto",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            fig.colorbar(image, ax=axes[row, column], fraction=0.034, pad=0.018)
        axes[row, 0].set_ylabel(VARIABLE_LABELS.get(variable, variable), fontsize=9)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.05, top=0.95, wspace=0.18, hspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "figure": "Figure 2-style super-resolution comparison",
        "run_glob": args.run_glob,
        "sample_directories": [str(path) for path in sample_dirs],
        "posterior_draws": sample_count,
        "seeds": seeds,
        "frame": frame,
        "observed_frames": obs_frames,
        "obs_downscale": downscale,
        "variables": selected_names,
    }
    provenance_path = args.output.with_suffix(".json")
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)

    print(f"Wrote {args.output}")
    print(f"Wrote {provenance_path}")
    print(f"Combined {sample_count} posterior draw(s) from seeds: {seeds}")
    print(f"Displayed observed frame {frame}; observation grid is {low_resolution.shape[-2:]}")


if __name__ == "__main__":
    main()
