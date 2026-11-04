#!/usr/bin/env python3
"""Compare one K=8 repulsion smoke run with a matched independent K=8 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_baseline_ensemble import DEFAULT_PLOT_VARIABLES, resolve_variables
from evaluate_repulsion_smoke import baseline_members, evaluate


EXPECTED_MEMBERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repulsion-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, default=None)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path("outputs/sr/baseline_independent"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument(
        "--baseline-seeds", type=int, nargs="+", default=list(range(17, 25))
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_batched_run(run_dir: Path, window: int, label: str):
    metadata_path = run_dir / "metadata.json"
    sample_dir = run_dir / f"sample_{window:04d}"
    paths = {
        "metadata": metadata_path,
        "samples": sample_dir / "samples.npy",
        "truth": sample_dir / "ground_truth.npy",
        "observation": sample_dir / "observation_lowres.npy",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text())
    samples = np.load(paths["samples"], mmap_mode="r")
    if samples.ndim != 5 or samples.shape[0] != EXPECTED_MEMBERS:
        raise ValueError(
            f"expected {label} K={EXPECTED_MEMBERS} samples, found {samples.shape}"
        )
    return (
        metadata,
        samples,
        np.load(paths["truth"], mmap_mode="r"),
        np.load(paths["observation"], mmap_mode="r"),
    )


def plot_comparison(path: Path, control: dict, repulsion: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        "pairwise_mse_unobserved",
        "coverage_2sigma_unobserved",
        "observation_rmse",
        "skill_rmse_unobserved",
    ]
    labels = ["pairwise MSE", "2-sigma coverage", "obs RMSE", "skill RMSE"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        x - 0.2, [control[key] for key in metrics], width=0.4,
        label="independent K=8",
    )
    ax.bar(
        x + 0.2, [repulsion[key] for key in metrics], width=0.4,
        label="repulsion K=8",
    )
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def smoke_checks(control: dict, repulsion: dict) -> dict[str, bool]:
    scalar_values = [
        value for key, value in repulsion.items()
        if isinstance(value, (int, float)) and key != "members"
    ]
    rep_physical = repulsion.get("physical_consistency", {})
    ctl_physical = control.get("physical_consistency", {})
    physical_ok = True
    if rep_physical and ctl_physical:
        physical_ok = bool(
            rep_physical["geostrophic_cosine"]
            >= ctl_physical["geostrophic_cosine"] - 0.02
            and abs(
                rep_physical["geostrophic_speed_ratio"]
                - rep_physical["era5_geostrophic_speed_ratio"]
            )
            <= abs(
                ctl_physical["geostrophic_speed_ratio"]
                - ctl_physical["era5_geostrophic_speed_ratio"]
            ) + 0.05
            and rep_physical["hypsometric_rmse_k"]
            <= max(
                1.15 * ctl_physical["hypsometric_rmse_k"],
                ctl_physical["hypsometric_rmse_k"] + 0.1,
            )
        )

    spectral_ok = True
    for variable, rep_spectral in repulsion["spectral_consistency"].items():
        ctl_spectral = control["spectral_consistency"][variable]
        ratio = rep_spectral["member_high_wavenumber_power_ratio"]
        ctl_ratio = ctl_spectral["member_high_wavenumber_power_ratio"]
        spectral_ok = spectral_ok and bool(0.1 <= ratio <= max(4.0, 2.0 * ctl_ratio))

    return {
        "finite_metrics": bool(all(np.isfinite(value) for value in scalar_values)),
        "diversity_increased": (
            repulsion["pairwise_mse_unobserved"]
            > control["pairwise_mse_unobserved"]
        ),
        "observation_consistency_preserved": (
            repulsion["observation_rmse"]
            <= max(
                1.10 * control["observation_rmse"],
                control["observation_rmse"] + 0.01,
            )
        ),
        "no_catastrophic_skill_loss": (
            repulsion["skill_rmse_unobserved"]
            <= 1.25 * control["skill_rmse_unobserved"]
        ),
        "physical_consistency_preserved": physical_ok,
        "no_catastrophic_spectral_distortion": spectral_ok,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, rep_samples, truth, observation = load_batched_run(
        args.repulsion_run, args.window, "repulsion"
    )
    if float(metadata.get("repulsion_strength", 0.0)) <= 0:
        raise ValueError("repulsion run metadata has no active repulsion strength")

    if args.control_run is not None:
        control_metadata, control_samples, control_truth, control_obs = load_batched_run(
            args.control_run, args.window, "control"
        )
        if float(control_metadata.get("repulsion_strength", 0.0)) != 0.0:
            raise ValueError("control run has nonzero repulsion strength")
        if control_truth.shape != truth.shape or control_obs.shape != observation.shape:
            raise ValueError("control and repulsion truth/observation shapes differ")
        source_files = sorted((args.data_root / "latents" / "test").glob("*.h5"))
        source = source_files[args.window].stem.removeprefix("2020_")
        control_loader = lambda channel: np.asarray(
            control_samples[:, channel], dtype=np.float32
        )
    else:
        paths, source = baseline_members(
            args.baseline_root, args.data_root, args.window, args.baseline_seeds
        )
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        control_loader = lambda channel: np.stack(
            [np.asarray(array[0, channel], dtype=np.float32) for array in arrays],
            axis=0,
        )

    variables = list(metadata["variables"])
    plot_variables = resolve_variables(DEFAULT_PLOT_VARIABLES, variables)
    lat = np.asarray(np.load(args.data_root / "era5" / "lat.npy")).squeeze()
    lon = np.asarray(np.load(args.data_root / "era5" / "lon.npy")).squeeze()
    lat_weights = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    obs_lat_weights = lat_weights[::8]
    means = np.load(args.data_root / "era5" / "normalize_mean.npz")
    stds = np.load(args.data_root / "era5" / "normalize_std.npz")
    repulsion_loader = lambda channel: np.asarray(
        rep_samples[:, channel], dtype=np.float32
    )

    rep_summary, _ = evaluate(
        repulsion_loader, truth, observation, variables, means, stds,
        lat, lon, lat_weights, obs_lat_weights, plot_variables,
    )
    control_summary, _ = evaluate(
        control_loader, truth, observation, variables, means, stds,
        lat, lon, lat_weights, obs_lat_weights, plot_variables,
    )
    checks = smoke_checks(control_summary, rep_summary)
    scalar_keys = [
        key for key, value in rep_summary.items()
        if isinstance(value, (int, float)) and key != "members"
    ]
    result = {
        "configuration": {
            "window": args.window,
            "source_index": source,
            "members": EXPECTED_MEMBERS,
            "control_run": None if args.control_run is None else str(args.control_run),
            "repulsion_run": str(args.repulsion_run),
            "repulsion_strength": metadata["repulsion_strength"],
            "repulsion_sigma_band": [
                metadata["repulsion_min_sigma"], metadata["repulsion_max_sigma"]
            ],
            "repulsion_schedule": metadata["repulsion_schedule"],
            "repulsion_unobserved_only": metadata["repulsion_unobserved_only"],
        },
        "control_k8": control_summary,
        "repulsion_k8": rep_summary,
        "relative_change": {
            key: rep_summary[key] / max(control_summary[key], 1e-30) - 1.0
            for key in scalar_keys
        },
        "smoke_checks": checks,
        "smoke_passed": bool(all(checks.values())),
        "note": (
            "This one-window K=8 run is a smoke/tuning comparison, not the "
            "final eight-window Stage 3 evaluation."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(
        args.output_dir / "smoke_comparison_k8.png", control_summary, rep_summary
    )
    print(json.dumps(result, indent=2))
    print(f"Wrote K=8 repulsion smoke comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
