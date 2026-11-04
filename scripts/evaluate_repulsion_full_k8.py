#!/usr/bin/env python3
"""Evaluate one window of the final matched independent-vs-repulsion K=8 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_baseline_ensemble import DEFAULT_PLOT_VARIABLES, resolve_variables
from evaluate_repulsion_smoke import evaluate
from evaluate_repulsion_smoke_k8 import (
    EXPECTED_MEMBERS,
    load_batched_run,
    plot_comparison,
    smoke_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repulsion-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative(new: float, old: float) -> float:
    return new / max(old, 1e-30) - 1.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rep_meta, rep_samples, truth, observation = load_batched_run(
        args.repulsion_run, args.window, "repulsion"
    )
    ctl_meta, ctl_samples, ctl_truth, ctl_observation = load_batched_run(
        args.control_run, args.window, "control"
    )
    if float(rep_meta.get("repulsion_strength", 0.0)) <= 0:
        raise ValueError("repulsion run metadata has no active repulsion strength")
    if float(ctl_meta.get("repulsion_strength", 0.0)) != 0.0:
        raise ValueError("control run metadata has nonzero repulsion strength")
    # The standard 80-GB NERSC path runs all K particles in one batch and
    # therefore predates the explicit particle_coupling metadata added by the
    # ICDS microbatch wrapper.  Infer the same global coupling only for that
    # validated full-batch K=8 configuration.
    particle_coupling = rep_meta.get("particle_coupling")
    if (
        particle_coupling is None
        and int(rep_meta.get("num_posterior_samples", 0)) == 8
    ):
        particle_coupling = "global_all_particles"
    if particle_coupling != "global_all_particles":
        raise ValueError("repulsion run does not declare global K-particle coupling")
    if ctl_truth.shape != truth.shape or ctl_observation.shape != observation.shape:
        raise ValueError("control and repulsion truth/observation shapes differ")

    latent_files = sorted((args.data_root / "latents" / "test").glob("*.h5"))
    if not 0 <= args.window < len(latent_files):
        raise ValueError(f"window {args.window} is outside the active data bundle")
    source = latent_files[args.window].stem.removeprefix("2020_")

    variables = list(rep_meta["variables"])
    if list(ctl_meta["variables"]) != variables:
        raise ValueError("control and repulsion variable lists differ")
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
    control_loader = lambda channel: np.asarray(
        ctl_samples[:, channel], dtype=np.float32
    )
    repulsion, _ = evaluate(
        repulsion_loader,
        truth,
        observation,
        variables,
        means,
        stds,
        lat,
        lon,
        lat_weights,
        obs_lat_weights,
        plot_variables,
    )
    control, _ = evaluate(
        control_loader,
        truth,
        observation,
        variables,
        means,
        stds,
        lat,
        lon,
        lat_weights,
        obs_lat_weights,
        plot_variables,
    )

    checks = smoke_checks(control, repulsion)
    checks.update(
        {
            "skill_rmse_preserved_2pct": relative(
                repulsion["skill_rmse_unobserved"],
                control["skill_rmse_unobserved"],
            )
            <= 0.02,
            "observation_rmse_preserved_5pct": relative(
                repulsion["observation_rmse"], control["observation_rmse"]
            )
            <= 0.05,
            "temporal_rmse_preserved_2pct": relative(
                repulsion["temporal_increment_rmse"],
                control["temporal_increment_rmse"],
            )
            <= 0.02,
            "not_overdispersed": repulsion[
                "spread_skill_ratio_unobserved_finite_ensemble_corrected"
            ]
            <= 1.20,
        }
    )
    scalar_keys = [
        key
        for key, value in repulsion.items()
        if isinstance(value, (int, float)) and key != "members"
    ]
    result = {
        "configuration": {
            "window": args.window,
            "source_index": source,
            "members": EXPECTED_MEMBERS,
            "control_run": str(args.control_run),
            "repulsion_run": str(args.repulsion_run),
            "repulsion_strength": rep_meta["repulsion_strength"],
            "repulsion_sigma_band": [
                rep_meta["repulsion_min_sigma"],
                rep_meta["repulsion_max_sigma"],
            ],
            "repulsion_schedule": rep_meta["repulsion_schedule"],
            "repulsion_unobserved_only": rep_meta["repulsion_unobserved_only"],
            "particle_microbatch": rep_meta.get("particle_microbatch"),
            "particle_coupling": particle_coupling,
        },
        "control_k8": control,
        "repulsion_k8": repulsion,
        "relative_change": {
            key: relative(repulsion[key], control[key]) for key in scalar_keys
        },
        "validity_checks": checks,
        "validity_passed": bool(all(checks.values())),
        "note": "One window of the final eight-window matched K=8 evaluation.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_comparison(
        args.output_dir / "matched_comparison_k8.png", control, repulsion
    )
    print(json.dumps(result, indent=2))
    print(f"Wrote final K=8 window evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
