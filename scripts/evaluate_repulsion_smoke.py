#!/usr/bin/env python3
"""Compare one K=4 repulsion smoke run with four independent baseline draws."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_baseline_ensemble import (
    DEFAULT_PLOT_VARIABLES,
    OBS_FRAMES,
    area_mean,
    geostrophic_metrics,
    hypsometric_rmse,
    mean_frame_area_rmse,
    resolve_variables,
    scalar_stat,
    zonal_power,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repulsion-run", type=Path, required=True)
    parser.add_argument(
        "--control-run", type=Path, default=None,
        help="Optional batched K=4 no-repulsion run made with identical sampler settings.",
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path("outputs/sr/baseline_independent"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--baseline-seeds", type=int, nargs="+", default=[17, 18, 19, 20])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def baseline_members(
    root: Path, data_root: Path, window: int, seeds: list[int]
) -> tuple[list[Path], str]:
    latent_files = sorted((data_root / "latents" / "test").glob("*.h5"))
    if not 0 <= window < len(latent_files):
        raise ValueError(f"window {window} is outside 0..{len(latent_files) - 1}")
    source = latent_files[window].stem.removeprefix("2020_")
    paths = []
    for seed in seeds:
        path = (
            root / f"w{window:02d}_src{source}_seed{seed}"
            / f"sample_{window:04d}" / "samples.npy"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths, source


def evaluate(
    member_loader,
    truth: np.ndarray,
    observation: np.ndarray,
    variables: list[str],
    means_npz,
    stds_npz,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_weights: np.ndarray,
    obs_lat_weights: np.ndarray,
    plot_variables: list[str],
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    rows = []
    spectra: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"truth": [], "members": [], "ensemble_mean": []}
        for name in plot_variables
    }
    observed = list(OBS_FRAMES)

    for channel, variable in enumerate(variables):
        samples = member_loader(channel)
        target = np.asarray(truth[channel], dtype=np.float32)
        if samples.ndim != 4 or samples.shape[1:] != target.shape:
            raise ValueError(
                f"unexpected {variable} sample/truth shapes: {samples.shape}, {target.shape}"
            )
        mean = scalar_stat(means_npz, variable)
        std = scalar_stat(stds_npz, variable)
        samples_std = (samples - mean) / std
        target_std = (target - mean) / std
        ensemble_mean = np.mean(samples_std, axis=0)
        sample_stddev = np.std(samples_std, axis=0, ddof=1)
        unobserved = np.ones(target.shape[0], dtype=bool)
        unobserved[observed] = False

        spread_mse = float(area_mean(np.square(sample_stddev), lat_weights))
        skill_mse = float(area_mean(np.square(ensemble_mean - target_std), lat_weights))
        spread_mse_unobs = float(
            area_mean(np.square(sample_stddev[unobserved]), lat_weights)
        )
        skill_mse_unobs = float(
            area_mean(
                np.square(ensemble_mean[unobserved] - target_std[unobserved]),
                lat_weights,
            )
        )
        predicted_obs = samples_std[:, observed, ::8, ::8]
        measured = np.asarray(observation[0, channel], dtype=np.float32)
        if predicted_obs.shape[1:] != measured.shape:
            raise ValueError(
                f"observation shape mismatch for {variable}: "
                f"{predicted_obs.shape[1:]} != {measured.shape}"
            )
        increments = np.diff(samples_std, axis=1)
        truth_increments = np.diff(target_std, axis=0)

        rows.append(
            {
                "skill_mse": skill_mse,
                "spread_mse": spread_mse,
                "pairwise_mse": 2.0 * spread_mse,
                "coverage_range": float(area_mean(
                    (target_std >= np.min(samples_std, axis=0))
                    & (target_std <= np.max(samples_std, axis=0)),
                    lat_weights,
                )),
                "coverage_2sigma": float(area_mean(
                    np.abs(target_std - ensemble_mean) <= 2.0 * sample_stddev,
                    lat_weights,
                )),
                "skill_mse_unobserved": skill_mse_unobs,
                "spread_mse_unobserved": spread_mse_unobs,
                "pairwise_mse_unobserved": 2.0 * spread_mse_unobs,
                "coverage_range_unobserved": float(area_mean(
                    (target_std[unobserved] >= np.min(samples_std[:, unobserved], axis=0))
                    & (target_std[unobserved] <= np.max(samples_std[:, unobserved], axis=0)),
                    lat_weights,
                )),
                "coverage_2sigma_unobserved": float(area_mean(
                    np.abs(target_std[unobserved] - ensemble_mean[unobserved])
                    <= 2.0 * sample_stddev[unobserved],
                    lat_weights,
                )),
                "observation_rmse": float(mean_frame_area_rmse(
                    np.mean(predicted_obs, axis=0) - measured, obs_lat_weights
                )),
                "temporal_increment_rmse": float(np.sqrt(area_mean(
                    np.square(np.mean(increments, axis=0) - truth_increments),
                    lat_weights,
                ))),
                "temporal_increment_energy_ratio": float(
                    np.mean(area_mean(np.square(increments), lat_weights))
                    / max(float(area_mean(np.square(truth_increments), lat_weights)), 1e-30)
                ),
            }
        )

        if variable in spectra:
            spectra[variable]["truth"].append(zonal_power(target_std, lat_weights))
            spectra[variable]["members"].append(zonal_power(samples_std, lat_weights))
            spectra[variable]["ensemble_mean"].append(
                zonal_power(ensemble_mean, lat_weights)
            )

    spread_mse = float(np.mean([row["spread_mse"] for row in rows]))
    skill_mse = float(np.mean([row["skill_mse"] for row in rows]))
    spread_mse_u = float(np.mean([row["spread_mse_unobserved"] for row in rows]))
    skill_mse_u = float(np.mean([row["skill_mse_unobserved"] for row in rows]))
    k = int(member_loader(0).shape[0])
    summary = {
        "members": k,
        "skill_rmse": math.sqrt(skill_mse),
        "spread": math.sqrt(spread_mse),
        "spread_skill_ratio_finite_ensemble_corrected": (
            math.sqrt((k + 1) / k) * math.sqrt(spread_mse / max(skill_mse, 1e-30))
        ),
        "pairwise_mse": float(np.mean([row["pairwise_mse"] for row in rows])),
        "coverage_range": float(np.mean([row["coverage_range"] for row in rows])),
        "coverage_2sigma": float(np.mean([row["coverage_2sigma"] for row in rows])),
        "skill_rmse_unobserved": math.sqrt(skill_mse_u),
        "spread_unobserved": math.sqrt(spread_mse_u),
        "spread_skill_ratio_unobserved_finite_ensemble_corrected": (
            math.sqrt((k + 1) / k)
            * math.sqrt(spread_mse_u / max(skill_mse_u, 1e-30))
        ),
        "pairwise_mse_unobserved": float(
            np.mean([row["pairwise_mse_unobserved"] for row in rows])
        ),
        "coverage_range_unobserved": float(
            np.mean([row["coverage_range_unobserved"] for row in rows])
        ),
        "coverage_2sigma_unobserved": float(
            np.mean([row["coverage_2sigma_unobserved"] for row in rows])
        ),
        "observation_rmse": float(np.mean([row["observation_rmse"] for row in rows])),
        "temporal_increment_rmse": float(
            np.mean([row["temporal_increment_rmse"] for row in rows])
        ),
        "temporal_increment_energy_ratio": float(
            np.mean([row["temporal_increment_energy_ratio"] for row in rows])
        ),
    }

    spectral_summary = {}
    for variable, values in spectra.items():
        truth_power = values["truth"][0][1:]
        member_power = values["members"][0][1:]
        floor = max(float(np.max(truth_power)) * 1e-12, 1e-30)
        split = len(truth_power) // 2
        spectral_summary[variable] = {
            "member_log_power_rmse": float(np.sqrt(np.mean(np.square(
                np.log10(np.maximum(member_power, floor))
                - np.log10(np.maximum(truth_power, floor))
            )))),
            "member_high_wavenumber_power_ratio": float(
                np.sum(member_power[split:])
                / max(float(np.sum(truth_power[split:])), 1e-30)
            ),
        }
    summary["spectral_consistency"] = spectral_summary

    channels = {name: index for index, name in enumerate(variables)}
    required = [
        "u_component_of_wind_500", "v_component_of_wind_500", "geopotential_500",
        "geopotential_850", "temperature_500", "temperature_600",
        "temperature_700", "temperature_850",
    ]
    if all(name in channels for name in required):
        fields = {name: member_loader(channels[name]) for name in required}
        truth_fields = {
            name: np.asarray(truth[channels[name]], dtype=np.float32)[None]
            for name in required
        }
        geo_cos, geo_speed = geostrophic_metrics(
            fields["u_component_of_wind_500"],
            fields["v_component_of_wind_500"],
            fields["geopotential_500"], lat, lon,
        )
        hyp = hypsometric_rmse(
            fields["geopotential_500"], fields["geopotential_850"],
            [fields["temperature_500"], fields["temperature_600"],
             fields["temperature_700"], fields["temperature_850"]],
            lat_weights,
        )
        truth_geo_cos, truth_geo_speed = geostrophic_metrics(
            truth_fields["u_component_of_wind_500"],
            truth_fields["v_component_of_wind_500"],
            truth_fields["geopotential_500"], lat, lon,
        )
        truth_hyp = hypsometric_rmse(
            truth_fields["geopotential_500"], truth_fields["geopotential_850"],
            [truth_fields["temperature_500"], truth_fields["temperature_600"],
             truth_fields["temperature_700"], truth_fields["temperature_850"]],
            lat_weights,
        )
        summary["physical_consistency"] = {
            "geostrophic_cosine": float(np.mean(geo_cos)),
            "geostrophic_speed_ratio": float(np.mean(geo_speed)),
            "hypsometric_rmse_k": float(np.mean(hyp)),
            "era5_geostrophic_cosine": float(truth_geo_cos[0]),
            "era5_geostrophic_speed_ratio": float(truth_geo_speed[0]),
            "era5_hypsometric_rmse_k": float(truth_hyp[0]),
        }
    return summary, spectra


def plot_comparison(path: Path, baseline: dict, repulsion: dict) -> None:
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
    ax.bar(x - 0.2, [baseline[m] for m in metrics], width=0.4, label="independent K=4")
    ax.bar(x + 0.2, [repulsion[m] for m in metrics], width=0.4, label="repulsion K=4")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rep_metadata_path = args.repulsion_run / "metadata.json"
    sample_dir = args.repulsion_run / f"sample_{args.window:04d}"
    rep_path = sample_dir / "samples.npy"
    truth_path = sample_dir / "ground_truth.npy"
    obs_path = sample_dir / "observation_lowres.npy"
    for path in (rep_metadata_path, rep_path, truth_path, obs_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(rep_metadata_path.read_text())
    rep_samples = np.load(rep_path, mmap_mode="r")
    truth = np.load(truth_path, mmap_mode="r")
    observation = np.load(obs_path, mmap_mode="r")
    if rep_samples.ndim != 5 or rep_samples.shape[0] != 4:
        raise ValueError(f"expected K=4 repulsion samples, found {rep_samples.shape}")
    if float(metadata.get("repulsion_strength", 0.0)) <= 0:
        raise ValueError("run metadata does not describe an active repulsion sampler")

    if args.control_run is not None:
        control_path = args.control_run / f"sample_{args.window:04d}" / "samples.npy"
        if not control_path.is_file():
            raise FileNotFoundError(control_path)
        control_samples = np.load(control_path, mmap_mode="r")
        if control_samples.ndim != 5 or control_samples.shape[0] != 4:
            raise ValueError(f"expected K=4 control samples, found {control_samples.shape}")
        latent_files = sorted((args.data_root / "latents" / "test").glob("*.h5"))
        source = latent_files[args.window].stem.removeprefix("2020_")
        base_arrays = None
    else:
        base_paths, source = baseline_members(
            args.baseline_root, args.data_root, args.window, args.baseline_seeds
        )
        base_arrays = [np.load(path, mmap_mode="r") for path in base_paths]
        control_samples = None
    variables = list(metadata["variables"])
    plot_variables = resolve_variables(DEFAULT_PLOT_VARIABLES, variables)
    lat = np.asarray(np.load(args.data_root / "era5" / "lat.npy")).squeeze()
    lon = np.asarray(np.load(args.data_root / "era5" / "lon.npy")).squeeze()
    lat_weights = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    obs_lat_weights = lat_weights[::8]
    means = np.load(args.data_root / "era5" / "normalize_mean.npz")
    stds = np.load(args.data_root / "era5" / "normalize_std.npz")

    rep_loader = lambda channel: np.asarray(rep_samples[:, channel], dtype=np.float32)
    if control_samples is not None:
        base_loader = lambda channel: np.asarray(
            control_samples[:, channel], dtype=np.float32
        )
    else:
        base_loader = lambda channel: np.stack(
            [np.asarray(array[0, channel], dtype=np.float32) for array in base_arrays], axis=0
        )
    rep_summary, _ = evaluate(
        rep_loader, truth, observation, variables, means, stds,
        lat, lon, lat_weights, obs_lat_weights, plot_variables,
    )
    base_summary, _ = evaluate(
        base_loader, truth, observation, variables, means, stds,
        lat, lon, lat_weights, obs_lat_weights, plot_variables,
    )

    scalar_keys = [
        key for key, value in rep_summary.items()
        if isinstance(value, (int, float)) and key != "members"
    ]
    rep_physical = rep_summary.get("physical_consistency", {})
    base_physical = base_summary.get("physical_consistency", {})
    physical_ok = True
    if rep_physical and base_physical:
        physical_ok = bool(
            rep_physical["geostrophic_cosine"]
            >= base_physical["geostrophic_cosine"] - 0.02
            and abs(
                rep_physical["geostrophic_speed_ratio"]
                - rep_physical["era5_geostrophic_speed_ratio"]
            )
            <= abs(
                base_physical["geostrophic_speed_ratio"]
                - base_physical["era5_geostrophic_speed_ratio"]
            ) + 0.05
            and rep_physical["hypsometric_rmse_k"]
            <= max(1.15 * base_physical["hypsometric_rmse_k"],
                   base_physical["hypsometric_rmse_k"] + 0.1)
        )
    spectral_ok = True
    for variable, rep_spectral in rep_summary["spectral_consistency"].items():
        base_spectral = base_summary["spectral_consistency"][variable]
        ratio = rep_spectral["member_high_wavenumber_power_ratio"]
        baseline_ratio = base_spectral["member_high_wavenumber_power_ratio"]
        spectral_ok = spectral_ok and bool(
            0.1 <= ratio <= max(4.0, 2.0 * baseline_ratio)
        )

    checks = {
        "finite_metrics": bool(all(np.isfinite(rep_summary[key]) for key in scalar_keys)),
        "diversity_increased": (
            rep_summary["pairwise_mse_unobserved"]
            > base_summary["pairwise_mse_unobserved"]
        ),
        "observation_consistency_preserved": (
            rep_summary["observation_rmse"]
            <= max(base_summary["observation_rmse"] * 1.10, base_summary["observation_rmse"] + 0.01)
        ),
        "no_catastrophic_skill_loss": (
            rep_summary["skill_rmse_unobserved"]
            <= 1.25 * base_summary["skill_rmse_unobserved"]
        ),
        "physical_consistency_preserved": physical_ok,
        "no_catastrophic_spectral_distortion": spectral_ok,
    }
    result = {
        "configuration": {
            "window": args.window,
            "source_index": source,
            "baseline_seeds": args.baseline_seeds,
            "control_run": None if args.control_run is None else str(args.control_run),
            "repulsion_run": str(args.repulsion_run),
            "repulsion_strength": metadata["repulsion_strength"],
            "repulsion_sigma_band": [
                metadata["repulsion_min_sigma"], metadata["repulsion_max_sigma"]
            ],
            "repulsion_schedule": metadata["repulsion_schedule"],
            "repulsion_unobserved_only": metadata["repulsion_unobserved_only"],
        },
        "baseline_k4": base_summary,
        "repulsion_k4": rep_summary,
        "relative_change": {
            key: rep_summary[key] / max(base_summary[key], 1e-30) - 1.0
            for key in scalar_keys
        },
        "smoke_checks": checks,
        "smoke_passed": bool(all(checks.values())),
        "note": (
            "This one-window K=4 run is a functional and tuning smoke test, not a "
            "statistically conclusive Stage 3 evaluation."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(args.output_dir / "smoke_comparison.png", base_summary, rep_summary)
    print(json.dumps(result, indent=2))
    print(f"Wrote repulsion smoke comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
