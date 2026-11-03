#!/usr/bin/env python3
"""Evaluate the 8-window x 8-member independent SR posterior baseline.

The evaluator is deliberately CPU-only and memory bounded.  It opens the NPY
files as memory maps and processes one weather variable at a time.  Metrics in
standardized units give every ERA5 channel equal weight; physical-unit RMSE is
also reported per variable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ALIASES = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "q1000": "specific_humidity_1000",
    "z500": "geopotential_500",
    "t850": "temperature_850",
}
DEFAULT_PLOT_VARIABLES = ["t2m", "u10m", "v10m", "q1000", "z500", "t850"]
OBS_FRAMES = tuple(range(0, 32, 4))
EARTH_RADIUS_M = 6_371_000.0
EARTH_OMEGA_S = 7.2921159e-5
DRY_AIR_GAS_CONSTANT = 287.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/sr/baseline_independent")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/metrics/baseline_independent"),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(17, 25)))
    parser.add_argument(
        "--plot-variables", nargs="+", default=DEFAULT_PLOT_VARIABLES,
        help="Aliases or full variable names used in summary and spectrum plots.",
    )
    return parser.parse_args()


def area_mean(values: np.ndarray, latitude_weights: np.ndarray) -> np.ndarray:
    """Area/time mean over the last (T,H,W) axes, preserving leading axes."""
    if values.ndim < 3:
        raise ValueError(f"expected at least three axes, got {values.shape}")
    if values.shape[-2] != latitude_weights.size:
        raise ValueError(
            f"latitude mismatch: values={values.shape}, weights={latitude_weights.shape}"
        )
    lon_mean = np.mean(values, axis=-1, dtype=np.float64)
    lat_mean = np.sum(
        lon_mean * latitude_weights.reshape((1,) * (lon_mean.ndim - 1) + (-1,)),
        axis=-1,
        dtype=np.float64,
    ) / np.sum(latitude_weights, dtype=np.float64)
    return np.mean(lat_mean, axis=-1, dtype=np.float64)


def mean_frame_area_rmse(
    error: np.ndarray, latitude_weights: np.ndarray
) -> np.ndarray:
    """WeatherBench-style mean of per-frame area-weighted RMSE."""
    lon_mse = np.mean(np.square(error, dtype=np.float64), axis=-1)
    lat_mse = np.sum(
        lon_mse * latitude_weights.reshape((1,) * (lon_mse.ndim - 1) + (-1,)),
        axis=-1,
    ) / np.sum(latitude_weights)
    return np.mean(np.sqrt(lat_mse), axis=-1)


def scalar_stat(npz: np.lib.npyio.NpzFile, variable: str) -> float:
    value = np.asarray(npz[variable], dtype=np.float64).reshape(-1)
    if value.size != 1:
        raise ValueError(
            f"expected scalar normalization for {variable}, found shape "
            f"{np.asarray(npz[variable]).shape}"
        )
    return float(value[0])


def resolve_variables(requested: Iterable[str], variables: list[str]) -> list[str]:
    resolved = []
    for name in requested:
        full = ALIASES.get(name, name)
        if full not in variables:
            raise ValueError(f"unknown variable {name!r} (resolved to {full!r})")
        if full not in resolved:
            resolved.append(full)
    return resolved


def load_run_paths(
    root: Path,
    data_root: Path,
    windows: list[int],
    seeds: list[int],
) -> tuple[list[Path], dict[int, list[tuple[int, Path, Path]]]]:
    latent_files = sorted((data_root / "latents" / "test").glob("*.h5"))
    if len(latent_files) != 8:
        raise ValueError(f"expected 8 active latent files, found {len(latent_files)}")
    invalid = [index for index in windows if not 0 <= index < len(latent_files)]
    if invalid:
        raise ValueError(f"window indices must be in 0..7, got {invalid}")

    runs: dict[int, list[tuple[int, Path, Path]]] = {}
    for window_index in windows:
        source_index = latent_files[window_index].stem.removeprefix("2020_")
        members = []
        for seed in seeds:
            run_dir = root / f"w{window_index:02d}_src{source_index}_seed{seed}"
            sample_dir = run_dir / f"sample_{window_index:04d}"
            for path in (
                run_dir / "metadata.json",
                sample_dir / "samples.npy",
                sample_dir / "ground_truth.npy",
                sample_dir / "observation_lowres.npy",
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
            members.append((seed, run_dir, sample_dir))
        runs[window_index] = members
    return latent_files, runs


def load_variable_members(
    members: list[tuple[int, Path, Path]], channel: int
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    truth = None
    for _, _, sample_dir in members:
        sample = np.load(sample_dir / "samples.npy", mmap_mode="r")
        if sample.ndim != 5 or sample.shape[0] != 1:
            raise ValueError(f"unexpected sample shape in {sample_dir}: {sample.shape}")
        arrays.append(np.asarray(sample[0, channel], dtype=np.float32))
        if truth is None:
            truth_array = np.load(sample_dir / "ground_truth.npy", mmap_mode="r")
            truth = np.asarray(truth_array[channel], dtype=np.float32)
    return np.stack(arrays, axis=0), truth


def zonal_power(fields: np.ndarray, latitude_weights: np.ndarray) -> np.ndarray:
    """Mean latitude-weighted zonal power after removing each zonal mean."""
    anomaly = fields - np.mean(fields, axis=-1, keepdims=True)
    transform = np.fft.rfft(anomaly, axis=-1)
    power = np.square(np.abs(transform), dtype=np.float64) / fields.shape[-1] ** 2
    # power is (..., T, H, K); average all axes except wavenumber.
    weighted = np.sum(
        power * latitude_weights.reshape((1,) * (power.ndim - 2) + (-1, 1)),
        axis=-2,
    ) / np.sum(latitude_weights)
    axes = tuple(range(weighted.ndim - 1))
    return np.mean(weighted, axis=axes)


def geostrophic_metrics(
    u: np.ndarray,
    v: np.ndarray,
    geopotential: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-member 500-hPa geostrophic cosine similarity and speed ratio."""
    lat_rad = np.deg2rad(lat.astype(np.float64))
    lon_rad = np.unwrap(np.deg2rad(lon.astype(np.float64)))
    dlon = float(np.median(np.diff(lon_rad)))
    if dlon == 0:
        raise ValueError("longitude spacing is zero")

    phi = geopotential.astype(np.float64, copy=False)
    dphi_dy = np.gradient(phi, lat_rad, axis=-2, edge_order=2) / EARTH_RADIUS_M
    dphi_dlon = (np.roll(phi, -1, axis=-1) - np.roll(phi, 1, axis=-1)) / (2 * dlon)
    cos_lat = np.cos(lat_rad)
    dphi_dx = dphi_dlon / (EARTH_RADIUS_M * cos_lat[None, None, :, None])
    coriolis = 2 * EARTH_OMEGA_S * np.sin(lat_rad)
    ug = -dphi_dy / coriolis[None, None, :, None]
    vg = dphi_dx / coriolis[None, None, :, None]

    mask = (np.abs(lat) >= 25.0) & (np.abs(lat) <= 70.0)
    weights = np.clip(np.cos(lat_rad[mask]), 0.0, None)
    uu, vv = u[:, :, mask].astype(np.float64), v[:, :, mask].astype(np.float64)
    ugg, vgg = ug[:, :, mask], vg[:, :, mask]
    weight_shape = (1, 1, -1, 1)
    w = weights.reshape(weight_shape)

    dot = np.sum((uu * ugg + vv * vgg) * w, axis=(1, 2, 3))
    actual2 = np.sum((uu * uu + vv * vv) * w, axis=(1, 2, 3))
    geo2 = np.sum((ugg * ugg + vgg * vgg) * w, axis=(1, 2, 3))
    cosine = dot / np.sqrt(np.maximum(actual2 * geo2, 1e-30))
    speed_ratio = np.sqrt(actual2 / np.maximum(geo2, 1e-30))
    return cosine, speed_ratio


def hypsometric_rmse(
    z500: np.ndarray,
    z850: np.ndarray,
    temperatures: list[np.ndarray],
    lat_weights: np.ndarray,
) -> np.ndarray:
    """Per-member residual of thickness-implied and sampled layer-mean T."""
    pressures = np.asarray([500.0, 600.0, 700.0, 850.0])
    log_p = np.log(pressures)
    temperature_stack = np.stack(temperatures, axis=-1)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    direct_layer_mean = trapezoid(temperature_stack, x=log_p, axis=-1) / (
        log_p[-1] - log_p[0]
    )
    implied_layer_mean = (z500 - z850) / (
        DRY_AIR_GAS_CONSTANT * math.log(850.0 / 500.0)
    )
    return np.sqrt(area_mean(np.square(implied_layer_mean - direct_layer_mean), lat_weights))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: list[dict], group_key: str, metrics: list[str], std_suffix: str
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    output = []
    for group, group_rows in grouped.items():
        record: dict[str, object] = {group_key: group}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows])
            record[metric] = float(np.mean(values))
            record[f"{metric}_{std_suffix}"] = float(np.std(values))
        output.append(record)
    return output


def plot_results(
    output_dir: Path,
    per_variable: list[dict],
    plot_variables: list[str],
    spectra: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_variable = {row["variable"]: row for row in per_variable}
    labels = [next((a for a, v in ALIASES.items() if v == name), name) for name in plot_variables]
    x = np.arange(len(plot_variables))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(
        x - 0.2,
        [by_variable[v]["standardized_skill_rmse_unobserved"] for v in plot_variables],
        width=0.4,
        label="ensemble-mean error",
    )
    axes[0].bar(
        x + 0.2,
        [by_variable[v]["standardized_spread_unobserved"] for v in plot_variables],
        width=0.4,
        label="ensemble spread",
    )
    axes[0].set_ylabel("standardized units")
    axes[0].set_title("Spread and skill")
    axes[0].legend(fontsize=8)

    axes[1].bar(
        x - 0.2,
        [by_variable[v]["coverage_range_unobserved"] for v in plot_variables],
        width=0.4,
        label="ensemble range",
    )
    axes[1].bar(
        x + 0.2,
        [by_variable[v]["coverage_2sigma_unobserved"] for v in plot_variables],
        width=0.4,
        label=r"mean $\pm2\sigma$",
    )
    axes[1].axhline(0.954, color="0.3", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Empirical coverage")
    axes[1].legend(fontsize=8)

    axes[2].bar(
        x,
        [by_variable[v]["pairwise_mse_unobserved_standardized"] for v in plot_variables],
    )
    axes[2].set_title("Pairwise trajectory diversity")
    axes[2].set_ylabel("mean squared standardized difference")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_metrics.png", dpi=180)
    plt.close(fig)

    ncols = 3
    nrows = math.ceil(len(plot_variables) / ncols)
    fig, axes_array = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    for ax, variable, label in zip(axes_array.flat, plot_variables, labels):
        k = spectra["wavenumber"]
        valid = k > 0
        ax.loglog(k[valid], spectra[f"{variable}__truth"][valid], label="ERA5")
        ax.loglog(
            k[valid], spectra[f"{variable}__member_mean"][valid], label="members"
        )
        ax.loglog(
            k[valid], spectra[f"{variable}__ensemble_mean"][valid],
            label="ensemble mean", linestyle="--",
        )
        ax.set_title(label)
        ax.set_xlabel("zonal wavenumber")
        ax.set_ylabel("power")
        ax.grid(alpha=0.2, which="both")
    for ax in axes_array.flat[len(plot_variables):]:
        ax.axis("off")
    axes_array.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "zonal_spectra.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latent_files, runs = load_run_paths(
        args.root, args.data_root, args.windows, args.seeds
    )
    first_run = runs[args.windows[0]][0][1]
    first_metadata = json.loads((first_run / "metadata.json").read_text())
    variables = list(first_metadata["variables"])
    plot_variables = resolve_variables(args.plot_variables, variables)
    channels = {name: index for index, name in enumerate(variables)}

    lat = np.asarray(np.load(args.data_root / "era5" / "lat.npy")).squeeze()
    lon = np.asarray(np.load(args.data_root / "era5" / "lon.npy")).squeeze()
    lat_weights = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    obs_lat_weights = lat_weights[::8]
    means_npz = np.load(args.data_root / "era5" / "normalize_mean.npz")
    stds_npz = np.load(args.data_root / "era5" / "normalize_std.npz")

    rows: list[dict] = []
    physical_rows: list[dict] = []
    spectra_by_variable: dict[str, dict[str, list[np.ndarray]]] = {
        variable: {"truth": [], "member_mean": [], "ensemble_mean": []}
        for variable in plot_variables
    }

    for window_index in args.windows:
        members = runs[window_index]
        source_index = latent_files[window_index].stem.removeprefix("2020_")
        print(f"Window {window_index} (source {source_index})")

        reference_obs = np.load(
            members[0][2] / "observation_lowres.npy", mmap_mode="r"
        )
        if reference_obs.ndim != 5 or reference_obs.shape[0] != 1:
            raise ValueError(f"unexpected observation shape: {reference_obs.shape}")

        for channel, variable in enumerate(variables):
            samples, truth = load_variable_members(members, channel)
            observed_indices = list(OBS_FRAMES)
            mean = scalar_stat(means_npz, variable)
            std = scalar_stat(stds_npz, variable)
            if std <= 0:
                raise ValueError(f"non-positive normalization std for {variable}: {std}")
            samples_std = (samples - mean) / std
            truth_std = (truth - mean) / std
            ensemble_mean = np.mean(samples, axis=0)
            ensemble_mean_std = np.mean(samples_std, axis=0)
            sample_stddev = np.std(samples_std, axis=0, ddof=1)

            member_rmse = mean_frame_area_rmse(
                samples - truth[None], lat_weights
            )
            ensemble_rmse = float(
                mean_frame_area_rmse(ensemble_mean - truth, lat_weights)
            )
            obs_rmse = float(
                mean_frame_area_rmse(
                    ensemble_mean[observed_indices] - truth[observed_indices],
                    lat_weights,
                )
            )
            unobs_mask = np.ones(truth.shape[0], dtype=bool)
            unobs_mask[observed_indices] = False
            unobs_rmse = float(
                mean_frame_area_rmse(
                    ensemble_mean[unobs_mask] - truth[unobs_mask], lat_weights
                )
            )
            member_unobs_rmse = mean_frame_area_rmse(
                samples[:, unobs_mask] - truth[None, unobs_mask], lat_weights
            )

            spread_mse = float(area_mean(np.square(sample_stddev), lat_weights))
            skill_mse = float(
                area_mean(np.square(ensemble_mean_std - truth_std), lat_weights)
            )
            pairwise_mse = 2.0 * spread_mse
            spread_mse_unobs = float(
                area_mean(np.square(sample_stddev[unobs_mask]), lat_weights)
            )
            skill_mse_unobs = float(area_mean(
                np.square(ensemble_mean_std[unobs_mask] - truth_std[unobs_mask]),
                lat_weights,
            ))
            coverage_range = float(
                area_mean(
                    (truth_std >= np.min(samples_std, axis=0))
                    & (truth_std <= np.max(samples_std, axis=0)),
                    lat_weights,
                )
            )
            coverage_range_unobs = float(
                area_mean(
                    (truth_std[unobs_mask] >= np.min(samples_std[:, unobs_mask], axis=0))
                    & (truth_std[unobs_mask] <= np.max(samples_std[:, unobs_mask], axis=0)),
                    lat_weights,
                )
            )
            coverage_2sigma_unobs = float(
                area_mean(
                    np.abs(
                        truth_std[unobs_mask] - ensemble_mean_std[unobs_mask]
                    ) <= 2.0 * sample_stddev[unobs_mask],
                    lat_weights,
                )
            )
            coverage_2sigma = float(
                area_mean(
                    np.abs(truth_std - ensemble_mean_std) <= 2.0 * sample_stddev,
                    lat_weights,
                )
            )

            measurement = np.asarray(reference_obs[0, channel], dtype=np.float32)
            predicted_obs = samples_std[:, observed_indices, ::8, ::8]
            if predicted_obs.shape[1:] != measurement.shape:
                raise ValueError(
                    f"observation shape mismatch for {variable}: "
                    f"prediction {predicted_obs.shape}, measurement {measurement.shape}"
                )
            obs_ensemble_error = np.mean(predicted_obs, axis=0) - measurement
            obs_member_error = predicted_obs - measurement[None]

            sample_increments = np.diff(samples_std, axis=1)
            truth_increments = np.diff(truth_std, axis=0)
            mean_increment_error = np.mean(sample_increments, axis=0) - truth_increments
            truth_increment_energy = float(
                area_mean(np.square(truth_increments), lat_weights)
            )
            sample_increment_energy = float(
                np.mean(area_mean(np.square(sample_increments), lat_weights))
            )

            row = {
                "window_index": window_index,
                "source_index": source_index,
                "variable": variable,
                "ensemble_rmse_physical": ensemble_rmse,
                "ensemble_rmse_observed_frames_physical": obs_rmse,
                "ensemble_rmse_unobserved_frames_physical": unobs_rmse,
                "mean_member_rmse_physical": float(np.mean(member_rmse)),
                "best_of_k_rmse_physical": float(np.min(member_rmse)),
                "best_of_k_rmse_unobserved_physical": float(
                    np.min(member_unobs_rmse)
                ),
                "standardized_skill_rmse": math.sqrt(skill_mse),
                "standardized_spread": math.sqrt(spread_mse),
                "spread_skill_ratio": math.sqrt(spread_mse / max(skill_mse, 1e-30)),
                "pairwise_mse_standardized": pairwise_mse,
                "coverage_range": coverage_range,
                "coverage_2sigma": coverage_2sigma,
                "standardized_skill_rmse_unobserved": math.sqrt(skill_mse_unobs),
                "standardized_spread_unobserved": math.sqrt(spread_mse_unobs),
                "spread_skill_ratio_unobserved": math.sqrt(
                    spread_mse_unobs / max(skill_mse_unobs, 1e-30)
                ),
                "pairwise_mse_unobserved_standardized": 2.0 * spread_mse_unobs,
                "coverage_range_unobserved": coverage_range_unobs,
                "coverage_2sigma_unobserved": coverage_2sigma_unobs,
                "observation_rmse_standardized": float(
                    mean_frame_area_rmse(obs_ensemble_error, obs_lat_weights)
                ),
                "mean_member_observation_rmse_standardized": float(
                    np.mean(mean_frame_area_rmse(obs_member_error, obs_lat_weights))
                ),
                "temporal_increment_rmse_standardized": float(
                    np.sqrt(area_mean(np.square(mean_increment_error), lat_weights))
                ),
                "temporal_increment_energy_ratio": sample_increment_energy
                / max(truth_increment_energy, 1e-30),
            }
            rows.append(row)

            if variable in spectra_by_variable:
                spectra_by_variable[variable]["truth"].append(
                    zonal_power(truth_std, lat_weights)
                )
                spectra_by_variable[variable]["member_mean"].append(
                    zonal_power(samples_std, lat_weights)
                )
                spectra_by_variable[variable]["ensemble_mean"].append(
                    zonal_power(ensemble_mean_std, lat_weights)
                )

        required_physical = [
            "u_component_of_wind_500", "v_component_of_wind_500", "geopotential_500",
            "geopotential_850", "temperature_500", "temperature_600",
            "temperature_700", "temperature_850",
        ]
        if all(name in channels for name in required_physical):
            loaded_fields = {
                name: load_variable_members(members, channels[name])
                for name in required_physical
            }
            fields = {name: value[0] for name, value in loaded_fields.items()}
            truth_fields = {name: value[1][None] for name, value in loaded_fields.items()}
            geo_cos, geo_speed = geostrophic_metrics(
                fields["u_component_of_wind_500"],
                fields["v_component_of_wind_500"],
                fields["geopotential_500"],
                lat,
                lon,
            )
            hyp_rmse = hypsometric_rmse(
                fields["geopotential_500"],
                fields["geopotential_850"],
                [
                    fields["temperature_500"], fields["temperature_600"],
                    fields["temperature_700"], fields["temperature_850"],
                ],
                lat_weights,
            )
            truth_geo_cos, truth_geo_speed = geostrophic_metrics(
                truth_fields["u_component_of_wind_500"],
                truth_fields["v_component_of_wind_500"],
                truth_fields["geopotential_500"],
                lat,
                lon,
            )
            truth_hyp_rmse = hypsometric_rmse(
                truth_fields["geopotential_500"],
                truth_fields["geopotential_850"],
                [
                    truth_fields["temperature_500"], truth_fields["temperature_600"],
                    truth_fields["temperature_700"], truth_fields["temperature_850"],
                ],
                lat_weights,
            )
            physical_rows.append(
                {
                    "window_index": window_index,
                    "source_index": source_index,
                    "geostrophic_cosine": float(np.mean(geo_cos)),
                    "geostrophic_cosine_member_std": float(np.std(geo_cos)),
                    "geostrophic_speed_ratio": float(np.mean(geo_speed)),
                    "geostrophic_speed_ratio_member_std": float(np.std(geo_speed)),
                    "hypsometric_rmse_k": float(np.mean(hyp_rmse)),
                    "hypsometric_rmse_k_member_std": float(np.std(hyp_rmse)),
                    "era5_geostrophic_cosine": float(truth_geo_cos[0]),
                    "era5_geostrophic_speed_ratio": float(truth_geo_speed[0]),
                    "era5_hypsometric_rmse_k": float(truth_hyp_rmse[0]),
                }
            )

    metric_names = [
        key for key in rows[0]
        if key not in {"window_index", "source_index", "variable"}
    ]
    per_variable = aggregate_rows(rows, "variable", metric_names, "window_std")
    per_variable.sort(key=lambda row: variables.index(str(row["variable"])))
    per_window_metrics = [
        "standardized_skill_rmse", "standardized_spread", "spread_skill_ratio",
        "pairwise_mse_standardized", "coverage_range", "coverage_2sigma",
        "standardized_skill_rmse_unobserved", "standardized_spread_unobserved",
        "spread_skill_ratio_unobserved", "pairwise_mse_unobserved_standardized",
        "coverage_range_unobserved", "coverage_2sigma_unobserved",
        "observation_rmse_standardized", "temporal_increment_rmse_standardized",
        "temporal_increment_energy_ratio",
    ]
    per_window = aggregate_rows(
        rows, "window_index", per_window_metrics, "variable_std"
    )
    for record in per_window:
        index = int(record["window_index"])
        record["source_index"] = latent_files[index].stem.removeprefix("2020_")

    write_csv(args.output_dir / "per_window_per_variable.csv", rows)
    write_csv(args.output_dir / "per_window_summary.csv", per_window)
    write_csv(args.output_dir / "per_variable_summary.csv", per_variable)
    write_csv(args.output_dir / "physical_consistency.csv", physical_rows)

    spectra: dict[str, np.ndarray] = {
        "wavenumber": np.arange(lon.size // 2 + 1, dtype=np.int32)
    }
    for variable, kinds in spectra_by_variable.items():
        for kind, arrays in kinds.items():
            spectra[f"{variable}__{kind}"] = np.mean(np.stack(arrays), axis=0)
    np.savez_compressed(args.output_dir / "zonal_spectra.npz", **spectra)
    spectral_rows = []
    for variable in plot_variables:
        truth_power = spectra[f"{variable}__truth"][1:]
        member_power = spectra[f"{variable}__member_mean"][1:]
        ensemble_power = spectra[f"{variable}__ensemble_mean"][1:]
        floor = max(float(np.max(truth_power)) * 1e-12, 1e-30)
        spectral_rows.append(
            {
                "variable": variable,
                "member_log_power_rmse": float(np.sqrt(np.mean(np.square(
                    np.log10(np.maximum(member_power, floor))
                    - np.log10(np.maximum(truth_power, floor))
                )))),
                "ensemble_mean_log_power_rmse": float(np.sqrt(np.mean(np.square(
                    np.log10(np.maximum(ensemble_power, floor))
                    - np.log10(np.maximum(truth_power, floor))
                )))),
                "member_high_wavenumber_power_ratio": float(
                    np.sum(member_power[len(member_power) // 2:])
                    / max(float(np.sum(truth_power[len(truth_power) // 2:])), 1e-30)
                ),
            }
        )
    write_csv(args.output_dir / "spectral_summary.csv", spectral_rows)

    spread_mse_all = np.mean(
        [float(row["standardized_spread"]) ** 2 for row in rows]
    )
    skill_mse_all = np.mean(
        [float(row["standardized_skill_rmse"]) ** 2 for row in rows]
    )
    spread_mse_unobs_all = np.mean(
        [float(row["standardized_spread_unobserved"]) ** 2 for row in rows]
    )
    skill_mse_unobs_all = np.mean(
        [float(row["standardized_skill_rmse_unobserved"]) ** 2 for row in rows]
    )
    k_members = len(args.seeds)
    summary = {
        "configuration": {
            "baseline_root": str(args.root),
            "data_root": str(args.data_root),
            "windows": args.windows,
            "source_indices": [
                latent_files[index].stem.removeprefix("2020_") for index in args.windows
            ],
            "seeds": args.seeds,
            "members_per_window": k_members,
            "variables": len(variables),
            "observed_frames": list(OBS_FRAMES),
            "spatial_observation_stride": 8,
        },
        "definitions": {
            "pairwise_mse_standardized": "mean squared difference over all unordered member pairs, area/time averaged",
            "standardized_spread": "sqrt of area/time averaged unbiased ensemble variance",
            "coverage_range": "fraction of truth inside the member minimum-maximum interval",
            "coverage_2sigma": "fraction of truth inside ensemble mean plus/minus two sample standard deviations",
            "unobserved": "metrics carrying this suffix use the 24 frames not supplied to the SR likelihood",
            "best_of_k_rmse_physical": "minimum member WeatherBench-style trajectory RMSE",
            "observation_rmse_standardized": "ensemble-mean error after the exact every-8th-gridpoint SR operator on observed frames",
        },
        "all_variable_standardized": {
            "skill_rmse": math.sqrt(skill_mse_all),
            "spread": math.sqrt(spread_mse_all),
            "spread_skill_ratio": math.sqrt(spread_mse_all / max(skill_mse_all, 1e-30)),
            "spread_skill_ratio_finite_ensemble_corrected": math.sqrt(
                (k_members + 1) / k_members
            ) * math.sqrt(spread_mse_all / max(skill_mse_all, 1e-30)),
            "pairwise_mse": float(np.mean([row["pairwise_mse_standardized"] for row in rows])),
            "coverage_range": float(np.mean([row["coverage_range"] for row in rows])),
            "coverage_2sigma": float(np.mean([row["coverage_2sigma"] for row in rows])),
            "skill_rmse_unobserved": math.sqrt(skill_mse_unobs_all),
            "spread_unobserved": math.sqrt(spread_mse_unobs_all),
            "spread_skill_ratio_unobserved": math.sqrt(
                spread_mse_unobs_all / max(skill_mse_unobs_all, 1e-30)
            ),
            "spread_skill_ratio_unobserved_finite_ensemble_corrected": math.sqrt(
                (k_members + 1) / k_members
            ) * math.sqrt(spread_mse_unobs_all / max(skill_mse_unobs_all, 1e-30)),
            "pairwise_mse_unobserved": float(np.mean([
                row["pairwise_mse_unobserved_standardized"] for row in rows
            ])),
            "coverage_range_unobserved": float(np.mean([
                row["coverage_range_unobserved"] for row in rows
            ])),
            "coverage_2sigma_unobserved": float(np.mean([
                row["coverage_2sigma_unobserved"] for row in rows
            ])),
            "observation_rmse": float(np.mean([row["observation_rmse_standardized"] for row in rows])),
            "temporal_increment_rmse": float(np.mean([row["temporal_increment_rmse_standardized"] for row in rows])),
            "temporal_increment_energy_ratio": float(np.mean([row["temporal_increment_energy_ratio"] for row in rows])),
        },
        "physical_consistency": {
            key: float(np.mean([row[key] for row in physical_rows]))
            for key in (
                "geostrophic_cosine", "geostrophic_speed_ratio", "hypsometric_rmse_k",
                "era5_geostrophic_cosine", "era5_geostrophic_speed_ratio",
                "era5_hypsometric_rmse_k",
            )
        } if physical_rows else {},
        "spectral_consistency": {
            row["variable"]: {
                key: value for key, value in row.items() if key != "variable"
            }
            for row in spectral_rows
        },
        "artifacts": {
            "per_window_per_variable": "per_window_per_variable.csv",
            "per_window_summary": "per_window_summary.csv",
            "per_variable_summary": "per_variable_summary.csv",
            "physical_consistency": "physical_consistency.csv",
            "spectral_summary": "spectral_summary.csv",
            "spectra": "zonal_spectra.npz",
            "ensemble_plot": "ensemble_metrics.png",
            "spectra_plot": "zonal_spectra.png",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_results(args.output_dir, per_variable, plot_variables, spectra)

    print(f"Wrote Step 2 baseline metrics to {args.output_dir}")
    print(json.dumps(summary["all_variable_standardized"], indent=2))


if __name__ == "__main__":
    main()
