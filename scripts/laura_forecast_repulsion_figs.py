#!/usr/bin/env python3
"""Create repulsion-guided analogues of paper Figures 17 and 19."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from laura_hindcast_repulsion_fig import (
    Tracker,
    gc_km,
    load_land_mask,
    load_observation_locations,
)


SOURCE_INDEX = 941
START = np.datetime64("2020-08-23T06:00")
LANDFALL_TIME = np.datetime64("2020-08-27T06:00")
COLORS = {2: "#377eb8", 3: "#ff7f00", 4: "#4daf4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-2d", type=Path, required=True)
    parser.add_argument("--run-3d", type=Path, required=True)
    parser.add_argument("--run-4d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=SOURCE_INDEX)
    parser.add_argument("--observations-pkl", type=Path, default=None)
    parser.add_argument(
        "--region-lat", type=float, nargs=2, default=(10.0, 42.0),
        metavar=("MIN", "MAX"),
        help=(
            "Latitude limits for the Figure-19 wind distribution "
            "(default matches DJ's preserved Laura plotting scripts)."
        ),
    )
    parser.add_argument(
        "--region-lon", type=float, nargs=2, default=(255.0, 300.0),
        metavar=("MIN", "MAX"),
        help=(
            "0--360 longitude limits for the Figure-19 wind distribution "
            "(default matches DJ's preserved Laura plotting scripts)."
        ),
    )
    parser.add_argument(
        "--kde-max-points", type=int, default=250_000,
        help="Deterministic maximum number of values used per KDE curve.",
    )
    return parser.parse_args()


def members_view(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.ndim < 5:
        raise ValueError(f"expected sample array with at least 5 dimensions: {path}")
    return array.reshape(-1, *array.shape[-4:])


def field_view(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return array.reshape(-1, *array.shape[-4:])[0]


def load_run(path: Path, expected_days: int, source_index: int) -> dict:
    path = path.resolve()
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("particle_coupling") != "global_all_particles":
        raise ValueError(f"{path} is not a joint-particle repulsion run")
    if list(metadata.get("source_indices", [])) != [source_index]:
        raise ValueError(f"{path} does not contain source index {source_index}")
    obs_frames = list(metadata.get("obs_frames", []))
    expected_frames = list(range(expected_days * 4))
    if obs_frames != expected_frames:
        raise ValueError(
            f"{path} has observed frames {obs_frames}; expected {expected_frames}"
        )
    if float(metadata.get("repulsion_strength", 0.0)) <= 0:
        raise ValueError(f"{path} has no active repulsion strength")
    sample_path = path / f"sample_{source_index:04d}" / "samples.npy"
    truth_path = path / f"sample_{source_index:04d}" / "ground_truth.npy"
    if not sample_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError(f"incomplete sample directory under {path}")
    return {
        "days": expected_days,
        "path": path,
        "metadata": metadata,
        "sample_path": sample_path,
        "truth_path": truth_path,
    }


def lon_for_plot(lon):
    lon = np.asarray(lon)
    return np.where(lon > 180.0, lon - 360.0, lon)


def make_map(lat, lon, land_mask):
    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    ax.set_facecolor("#edf5fa")
    ax.set_xlim(-100.0, -65.0)
    ax.set_ylim(15.0, 35.0)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if land_mask is not None:
        lat = np.asarray(lat).squeeze()
        plot_lon = lon_for_plot(np.asarray(lon).squeeze())
        order = np.argsort(plot_lon)
        mask = np.asarray(land_mask)[:, order]
        ax.contourf(
            plot_lon[order], lat, mask,
            levels=[0.5, max(1.01, float(np.nanmax(mask)) + 0.01)],
            colors=["#e5e0d5"], zorder=0,
        )
        ax.contour(
            plot_lon[order], lat, mask, levels=[0.5],
            colors=["0.35"], linewidths=0.65, zorder=1,
        )
    ax.grid(color="0.75", linewidth=0.45, alpha=0.4)
    return fig, ax


def gaussian_kde(values: np.ndarray, grid: np.ndarray, max_points: int):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("cannot estimate a density from an empty array")
    if len(values) > max_points:
        select = np.linspace(0, len(values) - 1, max_points, dtype=np.int64)
        values = values[select]

    std = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    robust_sigma = min(std, float((q75 - q25) / 1.34))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = max(std, float(np.ptp(values)) / 50.0, 1e-3)
    bandwidth = 0.9 * robust_sigma * len(values) ** (-0.2)
    bandwidth = max(float(bandwidth), 1e-3)

    density = np.zeros_like(grid, dtype=np.float64)
    chunk = 8_000
    for first in range(0, len(values), chunk):
        z = (grid[:, None] - values[None, first:first + chunk]) / bandwidth
        density += np.exp(-0.5 * z * z).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    return density, bandwidth, values


def region_values(video, channel, iy, ix):
    field = video[:, channel] if video.ndim == 5 else video[channel][None]
    region = np.take(np.take(field, iy, axis=-2), ix, axis=-1)
    return np.asarray(region, dtype=np.float32).reshape(-1)


def track_metrics(tracks, truth, first_forecast_frame, landfall_frame):
    frames = np.arange(first_forecast_frame, landfall_frame + 1)
    member_error = [
        float(
            gc_km(
                track["lat"][frames], track["lon"][frames],
                truth["lat"][frames], truth["lon"][frames],
            ).mean()
        )
        for track in tracks
    ]
    landfall_error = [
        float(
            gc_km(
                track["lat"][landfall_frame], track["lon"][landfall_frame],
                truth["lat"][landfall_frame], truth["lon"][landfall_frame],
            )
        )
        for track in tracks
    ]
    pairwise = [
        float(
            gc_km(
                left["lat"][frames], left["lon"][frames],
                right["lat"][frames], right["lon"][frames],
            ).mean()
        )
        for left, right in combinations(tracks, 2)
    ]
    return {
        "forecast_frames": frames.tolist(),
        "track_error_km_members": member_error,
        "track_error_km_mean": float(np.mean(member_error)),
        "track_error_km_median": float(np.median(member_error)),
        "landfall_error_km_members": landfall_error,
        "landfall_error_km_mean": float(np.mean(landfall_error)),
        "pairwise_track_distance_km_mean": float(np.mean(pairwise)),
    }


def save_figure(fig, output_dir: Path, stem: str):
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    args = parse_args()
    if args.source_index != SOURCE_INDEX:
        raise ValueError(
            f"the paper Figure-17 experiment starts at source {SOURCE_INDEX}; "
            f"received {args.source_index}"
        )
    runs = {
        days: load_run(path, days, args.source_index)
        for days, path in ((2, args.run_2d), (3, args.run_3d), (4, args.run_4d))
    }
    variables = list(runs[2]["metadata"]["variables"])
    for run in runs.values():
        if list(run["metadata"]["variables"]) != variables:
            raise ValueError("the three runs do not share an identical variable order")
    required = [
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    missing = [name for name in required if name not in variables]
    if missing:
        raise ValueError(f"missing required variables: {missing}")

    bundle = args.bundle.resolve()
    lat = np.load(bundle / "data" / "source_root" / "lat.npy")
    lon = np.load(bundle / "data" / "source_root" / "lon.npy")
    tracker = Tracker(
        lat, lon,
        variables.index("mean_sea_level_pressure"),
        variables.index("10m_u_component_of_wind"),
        variables.index("10m_v_component_of_wind"),
    )
    truth = field_view(runs[2]["truth_path"])
    initial_truth_track = tracker.track(truth)
    seed = (initial_truth_track["lat"][0], initial_truth_track["lon"][0])
    truth_track = tracker.track(truth, seed=seed, seed_frame=0)
    timesteps = truth.shape[1]
    times = START + np.arange(timesteps) * np.timedelta64(6, "h")
    landfall_frame = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    plot_stop = min(landfall_frame + 5, timesteps)

    lat1, lat2 = args.region_lat
    lon1, lon2 = args.region_lon
    lat_flat = np.asarray(lat).squeeze()
    lon_flat = np.asarray(lon).squeeze() % 360.0
    iy = np.where((lat_flat >= lat1) & (lat_flat <= lat2))[0]
    ix = np.where((lon_flat >= lon1) & (lon_flat <= lon2))[0]
    if not len(iy) or not len(ix):
        raise ValueError("the requested Figure-19 region is outside the ERA5 grid")

    track_sets = {}
    wind_values = {}
    particle_counts = {}
    alphas = set()
    guidance_modes = set()
    for days, run in runs.items():
        samples = members_view(run["sample_path"])
        particle_counts[days] = int(samples.shape[0])
        track_sets[days] = [
            tracker.track(samples[index], seed=seed, seed_frame=0)
            for index in range(samples.shape[0])
        ]
        wind_values[days] = region_values(
            samples, variables.index("10m_u_component_of_wind"), iy, ix
        )
        alphas.add(float(run["metadata"]["repulsion_strength"]))
        guidance_modes.add(
            "full-DPS" if run["metadata"].get("guide_through_net") else "decoder-only"
        )
    if len(alphas) != 1:
        raise ValueError(f"runs use different repulsion strengths: {sorted(alphas)}")
    alpha = alphas.pop()

    obs_path = args.observations_pkl
    if obs_path is None:
        raw = bundle / "data" / "observations" / "global_joint_2020_year.pkl"
        superob = bundle / "data" / "observations" / "global_joint_2020_year_superob.pkl"
        obs_path = raw if raw.is_file() else superob
    obs_points = load_observation_locations(obs_path, args.source_index)
    land_mask = load_land_mask(bundle, lat, lon)

    fig, ax = make_map(lat, lon, land_mask)
    if obs_points is not None:
        ax.scatter(
            lon_for_plot(obs_points[:, 1]), obs_points[:, 0],
            s=5, color="red", alpha=0.5, zorder=2,
            label="in-situ observation sites (Aug 23 06Z)",
        )
    for days in (2, 3, 4):
        for index, track in enumerate(track_sets[days]):
            label = None
            if index == 0:
                label = (
                    f"Repulsion-guided forecast, {days} days observed "
                    f"(K={particle_counts[days]})"
                )
            ax.plot(
                lon_for_plot(track["lon"][:plot_stop]),
                track["lat"][:plot_stop],
                color=COLORS[days], linewidth=1.0, alpha=0.58,
                zorder=3 + days, label=label,
            )
    ax.plot(
        lon_for_plot(truth_track["lon"][:plot_stop]),
        truth_track["lat"][:plot_stop],
        color="black", linewidth=3.0, zorder=8, label="ERA5 reference",
    )
    for frame in range(0, landfall_frame + 1, 4):
        x = float(lon_for_plot(truth_track["lon"][frame]))
        y = float(truth_track["lat"][frame])
        ax.scatter([x], [y], color="black", s=22, zorder=9)
        ax.annotate(str(times[frame])[5:10], (x + 0.35, y - 0.75), fontsize=8)
    ax.scatter(
        [lon_for_plot(truth_track["lon"][landfall_frame])],
        [truth_track["lat"][landfall_frame]],
        marker="*", s=310, color="gold", edgecolor="black", zorder=10,
        label="landfall (Aug 27 06Z)",
    )
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.95)
    ax.set_title(
        "Hurricane Laura forecasts with repulsion-guided posterior sampling "
        f"(alpha={alpha:.2f})"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    track_outputs = save_figure(fig, args.output_dir, "figure17_repulsion_tracks")

    truth_wind = region_values(
        truth, variables.index("10m_u_component_of_wind"), iy, ix
    )
    # all_values = np.concatenate([truth_wind, *[wind_values[d] for d in (2, 3, 4)]])
    # lo, hi = np.percentile(all_values[np.isfinite(all_values)], [0.002, 99.998])
    # margin = 0.08 * max(float(hi - lo), 1.0)
    # grid = np.linspace(float(lo - margin), float(hi + margin), 600)

    # Use the same fixed wind range for all KDE comparisons.
    grid = np.linspace(-22.0, 22.0, 800)

    density_info = {}
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for days in (2, 3, 4):
        density, bandwidth, used = gaussian_kde(
            wind_values[days], grid, args.kde_max_points
        )
        ax.plot(
            grid, density, color=COLORS[days], linewidth=2.1,
            label=f"Repulsion forecast, {days} days observed",
        )
        density_info[str(days)] = {
            "count_total": int(wind_values[days].size),
            "count_kde": int(used.size),
            "bandwidth_ms": float(bandwidth),
            "mean_ms": float(np.mean(wind_values[days])),
            "std_ms": float(np.std(wind_values[days])),
            "quantiles_ms": np.percentile(
                wind_values[days], [0.1, 1.0, 50.0, 99.0, 99.9]
            ).tolist(),
        }
    truth_density, truth_bandwidth, truth_used = gaussian_kde(
        truth_wind, grid, args.kde_max_points
    )
    ax.plot(grid, truth_density, color="black", linewidth=2.6, label="ERA5")
    ax.axvline(0.0, color="0.72", linewidth=0.8)
    ax.set_xlim(-22.0, 22.0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5, top=10**(-0.5))
    ax.set_xlabel("10 m zonal wind u [m/s]")
    ax.set_ylabel("probability density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    ax.set_title(
        "Hurricane Laura regional wind distribution with repulsion-guided sampling"
    )
    fig.tight_layout()
    distribution_outputs = save_figure(
        fig, args.output_dir, "figure19_repulsion_u10_distribution"
    )

    provenance = {
        "experiment": "repulsion-guided analogues of paper Figures 17 and 19",
        "source_index": args.source_index,
        "window_start": str(START),
        "landfall_time": str(LANDFALL_TIME),
        "repulsion_strength": alpha,
        "guidance_modes": sorted(guidance_modes),
        "runs": {
            str(days): {
                "path": str(runs[days]["path"]),
                "observed_frames": runs[days]["metadata"]["obs_frames"],
                "particles": particle_counts[days],
                "track_metrics": track_metrics(
                    track_sets[days], truth_track, days * 4, landfall_frame
                ),
            }
            for days in (2, 3, 4)
        },
        "figure17": {
            "definition": "individual repulsion-guided member tracks; no ensemble mean",
            "plot_frames": list(range(plot_stop)),
            "outputs": [str(path) for path in track_outputs],
        },
        "figure19": {
            "definition": "KDE of all member, time, and gridpoint u10 values in the stated region",
            "region_lat": list(args.region_lat),
            "region_lon_0_360": list(args.region_lon),
            "frames": list(range(timesteps)),
            "era5": {
                "count_total": int(truth_wind.size),
                "count_kde": int(truth_used.size),
                "bandwidth_ms": float(truth_bandwidth),
            },
            "posterior": density_info,
            "outputs": [str(path) for path in distribution_outputs],
        },
    }
    (args.output_dir / "laura_forecast_repulsion_figures.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print(f"wrote repulsion forecast figures to {args.output_dir}")


if __name__ == "__main__":
    main()
