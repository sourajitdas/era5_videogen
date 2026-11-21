#!/usr/bin/env python3
"""Aggregate batched K=8 Laura oracle-mask runs into a 100-member diagnostic."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from laura_forecast_repulsion_figs import (
    LANDFALL_TIME,
    SOURCE_INDEX,
    START,
    field_view,
    load_run,
    lon_for_plot,
    members_view,
    save_figure,
)
from laura_hindcast_repulsion_fig import Tracker, gc_km, load_land_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument(
        "--batch-run", type=Path, action="append", default=[],
        help="Additional K=8 run; repeat once per seed.",
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=SOURCE_INDEX)
    parser.add_argument("--obs-days", type=int, choices=[2], default=2)
    parser.add_argument("--target-members", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261112)
    parser.add_argument(
        "--track-density-bandwidth-km",
        type=float,
        default=0.0,
        help=(
            "Gaussian-kernel bandwidth for the empirical posterior density in "
            "trajectory space. A non-positive value uses the median pairwise "
            "track distance."
        ),
    )
    args = parser.parse_args()
    if args.target_members < 8:
        parser.error("--target-members must be at least 8")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


def track_error_metrics(tracks, truth_track, first_forecast, landfall):
    frames = np.arange(first_forecast, landfall + 1)
    track_errors = np.asarray(
        [
            np.mean(
                gc_km(
                    track["lat"][frames],
                    track["lon"][frames],
                    truth_track["lat"][frames],
                    truth_track["lon"][frames],
                )
            )
            for track in tracks
        ],
        dtype=np.float64,
    )
    landfall_errors = np.asarray(
        [
            gc_km(
                track["lat"][landfall],
                track["lon"][landfall],
                truth_track["lat"][landfall],
                truth_track["lon"][landfall],
            )
            for track in tracks
        ],
        dtype=np.float64,
    )
    pairwise = []
    for left, right in combinations(tracks, 2):
        pairwise.append(
            np.mean(
                gc_km(
                    left["lat"][frames],
                    left["lon"][frames],
                    right["lat"][frames],
                    right["lon"][frames],
                )
            )
        )
    return track_errors, landfall_errors, np.asarray(pairwise, dtype=np.float64)


def posterior_track_density(tracks, first_forecast, landfall, bandwidth_km=0.0):
    """Estimate relative posterior support from neighboring sampled tracks.

    The flow sampler returns equally weighted Monte Carlo samples rather than a
    normalized probability for each member.  We therefore estimate density in
    forecast-trajectory space.  The distance between two members is their mean
    great-circle separation across forecast frames.  A Gaussian kernel turns
    those distances into a local-density score.  The returned percentile is a
    robust plotting scale: zero denotes the most isolated/extreme member and
    one denotes the most densely supported/typical member.
    """
    count = len(tracks)
    if count < 2:
        raise ValueError("posterior track density requires at least two members")
    frames = np.arange(first_forecast, landfall + 1)
    distance = np.zeros((count, count), dtype=np.float64)
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            value = float(
                np.mean(
                    gc_km(
                        tracks[left_index]["lat"][frames],
                        tracks[left_index]["lon"][frames],
                        tracks[right_index]["lat"][frames],
                        tracks[right_index]["lon"][frames],
                    )
                )
            )
            distance[left_index, right_index] = value
            distance[right_index, left_index] = value
    positive = distance[np.triu_indices(count, k=1)]
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("all posterior track distances are zero or non-finite")
    bandwidth = float(bandwidth_km)
    if bandwidth <= 0.0:
        bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError(f"invalid track-density bandwidth: {bandwidth}")

    kernel = np.exp(-0.5 * np.square(distance / bandwidth))
    np.fill_diagonal(kernel, 0.0)
    density = np.sum(kernel, axis=1) / (count - 1)
    order = np.argsort(density, kind="stable")
    percentile = np.empty(count, dtype=np.float64)
    percentile[order] = np.linspace(0.0, 1.0, count)
    return density, percentile, bandwidth, distance


def draw_map(ax, latitude, longitude, land_mask):
    ax.set_facecolor("#edf5fa")
    ax.set_xlim(-100.0, -65.0)
    ax.set_ylim(15.0, 35.0)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if land_mask is not None:
        lat = np.asarray(latitude).squeeze()
        plot_lon = lon_for_plot(np.asarray(longitude).squeeze())
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


def main() -> None:
    args = parse_args()
    bundle = args.bundle.expanduser().resolve()
    manifest_path = args.selection_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("oracle_target_uses_future_era5", False):
        raise ValueError("the 100-member diagnostic requires the oracle site mask")

    run_paths = [args.reference_run, *args.batch_run]
    runs = [load_run(path, args.obs_days, args.source_index) for path in run_paths]
    available = sum(
        members_view(run["sample_path"]).shape[0] for run in runs
    )
    if available < args.target_members:
        raise RuntimeError(
            f"only {available} completed members are available; "
            f"need {args.target_members}"
        )
    variables = list(runs[0]["metadata"]["variables"])
    selected_keys = list(manifest["selected_site_keys"])
    seeds = []
    for run in runs:
        metadata = run["metadata"]
        if list(metadata["variables"]) != variables:
            raise ValueError(f"variable mismatch in {run['path']}")
        run_selection = metadata.get("observation_selection") or {}
        if list(run_selection.get("selected_site_keys", [])) != selected_keys:
            raise ValueError(f"observation mask mismatch in {run['path']}")
        seeds.append(int(metadata["seed"]))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"batch seeds are not unique: {seeds}")

    latitude = np.load(bundle / "data" / "source_root" / "lat.npy")
    longitude = np.load(bundle / "data" / "source_root" / "lon.npy")
    tracker = Tracker(
        latitude,
        longitude,
        variables.index("mean_sea_level_pressure"),
        variables.index("10m_u_component_of_wind"),
        variables.index("10m_v_component_of_wind"),
    )
    truth = field_view(runs[0]["truth_path"])
    initial_truth = tracker.track(truth)
    track_seed = (initial_truth["lat"][0], initial_truth["lon"][0])
    truth_track = tracker.track(truth, seed=track_seed, seed_frame=0)

    tracks = []
    member_provenance = []
    for run, run_seed in zip(runs, seeds, strict=True):
        members = members_view(run["sample_path"])
        for member_index in range(members.shape[0]):
            if len(tracks) >= args.target_members:
                break
            tracks.append(
                tracker.track(members[member_index], seed=track_seed, seed_frame=0)
            )
            member_provenance.append(
                {
                    "seed": run_seed,
                    "member_index_within_batch": member_index,
                    "run": str(run["path"]),
                }
            )
        if len(tracks) >= args.target_members:
            break

    first_forecast = args.obs_days * 4
    landfall = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    track_errors, landfall_errors, pairwise = track_error_metrics(
        tracks, truth_track, first_forecast, landfall
    )
    (
        track_density,
        track_density_percentile,
        track_density_bandwidth,
        track_distance_matrix,
    ) = posterior_track_density(
        tracks,
        first_forecast,
        landfall,
        bandwidth_km=args.track_density_bandwidth_km,
    )
    n = np.arange(1, args.target_members + 1)
    running_mean = np.cumsum(landfall_errors) / n
    rng = np.random.default_rng(args.bootstrap_seed)
    draws = rng.choice(
        landfall_errors,
        size=(args.bootstrap_replicates, args.target_members),
        replace=True,
    )
    bootstrap_running = np.cumsum(draws, axis=1) / n[None]
    bootstrap_low, bootstrap_high = np.quantile(
        bootstrap_running, [0.025, 0.975], axis=0
    )
    final_bootstrap = bootstrap_running[:, -1]

    selected_sites = np.asarray(
        [[x["latitude"], x["longitude"]] for x in manifest["selected_sites"]],
        dtype=np.float64,
    )
    dropped_sites = np.asarray(
        [[x["latitude"], x["longitude"]] for x in manifest["dropped_sites"]],
        dtype=np.float64,
    )
    land_mask = load_land_mask(bundle, latitude, longitude)
    fig, (ax_map, ax_convergence) = plt.subplots(1, 2, figsize=(16.0, 6.2))
    draw_map(ax_map, latitude, longitude, land_mask)
    ax_map.scatter(
        lon_for_plot(dropped_sites[:, 1]), dropped_sites[:, 0],
        marker="x", s=16, linewidths=0.65, color="0.5", alpha=0.55,
        zorder=2, label=f"dropped sites (n={len(dropped_sites)})",
    )
    ax_map.scatter(
        lon_for_plot(selected_sites[:, 1]), selected_sites[:, 0],
        marker="o", s=18, facecolor="#2ca25f", edgecolor="black",
        linewidth=0.3, alpha=0.8, zorder=3,
        label=f"oracle-selected sites (n={len(selected_sites)})",
    )
    plot_stop = min(landfall + 5, len(truth_track["lat"]))
    density_cmap = plt.get_cmap("coolwarm_r")
    density_norm = Normalize(vmin=0.0, vmax=1.0)
    # Draw well-supported tracks first so rare, extreme tracks remain visible.
    for index in np.argsort(track_density_percentile)[::-1]:
        track = tracks[index]
        ax_map.plot(
            lon_for_plot(track["lon"][:plot_stop]), track["lat"][:plot_stop],
            color=density_cmap(density_norm(track_density_percentile[index])),
            linewidth=0.8, alpha=0.42, zorder=4,
        )
    ax_map.plot(
        lon_for_plot(truth_track["lon"][:plot_stop]), truth_track["lat"][:plot_stop],
        color="black", linewidth=3.0, zorder=8, label="ERA5 reference track",
    )
    ax_map.scatter(
        [lon_for_plot(truth_track["lon"][landfall])],
        [truth_track["lat"][landfall]], marker="*", s=280,
        color="gold", edgecolor="black", zorder=10,
        label="landfall (Aug 27 06Z)",
    )
    typical_track_legend = Line2D(
        [0], [0], color=density_cmap(1.0), linewidth=1.5,
        label=f"higher-support posterior tracks (n={len(tracks)} total)",
    )
    extreme_track_legend = Line2D(
        [0], [0], color=density_cmap(0.0), linewidth=1.5,
        label="lower-support / extreme posterior tracks",
    )
    ax_map.set_title(
        f"Optimized sensor mask and {args.target_members} posterior trajectories"
    )
    handles, labels = ax_map.get_legend_handles_labels()
    handles[2:2] = [typical_track_legend, extreme_track_legend]
    labels[2:2] = [
        typical_track_legend.get_label(),
        extreme_track_legend.get_label(),
    ]
    ax_map.legend(handles, labels, loc="upper right", fontsize=7.0, framealpha=0.95)
    density_mappable = ScalarMappable(norm=density_norm, cmap=density_cmap)
    density_mappable.set_array([])
    density_colorbar = fig.colorbar(
        density_mappable, ax=ax_map, pad=0.018, fraction=0.046, shrink=0.78,
    )
    density_colorbar.set_label(
        "relative posterior track support\nred = isolated/extreme; blue = typical"
    )
    density_colorbar.set_ticks([0.0, 0.5, 1.0])
    density_colorbar.set_ticklabels(["low", "median", "high"])

    ax_convergence.fill_between(
        n, bootstrap_low, bootstrap_high,
        color="#9ecae1", alpha=0.45, label="95% bootstrap interval",
    )
    ax_convergence.plot(
        n, running_mean, color="#d7301f", linewidth=2.0,
        label="cumulative mean (seed order)",
    )
    first_batch_mean = float(np.mean(landfall_errors[:8]))
    final_mean = float(np.mean(landfall_errors))
    ax_convergence.axhline(
        first_batch_mean, color="0.45", linestyle=":", linewidth=1.5,
        label=f"original K=8 estimate: {first_batch_mean:.0f} km",
    )
    ax_convergence.axhline(
        final_mean, color="black", linestyle="--", linewidth=1.5,
        label=f"100-member mean: {final_mean:.0f} km",
    )
    ax_convergence.set_xlim(1, args.target_members)
    ax_convergence.set_xlabel("number of posterior members")
    ax_convergence.set_ylabel("mean landfall-position error [km]")
    ax_convergence.set_title("Landfall-error convergence")
    ax_convergence.grid(alpha=0.25)
    ax_convergence.legend(loc="upper right", fontsize=8.0)
    fig.suptitle(
        "Hurricane Laura oracle-mask ensemble: repulsion coupled within K=8 batches",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"oracle_ensemble{args.target_members}_density_convergence"
    outputs = save_figure(fig, args.output_dir, stem)

    metrics = {
        "experiment": "oracle-mask batched posterior ensemble",
        "source_index": args.source_index,
        "observed_days": args.obs_days,
        "target_members": args.target_members,
        "available_members": available,
        "members_used": len(tracks),
        "batch_size": 8,
        "batch_count_used": len(set(x["seed"] for x in member_provenance)),
        "seeds": sorted(set(x["seed"] for x in member_provenance)),
        "repulsion_scope": "within each K=8 batch; no cross-batch repulsion",
        "selection_manifest": str(manifest_path),
        "selected_site_count": int(manifest["selected_site_count"]),
        "dropped_site_count": len(manifest["dropped_sites"]),
        "landfall_error_definition": (
            "mean of member-wise great-circle position errors at Aug 27 06Z"
        ),
        "landfall_error_km": {
            "members": landfall_errors.tolist(),
            "mean": final_mean,
            "median": float(np.median(landfall_errors)),
            "standard_deviation": float(np.std(landfall_errors, ddof=1)),
            "standard_error": float(
                np.std(landfall_errors, ddof=1) / np.sqrt(len(landfall_errors))
            ),
            "bootstrap_95pct_interval": [
                float(np.quantile(final_bootstrap, 0.025)),
                float(np.quantile(final_bootstrap, 0.975)),
            ],
            "original_k8_mean": first_batch_mean,
            "cumulative_mean_seed_order": running_mean.tolist(),
            "bootstrap_running_low": bootstrap_low.tolist(),
            "bootstrap_running_high": bootstrap_high.tolist(),
        },
        "track_error_km": {
            "members": track_errors.tolist(),
            "mean": float(np.mean(track_errors)),
            "median": float(np.median(track_errors)),
        },
        "pairwise_track_distance_km": {
            "pair_count": int(len(pairwise)),
            "mean": float(np.mean(pairwise)),
            "median": float(np.median(pairwise)),
        },
        "posterior_track_density": {
            "definition": (
                "Gaussian-kernel density in trajectory space; pairwise distance "
                "is mean great-circle separation over forecast frames through "
                "the fixed landfall time. This is relative empirical posterior "
                "support, not a normalized per-member probability from the sampler."
            ),
            "bandwidth_km": track_density_bandwidth,
            "bandwidth_mode": (
                "median pairwise track distance"
                if args.track_density_bandwidth_km <= 0.0
                else "user supplied"
            ),
            "raw_support": track_density.tolist(),
            "support_percentile": track_density_percentile.tolist(),
            "pairwise_distance_matrix_km": track_distance_matrix.tolist(),
            "color_mapping": (
                "coolwarm_r: low-support isolated/extreme tracks are red; "
                "high-support typical tracks are blue"
            ),
        },
        "member_provenance": member_provenance,
        "figure_outputs": [str(path) for path in outputs],
    }
    output_json = (
        args.output_dir
        / f"oracle_ensemble{args.target_members}_density_metrics.json"
    )
    output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "members": len(tracks),
        "landfall_error_mean_km": final_mean,
        "landfall_error_bootstrap_95pct": metrics["landfall_error_km"]["bootstrap_95pct_interval"],
        "track_error_mean_km": metrics["track_error_km"]["mean"],
        "pairwise_track_distance_mean_km": metrics["pairwise_track_distance_km"]["mean"],
        "figure": str(outputs[0]),
        "metrics": str(output_json),
    }, indent=2))


if __name__ == "__main__":
    main()
