#!/usr/bin/env python3
"""Compare frame-only and spatial-complement Laura repulsion forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from laura_forecast_repulsion_figs import (
    LANDFALL_TIME,
    SOURCE_INDEX,
    START,
    field_view,
    load_run,
    lon_for_plot,
    make_map,
    members_view,
    save_figure,
    track_metrics,
)
from laura_hindcast_repulsion_fig import (
    Tracker,
    load_land_mask,
    load_observation_locations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frame-run", type=Path, required=True)
    parser.add_argument("--spatial-run", type=Path, required=True)
    parser.add_argument("--obs-days", type=int, choices=[2, 3, 4], default=4)
    parser.add_argument("--source-index", type=int, default=SOURCE_INDEX)
    parser.add_argument("--observations-pkl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def relative_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def main() -> None:
    args = parse_args()
    runs = {
        "frame": load_run(args.frame_run, args.obs_days, args.source_index),
        "spatial": load_run(args.spatial_run, args.obs_days, args.source_index),
    }
    frame_mode = runs["frame"]["metadata"].get("repulsion_mask_mode", "frames")
    spatial_mode = runs["spatial"]["metadata"].get(
        "repulsion_mask_mode", "frames"
    )
    if frame_mode != "frames":
        raise ValueError(f"frame baseline declares repulsion_mask_mode={frame_mode!r}")
    if spatial_mode != "spatial":
        raise ValueError(
            f"spatial candidate declares repulsion_mask_mode={spatial_mode!r}"
        )

    variables = list(runs["frame"]["metadata"]["variables"])
    if list(runs["spatial"]["metadata"]["variables"]) != variables:
        raise ValueError("mask runs do not share the same variable order")
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
        lat,
        lon,
        variables.index("mean_sea_level_pressure"),
        variables.index("10m_u_component_of_wind"),
        variables.index("10m_v_component_of_wind"),
    )
    truth = field_view(runs["frame"]["truth_path"])
    spatial_truth = field_view(runs["spatial"]["truth_path"])
    if not np.allclose(truth, spatial_truth, equal_nan=True):
        raise ValueError("frame and spatial runs do not share identical ERA5 truth")

    initial_truth_track = tracker.track(truth)
    seed = (initial_truth_track["lat"][0], initial_truth_track["lon"][0])
    truth_track = tracker.track(truth, seed=seed, seed_frame=0)
    landfall_frame = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    plot_stop = min(landfall_frame + 5, truth.shape[1])

    tracks = {}
    metrics = {}
    for label, run in runs.items():
        samples = members_view(run["sample_path"])
        tracks[label] = [
            tracker.track(samples[index], seed=seed, seed_frame=0)
            for index in range(samples.shape[0])
        ]
        metrics[label] = track_metrics(
            tracks[label],
            truth_track,
            args.obs_days * 4,
            landfall_frame,
        )

    obs_path = args.observations_pkl
    if obs_path is None:
        raw = bundle / "data" / "observations" / "global_joint_2020_year.pkl"
        superob = (
            bundle
            / "data"
            / "observations"
            / "global_joint_2020_year_superob.pkl"
        )
        obs_path = raw if raw.is_file() else superob
    obs_points = load_observation_locations(obs_path, args.source_index)
    land_mask = load_land_mask(bundle, lat, lon)

    fig, ax = make_map(lat, lon, land_mask)
    if obs_points is not None:
        ax.scatter(
            lon_for_plot(obs_points[:, 1]),
            obs_points[:, 0],
            s=5,
            color="0.55",
            alpha=0.45,
            zorder=2,
            label="in-situ observation sites",
        )
    styles = {
        "frame": ("#377eb8", "Frame-only repulsion mask"),
        "spatial": ("#e41a1c", "Spatial observation-complement mask"),
    }
    for label in ("frame", "spatial"):
        color, legend = styles[label]
        for index, track in enumerate(tracks[label]):
            ax.plot(
                lon_for_plot(track["lon"][:plot_stop]),
                track["lat"][:plot_stop],
                color=color,
                linewidth=1.1,
                alpha=0.58,
                label=legend if index == 0 else None,
                zorder=3 if label == "frame" else 4,
            )
    ax.plot(
        lon_for_plot(truth_track["lon"][:plot_stop]),
        truth_track["lat"][:plot_stop],
        color="black",
        linewidth=3.0,
        zorder=8,
        label="ERA5 reference",
    )
    ax.scatter(
        [lon_for_plot(truth_track["lon"][landfall_frame])],
        [truth_track["lat"][landfall_frame]],
        marker="*",
        s=310,
        color="gold",
        edgecolor="black",
        zorder=10,
        label="landfall (Aug 27 06Z)",
    )
    ax.legend(loc="upper right", fontsize=8.2, framealpha=0.95)
    ax.set_title(
        "Hurricane Laura repulsion-mask comparison "
        f"({args.obs_days} days observed)"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_outputs = save_figure(
        fig,
        args.output_dir,
        f"repulsion_mask_comparison_obs{args.obs_days}d",
    )

    comparison = {
        "source_index": args.source_index,
        "observed_days": args.obs_days,
        "frame_run": str(runs["frame"]["path"]),
        "spatial_run": str(runs["spatial"]["path"]),
        "frame_metrics": metrics["frame"],
        "spatial_metrics": metrics["spatial"],
        "spatial_vs_frame_percent": {
            "mean_track_error": relative_change(
                metrics["spatial"]["track_error_km_mean"],
                metrics["frame"]["track_error_km_mean"],
            ),
            "mean_landfall_error": relative_change(
                metrics["spatial"]["landfall_error_km_mean"],
                metrics["frame"]["landfall_error_km_mean"],
            ),
            "mean_pairwise_track_distance": relative_change(
                metrics["spatial"]["pairwise_track_distance_km_mean"],
                metrics["frame"]["pairwise_track_distance_km_mean"],
            ),
        },
        "figure_outputs": [str(path) for path in figure_outputs],
    }
    output_json = (
        args.output_dir
        / f"repulsion_mask_comparison_obs{args.obs_days}d.json"
    )
    output_json.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    print(f"wrote comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
