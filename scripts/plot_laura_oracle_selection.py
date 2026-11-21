#!/usr/bin/env python3
"""Plot and score the retrospective Laura oracle site-selection experiment."""

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
from laura_hindcast_repulsion_fig import Tracker, load_land_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=SOURCE_INDEX)
    parser.add_argument("--obs-days", type=int, choices=[2], default=2)
    return parser.parse_args()


def relative_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def main() -> None:
    args = parse_args()
    bundle = args.bundle.expanduser().resolve()
    manifest = json.loads(
        args.selection_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    selection_mode = manifest.get("selection_mode", "unknown")
    if selection_mode not in {
        "retrospective_oracle_track_surrogate",
        "uniform_random_site_control",
    }:
        raise ValueError(f"unsupported selection mode: {selection_mode}")
    baseline = load_run(args.baseline_run, args.obs_days, args.source_index)
    selected = load_run(args.selected_run, args.obs_days, args.source_index)
    if not selected["metadata"].get("observation_selection"):
        raise ValueError("selected run metadata does not contain a site selection")
    variables = list(baseline["metadata"]["variables"])
    if list(selected["metadata"]["variables"]) != variables:
        raise ValueError("baseline and selected runs use different variables")

    truth = np.asarray(field_view(baseline["truth_path"]), dtype=np.float32)
    selected_truth = np.asarray(field_view(selected["truth_path"]), dtype=np.float32)
    if not np.allclose(truth, selected_truth, equal_nan=True):
        raise ValueError("baseline and selected runs do not share ERA5 truth")
    latitude = np.load(bundle / "data" / "source_root" / "lat.npy")
    longitude = np.load(bundle / "data" / "source_root" / "lon.npy")
    tracker = Tracker(
        latitude,
        longitude,
        variables.index("mean_sea_level_pressure"),
        variables.index("10m_u_component_of_wind"),
        variables.index("10m_v_component_of_wind"),
    )
    initial_truth = tracker.track(truth)
    seed = (initial_truth["lat"][0], initial_truth["lon"][0])
    truth_track = tracker.track(truth, seed=seed, seed_frame=0)
    tracks = {}
    for label, run in (("baseline", baseline), ("selected", selected)):
        samples = members_view(run["sample_path"])
        tracks[label] = [
            tracker.track(samples[index], seed=seed, seed_frame=0)
            for index in range(samples.shape[0])
        ]

    landfall_frame = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    plot_stop = min(landfall_frame + 5, truth.shape[1])
    first_forecast = args.obs_days * 4
    metrics = {
        label: track_metrics(
            member_tracks, truth_track, first_forecast, landfall_frame
        )
        for label, member_tracks in tracks.items()
    }

    selected_sites = np.asarray(
        [
            [site["latitude"], site["longitude"]]
            for site in manifest["selected_sites"]
        ],
        dtype=np.float64,
    )
    dropped_sites = np.asarray(
        [
            [site["latitude"], site["longitude"]]
            for site in manifest["dropped_sites"]
        ],
        dtype=np.float64,
    )
    land_mask = load_land_mask(bundle, latitude, longitude)
    fig, ax = make_map(latitude, longitude, land_mask)
    if len(dropped_sites):
        ax.scatter(
            lon_for_plot(dropped_sites[:, 1]),
            dropped_sites[:, 0],
            marker="x",
            s=18,
            linewidths=0.7,
            color="0.45",
            alpha=0.55,
            zorder=2,
            label=f"dropped regional sites (n={len(dropped_sites)})",
        )
    if len(selected_sites):
        ax.scatter(
            lon_for_plot(selected_sites[:, 1]),
            selected_sites[:, 0],
            marker="o",
            s=24,
            facecolor="#2ca25f",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
            zorder=5,
            label=(
                f"oracle-selected sites (n={len(selected_sites)})"
                if manifest.get("oracle_target_uses_future_era5", False)
                else f"randomly retained sites (n={len(selected_sites)})"
            ),
        )

    for index, track in enumerate(tracks["baseline"]):
        ax.plot(
            lon_for_plot(track["lon"][:plot_stop]),
            track["lat"][:plot_stop],
            color="#80b1d3",
            linewidth=0.8,
            alpha=0.3,
            linestyle="--",
            zorder=3,
            label="full-observation ensemble" if index == 0 else None,
        )
    for index, track in enumerate(tracks["selected"]):
        ax.plot(
            lon_for_plot(track["lon"][:plot_stop]),
            track["lat"][:plot_stop],
            color="#d7301f",
            linewidth=1.25,
            alpha=0.68,
            zorder=4,
            label="selected-site ensemble" if index == 0 else None,
        )
    ax.plot(
        lon_for_plot(truth_track["lon"][:plot_stop]),
        truth_track["lat"][:plot_stop],
        color="black",
        linewidth=3.0,
        zorder=8,
        label="ERA5 reference track",
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
    title_prefix = (
        "Retrospective oracle observation selection"
        if manifest.get("oracle_target_uses_future_era5", False)
        else "Random observation-drop control"
    )
    footer = (
        "Future ERA5 track used only to select sites; not an operational forecast"
        if manifest.get("oracle_target_uses_future_era5", False)
        else "Random site mask; future ERA5 used only for evaluation"
    )
    ax.set_title(f"{title_prefix} for Hurricane Laura ({args.obs_days} days observed)")
    ax.text(
        0.01,
        0.01,
        footer,
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.25",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        zorder=12,
    )
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.95)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, args.output_dir, "oracle_site_selection_tracks")

    comparison = {
        "experiment": selection_mode,
        "operational_forecast": False,
        "warning": manifest["warning"],
        "source_index": args.source_index,
        "observed_days": args.obs_days,
        "selected_site_count": int(manifest["selected_site_count"]),
        "candidate_site_count": int(manifest["candidate_site_count"]),
        "baseline_run": str(baseline["path"]),
        "selected_run": str(selected["path"]),
        "baseline_metrics": metrics["baseline"],
        "selected_metrics": metrics["selected"],
        "selected_vs_baseline_percent": {
            "mean_track_error": relative_change(
                metrics["selected"]["track_error_km_mean"],
                metrics["baseline"]["track_error_km_mean"],
            ),
            "mean_landfall_error": relative_change(
                metrics["selected"]["landfall_error_km_mean"],
                metrics["baseline"]["landfall_error_km_mean"],
            ),
            "mean_pairwise_track_distance": relative_change(
                metrics["selected"]["pairwise_track_distance_km_mean"],
                metrics["baseline"]["pairwise_track_distance_km_mean"],
            ),
        },
        "selection_manifest": str(args.selection_manifest.expanduser().resolve()),
        "figure_outputs": [str(path) for path in outputs],
    }
    output_json = args.output_dir / "oracle_site_selection_metrics.json"
    output_json.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
