#!/usr/bin/env python3
"""Retrospectively select Laura observation sites with an oracle track target.

This is deliberately an *oracle diagnostic*: the future ERA5 track is used in
the outer site-selection objective. It is never inserted into the posterior
measurement. The selected first-two-day in-situ observations are written as a
manifest that the normal posterior sampler can apply without seeing ERA5.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from laura_forecast_repulsion_figs import (
    LANDFALL_TIME,
    SOURCE_INDEX,
    START,
    field_view,
    load_run,
    members_view,
)
from laura_hindcast_repulsion_fig import Tracker, gc_km
from era5_videogen.observation_selection import signed_longitude, site_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--observations-pkl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=SOURCE_INDEX)
    parser.add_argument("--obs-days", type=int, choices=[2], default=2)
    parser.add_argument(
        "--selection-mode",
        choices=["oracle", "random"],
        default="oracle",
        help="Oracle greedy selection or a uniform random-site control.",
    )
    parser.add_argument("--random-seed", type=int, default=20261111)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument(
        "--selection-region",
        type=float,
        nargs=4,
        default=(15.0, 35.0, -100.0, -65.0),
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
    )
    parser.add_argument("--site-round-decimals", type=int, default=3)
    parser.add_argument(
        "--min-site-records",
        type=int,
        default=4,
        help="Require this many channel/time records at a candidate site.",
    )
    parser.add_argument("--min-site-separation-deg", type=float, default=0.5)
    parser.add_argument("--target-spread-ratio", type=float, default=0.75)
    parser.add_argument("--spread-penalty", type=float, default=4.0)
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--landfall-weight", type=float, default=0.5)
    parser.add_argument(
        "--keep-outside-region",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all observations outside the optimized Laura region.",
    )
    args = parser.parse_args()
    lat_min, lat_max, lon_min, lon_max = args.selection_region
    if lat_min >= lat_max or lon_min >= lon_max:
        parser.error("--selection-region bounds must be increasing")
    if args.budget < 1:
        parser.error("--budget must be positive")
    if args.site_round_decimals < 0:
        parser.error("--site-round-decimals must be non-negative")
    if args.min_site_records < 1:
        parser.error("--min-site-records must be positive")
    if args.min_site_separation_deg < 0:
        parser.error("--min-site-separation-deg must be non-negative")
    if not 0 < args.target_spread_ratio <= 1:
        parser.error("--target-spread-ratio must be in (0, 1]")
    if args.spread_penalty < 0 or args.weight_temperature <= 0:
        parser.error("spread penalty must be non-negative and temperature positive")
    return args


def fractional_latitude_indices(axis: np.ndarray, values: np.ndarray):
    axis = np.asarray(axis, dtype=np.float64).squeeze()
    values = np.asarray(values, dtype=np.float64)
    if axis[0] <= axis[-1]:
        return np.interp(values, axis, np.arange(len(axis), dtype=np.float64))
    reversed_index = np.interp(
        values, axis[::-1], np.arange(len(axis), dtype=np.float64)
    )
    return (len(axis) - 1) - reversed_index


def bilinear_members(
    fields: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Interpolate ``(K,H,W)`` physical fields at [lat,lon] queries."""
    fields = np.asarray(fields, dtype=np.float32)
    q = np.asarray(queries, dtype=np.float64).reshape(-1, 2)
    y = fractional_latitude_indices(latitude, q[:, 0])
    y = np.clip(y, 0.0, fields.shape[-2] - 1.0)
    y0 = np.floor(y).astype(np.int64)
    y1 = np.minimum(y0 + 1, fields.shape[-2] - 1)
    wy = y - y0

    lon_axis = np.asarray(longitude, dtype=np.float64).squeeze() % 360.0
    step = float(np.median(np.diff(lon_axis)))
    if step <= 0:
        raise ValueError("longitude grid must be regularly increasing")
    x = ((q[:, 1] % 360.0 - lon_axis[0]) % 360.0) / step
    x0 = np.floor(x).astype(np.int64) % fields.shape[-1]
    x1 = (x0 + 1) % fields.shape[-1]
    wx = x - np.floor(x)

    f00 = fields[:, y0, x0]
    f01 = fields[:, y0, x1]
    f10 = fields[:, y1, x0]
    f11 = fields[:, y1, x1]
    top = f00 * (1.0 - wx)[None] + f01 * wx[None]
    bottom = f10 * (1.0 - wx)[None] + f11 * wx[None]
    return top * (1.0 - wy)[None] + bottom * wy[None]


def track_objective(samples, truth, tracker, first_forecast, landfall, landfall_weight):
    truth_initial = tracker.track(truth)
    seed = (truth_initial["lat"][0], truth_initial["lon"][0])
    truth_track = tracker.track(truth, seed=seed, seed_frame=0)
    tracks = [
        tracker.track(samples[index], seed=seed, seed_frame=0)
        for index in range(samples.shape[0])
    ]
    frames = np.arange(first_forecast, landfall + 1)
    track_errors, landfall_errors = [], []
    for track in tracks:
        error = gc_km(
            track["lat"][frames],
            track["lon"][frames],
            truth_track["lat"][frames],
            truth_track["lon"][frames],
        )
        track_errors.append(float(np.mean(error)))
        landfall_errors.append(float(error[-1]))
    objective = np.asarray(track_errors) + landfall_weight * np.asarray(landfall_errors)

    pairwise = np.zeros((len(tracks), len(tracks)), dtype=np.float64)
    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            distance = float(
                np.mean(
                    gc_km(
                        tracks[left]["lat"][frames],
                        tracks[left]["lon"][frames],
                        tracks[right]["lat"][frames],
                        tracks[right]["lon"][frames],
                    )
                )
            )
            pairwise[left, right] = pairwise[right, left] = distance
    return truth_track, tracks, np.asarray(track_errors), np.asarray(landfall_errors), objective, pairwise


def normalized_site_misfits(
    samples,
    metadata,
    observations,
    source_index,
    obs_frames,
    latitude,
    longitude,
    means,
    stds,
    region,
    decimals,
):
    variables = list(metadata["variables"])
    obs_variables = list(metadata["obs_variables"])
    lat_min, lat_max, lon_min, lon_max = region
    site_sse: dict[str, np.ndarray] = {}
    site_count: dict[str, int] = {}
    site_coords: dict[str, tuple[float, float]] = {}
    site_records: dict[str, int] = {}

    for frame in obs_frames:
        queries_frame = observations[source_index + frame][0][0]
        values_frame = observations[source_index + frame][1][0]
        for obs_channel, variable in enumerate(obs_variables):
            q = np.asarray(queries_frame[obs_channel], dtype=np.float64).reshape(-1, 2)
            values = np.asarray(values_frame[obs_channel], dtype=np.float64).reshape(-1)
            if len(q) != len(values):
                raise ValueError(f"query/value mismatch for {variable} at frame {frame}")
            if not len(q):
                continue
            lon = signed_longitude(q[:, 1])
            keep = (
                np.isfinite(values)
                & np.isfinite(q).all(axis=1)
                & (q[:, 0] >= lat_min)
                & (q[:, 0] <= lat_max)
                & (lon >= lon_min)
                & (lon <= lon_max)
            )
            if not np.any(keep):
                continue
            q = q[keep]
            q[:, 1] = lon[keep]
            values = values[keep]
            channel = variables.index(variable)
            predicted_phys = bilinear_members(
                samples[:, channel, frame], latitude, longitude, q
            )
            mean = float(np.asarray(means[variable]).squeeze())
            std = float(np.asarray(stds[variable]).squeeze())
            predicted = (predicted_phys - mean) / std
            residual2 = np.square(predicted - values[None])
            keys = [site_key(lat, xlon, decimals) for lat, xlon in q]
            groups: dict[str, list[int]] = {}
            for index, key in enumerate(keys):
                groups.setdefault(key, []).append(index)
            for key, indices in groups.items():
                indices = np.asarray(indices, dtype=np.int64)
                if key not in site_sse:
                    site_sse[key] = np.zeros(samples.shape[0], dtype=np.float64)
                    site_count[key] = 0
                    site_records[key] = 0
                    first = int(indices[0])
                    site_coords[key] = (
                        round(float(q[first, 0]), decimals),
                        round(float(q[first, 1]), decimals),
                    )
                site_sse[key] += residual2[:, indices].sum(axis=1)
                site_count[key] += int(len(indices))
                site_records[key] += int(len(indices))

    keys = sorted(site_sse)
    if not keys:
        raise RuntimeError("no candidate in-situ sites were found in the selection region")
    raw = np.stack([site_sse[key] / max(site_count[key], 1) for key in keys])
    center = np.median(raw, axis=1, keepdims=True)
    scale = np.std(raw, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    normalized = (raw - center) / scale
    coords = np.asarray([site_coords[key] for key in keys], dtype=np.float64)
    records = np.asarray([site_records[key] for key in keys], dtype=np.int64)
    return keys, coords, records, normalized


def member_weights(misfit: np.ndarray, temperature: float) -> np.ndarray:
    logits = -np.asarray(misfit, dtype=np.float64) / temperature
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def weighted_spread(weights: np.ndarray, pairwise: np.ndarray) -> np.ndarray:
    numerator = np.einsum("...i,ij,...j->...", weights, pairwise, weights)
    denominator = np.maximum(1.0 - np.sum(weights * weights, axis=-1), 1e-8)
    return numerator / denominator


def greedy_select(
    keys,
    coords,
    misfits,
    member_objective,
    pairwise,
    budget,
    min_separation,
    target_spread_ratio,
    spread_penalty,
    temperature,
):
    members = len(member_objective)
    baseline_error = float(np.mean(member_objective))
    baseline_spread = float(np.mean(pairwise[np.triu_indices(members, 1)]))
    cumulative = np.zeros(members, dtype=np.float64)
    eligible = np.ones(len(keys), dtype=bool)
    selected: list[int] = []
    trace = []

    for step in range(min(budget, len(keys))):
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            break
        candidate_mean = (cumulative[None] + misfits[candidates]) / (step + 1)
        weights = member_weights(candidate_mean, temperature)
        error = weights @ member_objective
        spread = weighted_spread(weights, pairwise)
        spread_ratio = spread / max(baseline_spread, 1e-8)
        shortfall = np.maximum(target_spread_ratio - spread_ratio, 0.0)
        score = error / max(baseline_error, 1e-8) + spread_penalty * shortfall**2
        local = int(np.argmin(score))
        chosen = int(candidates[local])
        selected.append(chosen)
        cumulative += misfits[chosen]
        trace.append(
            {
                "step": step + 1,
                "site_key": keys[chosen],
                "surrogate_track_objective_km": float(error[local]),
                "surrogate_pairwise_spread_km": float(spread[local]),
                "spread_ratio": float(spread_ratio[local]),
                "objective": float(score[local]),
            }
        )
        eligible[chosen] = False
        if min_separation > 0:
            lat0, lon0 = coords[chosen]
            distance = np.sqrt(
                np.square(coords[:, 0] - lat0)
                + np.square(
                    np.cos(np.deg2rad(lat0)) * (coords[:, 1] - lon0)
                )
            )
            eligible[distance < min_separation] = False

    final_misfit = cumulative / max(len(selected), 1)
    final_weights = member_weights(final_misfit[None], temperature)[0]
    diagnostics = {
        "baseline_unweighted_track_objective_km": baseline_error,
        "baseline_unweighted_pairwise_spread_km": baseline_spread,
        "surrogate_weighted_track_objective_km": float(final_weights @ member_objective),
        "surrogate_weighted_pairwise_spread_km": float(
            weighted_spread(final_weights[None], pairwise)[0]
        ),
        "member_weights": final_weights.tolist(),
        "greedy_trace": trace,
    }
    return selected, diagnostics


def main() -> None:
    args = parse_args()
    bundle = args.bundle.expanduser().resolve()
    run = load_run(args.baseline_run, args.obs_days, args.source_index)
    samples = np.asarray(members_view(run["sample_path"]), dtype=np.float32)
    truth = np.asarray(field_view(run["truth_path"]), dtype=np.float32)
    metadata = run["metadata"]
    variables = list(metadata["variables"])

    latitude = np.load(bundle / "data" / "source_root" / "lat.npy")
    longitude = np.load(bundle / "data" / "source_root" / "lon.npy")
    tracker = Tracker(
        latitude,
        longitude,
        variables.index("mean_sea_level_pressure"),
        variables.index("10m_u_component_of_wind"),
        variables.index("10m_v_component_of_wind"),
    )
    first_forecast = args.obs_days * 4
    landfall = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    _, _, track_errors, landfall_errors, member_objective, pairwise = track_objective(
        samples,
        truth,
        tracker,
        first_forecast,
        landfall,
        args.landfall_weight,
    )

    with args.observations_pkl.expanduser().resolve().open("rb") as handle:
        observations = pickle.load(handle)
    means = np.load(bundle / "data" / "source_root" / "normalize_mean.npz")
    stds = np.load(bundle / "data" / "source_root" / "normalize_std.npz")
    obs_frames = list(range(first_forecast))
    all_keys, all_coords, all_records, all_misfits = normalized_site_misfits(
        samples,
        metadata,
        observations,
        args.source_index,
        obs_frames,
        latitude,
        longitude,
        means,
        stds,
        args.selection_region,
        args.site_round_decimals,
    )
    eligible = all_records >= args.min_site_records
    keys = [key for key, keep in zip(all_keys, eligible, strict=True) if keep]
    coords = all_coords[eligible]
    records = all_records[eligible]
    misfits = all_misfits[eligible]
    if not keys:
        raise RuntimeError(
            "no candidate sites satisfy "
            f"--min-site-records={args.min_site_records}"
        )
    if args.selection_mode == "oracle":
        selected_indices, diagnostics = greedy_select(
            keys,
            coords,
            misfits,
            member_objective,
            pairwise,
            args.budget,
            args.min_site_separation_deg,
            args.target_spread_ratio,
            args.spread_penalty,
            args.weight_temperature,
        )
        selection_mode = "retrospective_oracle_track_surrogate"
        oracle_target = True
        warning = (
            "Future ERA5 track enters only the outer site-selection score. "
            "This is a retrospective diagnostic, not an operational forecast."
        )
    else:
        rng = np.random.default_rng(args.random_seed)
        count = min(args.budget, len(keys))
        selected_indices = sorted(
            int(index) for index in rng.choice(len(keys), size=count, replace=False)
        )
        diagnostics = {
            "random_seed": args.random_seed,
            "selection_probability": count / len(keys),
            "baseline_unweighted_track_objective_km": float(
                np.mean(member_objective)
            ),
            "baseline_unweighted_pairwise_spread_km": float(
                np.mean(pairwise[np.triu_indices(len(member_objective), 1)])
            ),
        }
        selection_mode = "uniform_random_site_control"
        oracle_target = False
        warning = (
            "Regional sites were selected uniformly at random. Future ERA5 "
            "was used only for evaluation, never for site selection."
        )
    selected_sites = [
        {
            "key": keys[index],
            "latitude": float(coords[index, 0]),
            "longitude": float(coords[index, 1]),
            "records": int(records[index]),
        }
        for index in selected_indices
    ]
    selected_keys = {site["key"] for site in selected_sites}
    dropped_sites = [
        {
            "key": all_keys[index],
            "latitude": float(all_coords[index, 0]),
            "longitude": float(all_coords[index, 1]),
            "records": int(all_records[index]),
        }
        for index in range(len(all_keys))
        if all_keys[index] not in selected_keys
    ]
    manifest = {
        "schema_version": 1,
        "selection_mode": selection_mode,
        "oracle_target_uses_future_era5": oracle_target,
        "operational_forecast": False,
        "warning": warning,
        "random_seed": args.random_seed if args.selection_mode == "random" else None,
        "source_index": args.source_index,
        "observed_days": args.obs_days,
        "observed_frames": obs_frames,
        "selection_region": {
            "latitude": list(map(float, args.selection_region[:2])),
            "longitude": list(map(float, args.selection_region[2:])),
        },
        "keep_outside_region": args.keep_outside_region,
        "site_round_decimals": args.site_round_decimals,
        "min_site_records": args.min_site_records,
        "candidate_site_count": len(all_keys),
        "eligible_candidate_site_count": len(keys),
        "requested_budget": args.budget,
        "selected_site_count": len(selected_sites),
        "min_site_separation_deg": args.min_site_separation_deg,
        "target_spread_ratio": args.target_spread_ratio,
        "spread_penalty": args.spread_penalty,
        "weight_temperature": args.weight_temperature,
        "landfall_weight": args.landfall_weight,
        "selected_site_keys": [site["key"] for site in selected_sites],
        "selected_sites": selected_sites,
        "dropped_sites": dropped_sites,
        "baseline_member_track_error_km": track_errors.tolist(),
        "baseline_member_landfall_error_km": landfall_errors.tolist(),
        "surrogate_diagnostics": diagnostics,
        "baseline_run": str(run["path"]),
        "observations_pkl": str(args.observations_pkl.expanduser().resolve()),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"selected {len(selected_sites)} of {len(keys)} eligible "
        f"({len(all_keys)} total regional) sites; "
        f"manifest: {args.output_manifest}"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
