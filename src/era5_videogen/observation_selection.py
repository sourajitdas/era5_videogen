"""Utilities for applying interpretable site-selection masks to in-situ data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def signed_longitude(value):
    """Return longitude(s) in the half-open interval [-180, 180)."""
    array = np.asarray(value, dtype=np.float64)
    return (array + 180.0) % 360.0 - 180.0


def site_key(latitude: float, longitude: float, decimals: int) -> str:
    """Stable key shared by the selector and the sampler-side filter."""
    lat = round(float(latitude), decimals)
    lon = round(float(signed_longitude(longitude)), decimals)
    return f"{lat:.{decimals}f},{lon:.{decimals}f}"


def load_selection_manifest(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported observation-selection manifest: {path}")
    manifest["_path"] = str(path)
    return manifest


def filter_igra_measurement(
    measurement,
    manifest: dict,
    *,
    source_index: int,
    obs_frames: list[int],
):
    """Apply a site mask to ``[query_frames, value_frames]`` observations.

    The manifest selects rounded station/site coordinates inside a declared
    geographic region. Observations outside that region are retained by
    default, preserving the large-scale context while thinning the Laura
    corridor. The operation never reads ERA5 or any future target.
    """
    expected_source = int(manifest.get("source_index", source_index))
    if expected_source != int(source_index):
        raise ValueError(
            f"selection manifest is for source {expected_source}, not {source_index}"
        )
    expected_frames = [int(x) for x in manifest.get("observed_frames", obs_frames)]
    if expected_frames != [int(x) for x in obs_frames]:
        raise ValueError(
            f"selection manifest frames {expected_frames} do not match {obs_frames}"
        )

    decimals = int(manifest["site_round_decimals"])
    region = manifest["selection_region"]
    lat_min, lat_max = map(float, region["latitude"])
    lon_min, lon_max = map(float, region["longitude"])
    keep_outside = bool(manifest.get("keep_outside_region", True))
    selected = set(manifest.get("selected_site_keys", []))
    if not selected:
        raise ValueError("selection manifest contains no selected sites")

    q_frames, v_frames = measurement
    filtered_q_frames, filtered_v_frames = [], []
    frame_counts = []
    selected_records = 0
    dropped_records = 0
    outside_records = 0

    for q_frame, v_frame in zip(q_frames, v_frames, strict=True):
        filtered_q, filtered_v = [], []
        frame_count = 0
        for queries, values in zip(q_frame, v_frame, strict=True):
            queries = np.asarray(queries)
            values = np.asarray(values)
            if queries.reshape(-1, 2).shape[0] != values.reshape(-1).shape[0]:
                raise ValueError("query/value lengths differ in in-situ measurement")
            original_shape = queries.shape
            q = queries.reshape(-1, 2)
            v = values.reshape(-1)
            lon = signed_longitude(q[:, 1])
            inside = (
                (q[:, 0] >= lat_min)
                & (q[:, 0] <= lat_max)
                & (lon >= lon_min)
                & (lon <= lon_max)
            )
            chosen = np.fromiter(
                (
                    site_key(lat, xlon, decimals) in selected
                    for lat, xlon in zip(q[:, 0], lon, strict=True)
                ),
                dtype=bool,
                count=len(q),
            )
            keep = chosen | (~inside if keep_outside else np.zeros_like(inside))
            selected_records += int(np.count_nonzero(chosen & inside))
            dropped_records += int(np.count_nonzero(inside & ~chosen))
            outside_records += int(np.count_nonzero(~inside & keep))
            q_keep = q[keep]
            v_keep = v[keep]
            filtered_q.append(q_keep.reshape((-1, original_shape[-1])))
            filtered_v.append(v_keep)
            frame_count += int(v_keep.size)
        filtered_q_frames.append(filtered_q)
        filtered_v_frames.append(filtered_v)
        frame_counts.append(frame_count)

    stats = {
        "selected_records_in_region": selected_records,
        "dropped_records_in_region": dropped_records,
        "retained_records_outside_region": outside_records,
        "retained_records_per_frame": frame_counts,
    }
    return [filtered_q_frames, filtered_v_frames], stats
