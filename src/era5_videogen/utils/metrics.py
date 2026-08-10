"""Latitude-weighted (area-weighted) RMSE, following the WeatherBench / WeatherBench-X
convention: each grid cell is weighted by cos(latitude), normalized so the mean weight
over the latitude grid is 1. This corrects the equirectangular grid's pole over-sampling.

Ref: https://github.com/google-research/weatherbenchX/tree/main/weatherbenchX/metrics
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_SOURCE_ROOT = os.environ.get("ERA5_DATA_ROOT", "data/era5")

_CACHE: dict[str, np.ndarray] = {}


def latitude_weights(lat_deg) -> np.ndarray:
    """cos-latitude weights, normalized to mean 1. lat_deg: (H,) in degrees -> (H,)."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))
    w = np.clip(w, 0.0, None)
    return w / w.mean()


def load_lat_weights(source_root: str | None = None) -> np.ndarray:
    """Load lat.npy from the source ERA5 root and return normalized cos-lat weights (H,)."""
    root = source_root or os.environ.get("ERA5_DATA_ROOT", DEFAULT_SOURCE_ROOT)
    if root not in _CACHE:
        lat = np.load(os.path.join(root, "lat.npy"))
        _CACHE[root] = latitude_weights(lat)
    return _CACHE[root]


def weighted_rmse_ct(pred, truth, lat_weights=None, source_root: str | None = None):
    """Latitude-weighted RMSE over the spatial (H, W) axes.

    pred/truth: arrays shaped (..., H, W) (e.g. (C, T, H, W)); returns the
    weighted RMSE over the trailing (H, W), i.e. shape (...,) (e.g. (C, T)).
    """
    w = lat_weights if lat_weights is not None else load_lat_weights(source_root)
    w = np.asarray(w, dtype=np.float64)
    err2 = (np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2
    # weights depend only on latitude (axis -2); broadcast over longitude (axis -1).
    wmse = (err2 * w.reshape(-1, 1)).mean(axis=(-2, -1))
    return np.sqrt(wmse)
