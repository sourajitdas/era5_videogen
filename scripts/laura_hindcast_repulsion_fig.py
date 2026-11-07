#!/usr/bin/env python3
"""Compare an independent Laura control with a joint-repulsion ensemble."""

from __future__ import annotations

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OBS_FIRST = 16
OBS_FRAMES = list(range(16, 32))
START = np.datetime64("2020-08-21T00:00")
LANDFALL_TIME = np.datetime64("2020-08-27T06:00")
BOX_LAT = (6.0, 42.0)
BOX_LON = (250.0, 312.0)
TRACK_RADIUS = 4.5
SEED_RADIUS = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repulsion-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=932)
    parser.add_argument(
        "--observations-pkl",
        type=Path,
        default=None,
        help="Optional raw/super-ob pickle used to draw sites at frame 16.",
    )
    parser.add_argument("--plain-map", action="store_true")
    return parser.parse_args()


def members_from(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return np.asarray(array.reshape(-1, *array.shape[-4:]), dtype=np.float32)


def field_from(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    return np.asarray(array.reshape(-1, *array.shape[-4:])[0], dtype=np.float32)


def gc_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    hav = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        (lon2 - lon1) / 2
    ) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


class Tracker:
    def __init__(self, lat, lon, mslp_idx, u10_idx, v10_idx):
        self.lat = np.asarray(lat).squeeze()
        self.lon = np.asarray(lon).squeeze() % 360.0
        self.mslp_idx = mslp_idx
        self.u10_idx = u10_idx
        self.v10_idx = v10_idx
        self.iy = np.where(
            (self.lat >= BOX_LAT[0]) & (self.lat <= BOX_LAT[1])
        )[0]
        self.ix = np.where(
            (self.lon >= BOX_LON[0]) & (self.lon <= BOX_LON[1])
        )[0]
        self.blat = self.lat[self.iy]
        self.blon = self.lon[self.ix]
        if not len(self.iy) or not len(self.ix):
            raise ValueError("Laura tracking box is outside the supplied lat/lon grid")

    @staticmethod
    def _deg_dist(lat, lon, lat0, lon0):
        return np.sqrt(
            (lat - lat0) ** 2
            + (np.cos(np.deg2rad(lat0)) * (lon - lon0)) ** 2
        )

    def _argmin_within(self, pressure, lat0, lon0, radius):
        lat2 = np.broadcast_to(self.blat[:, None], pressure.shape)
        lon2 = np.broadcast_to(self.blon[None, :], pressure.shape)
        selected = self._deg_dist(lat2, lon2, lat0, lon0) <= radius
        masked = np.where(selected, pressure, np.inf)
        if not np.isfinite(masked).any():
            masked = pressure
        return np.unravel_index(np.argmin(masked), pressure.shape)

    def _refine(self, pressure, index):
        row0, row1 = max(index[0] - 2, 0), min(index[0] + 3, pressure.shape[0])
        col0, col1 = max(index[1] - 2, 0), min(index[1] + 3, pressure.shape[1])
        neighborhood = pressure[row0:row1, col0:col1]
        weight = np.maximum(neighborhood.max() - neighborhood, 0.0) ** 2
        if weight.sum() <= 0:
            return float(self.blat[index[0]]), float(self.blon[index[1]])
        lat = (weight * self.blat[row0:row1, None]).sum() / weight.sum()
        lon = (weight * self.blon[None, col0:col1]).sum() / weight.sum()
        return float(lat), float(lon)

    def track(self, video, seed=None, seed_frame=None):
        timesteps = video.shape[1]
        pressure = [
            video[self.mslp_idx, t][np.ix_(self.iy, self.ix)]
            for t in range(timesteps)
        ]
        wind = [
            np.hypot(video[self.u10_idx, t], video[self.v10_idx, t])[
                np.ix_(self.iy, self.ix)
            ]
            for t in range(timesteps)
        ]
        if seed_frame is None:
            seed_frame = min((p.min(), t) for t, p in enumerate(pressure))[1]
            index = np.unravel_index(
                np.argmin(pressure[seed_frame]), pressure[seed_frame].shape
            )
        else:
            index = self._argmin_within(
                pressure[seed_frame], seed[0], seed[1], SEED_RADIUS
            )
        positions = [None] * timesteps
        positions[seed_frame] = self._refine(pressure[seed_frame], index)
        for t in range(seed_frame + 1, timesteps):
            index = self._argmin_within(
                pressure[t], *positions[t - 1], TRACK_RADIUS
            )
            positions[t] = self._refine(pressure[t], index)
        for t in range(seed_frame - 1, -1, -1):
            index = self._argmin_within(
                pressure[t], *positions[t + 1], TRACK_RADIUS
            )
            positions[t] = self._refine(pressure[t], index)

        box_lat = np.broadcast_to(self.blat[:, None], pressure[0].shape)
        box_lon = np.broadcast_to(self.blon[None, :], pressure[0].shape)
        mslp, wmax = [], []
        for t, (track_lat, track_lon) in enumerate(positions):
            selected = (
                self._deg_dist(box_lat, box_lon, track_lat, track_lon)
                <= TRACK_RADIUS
            )
            mslp.append(float(np.where(selected, pressure[t], np.inf).min() / 100.0))
            wmax.append(float(np.where(selected, wind[t], -np.inf).max()))
        return {
            "lat": np.array([p[0] for p in positions]),
            "lon": np.array([p[1] for p in positions]),
            "mslp_hpa": np.asarray(mslp),
            "wmax_ms": np.asarray(wmax),
            "seed_frame": int(seed_frame),
        }


def load_observation_locations(path: Path | None, index: int):
    if path is None:
        return None
    print(f"loading observation locations from {path}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    points = set()
    for query in data[index][0][0]:
        for lat, lon in np.asarray(query).reshape(-1, 2):
            points.add((round(float(lat), 3), round(float(lon), 3)))
    if not points:
        return None
    return np.asarray(sorted(points), dtype=np.float32)


def ensemble_metrics(tracks, truth_track, inferred, observed, landfall):
    def member_errors(frames):
        return [
            float(
                gc_km(
                    track["lat"][frames],
                    track["lon"][frames],
                    truth_track["lat"][frames],
                    truth_track["lon"][frames],
                ).mean()
            )
            for track in tracks
        ]

    pairwise = []
    for left, right in combinations(tracks, 2):
        pairwise.append(
            gc_km(
                left["lat"][inferred],
                left["lon"][inferred],
                right["lat"][inferred],
                right["lon"][inferred],
            ).mean()
        )
    inferred_error = member_errors(inferred)
    observed_error = member_errors(observed)
    landfall_error = [
        float(
            gc_km(
                track["lat"][landfall],
                track["lon"][landfall],
                truth_track["lat"][landfall],
                truth_track["lon"][landfall],
            )
        )
        for track in tracks
    ]
    return {
        "members": len(tracks),
        "track_error_km_inferred_mean": float(np.mean(inferred_error)),
        "track_error_km_inferred_median": float(np.median(inferred_error)),
        "track_error_km_observed_mean": float(np.mean(observed_error)),
        "landfall_error_km_mean": float(np.mean(landfall_error)),
        "landfall_error_km_members": landfall_error,
        "pairwise_track_distance_km_inferred": float(np.mean(pairwise)),
        "min_mslp_hpa_members": [float(t["mslp_hpa"].min()) for t in tracks],
        "max_wind_ms_members": [float(t["wmax_ms"].max()) for t in tracks],
    }


def relative(new, old):
    return float(new / max(old, 1e-12) - 1.0)


def map_axes(plain):
    if not plain:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            projection = ccrs.PlateCarree()
            fig = plt.figure(figsize=(10.5, 6.6))
            ax = fig.add_axes([0.04, 0.05, 0.92, 0.89], projection=projection)
            ax.set_extent([-104, -54, 8, 40], crs=projection)
            ax.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
            return fig, ax, projection
        except ImportError:
            pass
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.set_xlim(-104, -54)
    ax.set_ylim(8, 40)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    return fig, ax, None


def plot_tracks(
    path,
    control,
    repulsion,
    truth,
    mean_tracks,
    obs_points,
    times,
    tend,
    plain,
    alpha,
):
    fig, ax, projection = map_axes(plain)
    transform = {"transform": projection} if projection is not None else {}
    if obs_points is not None:
        ax.scatter(
            obs_points[:, 1], obs_points[:, 0], s=5, color="0.55", alpha=0.35,
            zorder=1, label="in-situ super-ob sites (first observed frame)", **transform
        )
    for index, track in enumerate(control):
        ax.plot(
            track["lon"][:tend] - 360, track["lat"][:tend], color="#377eb8",
            linewidth=0.9, alpha=0.28, zorder=3,
            label="Independent-control members" if index == 0 else None,
            **transform,
        )
    for index, track in enumerate(repulsion):
        ax.plot(
            track["lon"][:tend] - 360, track["lat"][:tend], color="#e6550d",
            linewidth=0.9, alpha=0.32, zorder=4,
            label=(f"Joint-repulsion members (α={alpha:.2f})" if index == 0 else None),
            **transform,
        )
    for label, track, color in (
        ("Independent-control ensemble mean", mean_tracks["control"], "#08519c"),
        ("Joint-repulsion ensemble mean", mean_tracks["repulsion"], "#a63603"),
    ):
        ax.plot(
            track["lon"][:tend] - 360, track["lat"][:tend], color=color,
            linewidth=2.5, zorder=6, label=label, **transform
        )
    ax.plot(
        truth["lon"][:tend] - 360, truth["lat"][:tend], color="black",
        linewidth=3.0, zorder=7, label="ERA5", **transform
    )
    for frame in range(0, tend, 4):
        x, y = truth["lon"][frame] - 360, truth["lat"][frame]
        ax.scatter([x], [y], color="black", s=22, zorder=8, **transform)
        ax.annotate(str(times[frame])[5:10], (x + 0.45, y - 1.1), fontsize=8)
    landfall = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    ax.scatter(
        [truth["lon"][landfall] - 360], [truth["lat"][landfall]], marker="*",
        s=330, color="gold", edgecolor="black", zorder=9,
        label="landfall (Aug 27 06Z)", **transform
    )
    ax.scatter(
        [truth["lon"][OBS_FIRST] - 360], [truth["lat"][OBS_FIRST]], marker="s",
        s=65, facecolor="white", edgecolor="black", zorder=9,
        label="first observed frame", **transform
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.set_title(
        "Hurricane Laura posterior hindcast: independent control vs "
        f"joint repulsion (α={alpha:.2f})"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_track_error(path, control, repulsion, truth, times, tend, alpha):
    x = np.arange(tend)
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    for tracks, color, label in (
        (control, "#377eb8", "Independent control, K=8"),
        (repulsion, "#e6550d", f"Joint repulsion, K=8 (α={alpha:.2f})"),
    ):
        errors = np.stack(
            [gc_km(t["lat"][:tend], t["lon"][:tend], truth["lat"][:tend], truth["lon"][:tend]) for t in tracks]
        )
        ax.fill_between(x, errors.min(0), errors.max(0), color=color, alpha=0.16)
        ax.plot(x, np.median(errors, axis=0), color=color, linewidth=2.2, label=label)
    ax.axvspan(OBS_FIRST, tend - 1, color="0.85", alpha=0.6, label="observed frames")
    landfall = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    ax.axvline(landfall, color="goldenrod", linewidth=2, label="landfall")
    ax.set_xticks(x[::4], [str(t)[5:13].replace("T", " ") for t in times[:tend:4]], rotation=30)
    ax.set_ylabel("track error [km]")
    ax.set_xlabel("valid time [UTC]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_intensity(path, control, repulsion, truth, times, alpha):
    x = np.arange(len(times))
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    for ax, key, ylabel in (
        (axes[0], "mslp_hpa", "minimum MSLP [hPa]"),
        (axes[1], "wmax_ms", "maximum 10m wind [m/s]"),
    ):
        for tracks, color, label in (
            (control, "#377eb8", "Independent control, K=8"),
            (repulsion, "#e6550d", f"Joint repulsion, K=8 (α={alpha:.2f})"),
        ):
            values = np.stack([track[key] for track in tracks])
            ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=0.15)
            ax.plot(x, np.median(values, axis=0), color=color, linewidth=2.1, label=label)
        ax.plot(x, truth[key], color="black", linewidth=2.4, label="ERA5")
        ax.axvspan(OBS_FIRST, len(times) - 1, color="0.85", alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    axes[1].set_xticks(x[::4], [str(t)[5:13].replace("T", " ") for t in times[::4]], rotation=30)
    axes[1].set_xlabel("valid time [UTC]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    repulsion_run = args.repulsion_run.resolve()
    metadata = json.loads((repulsion_run / "metadata.json").read_text())
    if metadata.get("particle_coupling") != "global_all_particles":
        raise ValueError("repulsion metadata does not declare global particle coupling")
    if float(metadata.get("repulsion_strength", 0.0)) <= 0:
        raise ValueError("repulsion metadata has no active repulsion strength")
    if list(metadata.get("source_indices", [])) != [args.source_index]:
        raise ValueError("repulsion run does not match the requested Laura source index")

    control_paths = sorted(bundle.glob("control_k8/fc_seed*/sample_0932/samples.npy"))
    if len(control_paths) != 8:
        raise FileNotFoundError(f"expected 8 control samples, found {len(control_paths)}")
    control = np.concatenate([members_from(path) for path in control_paths], axis=0)
    repulsion = members_from(
        repulsion_run / f"sample_{args.source_index:04d}" / "samples.npy"
    )
    if control.shape[0] != 8 or repulsion.shape[0] != 8:
        raise ValueError(f"expected K=8 vs K=8, got {control.shape[0]} vs {repulsion.shape[0]}")
    truth = field_from(bundle / "control_k8" / "ground_truth.npy")
    if control.shape[1:] != repulsion.shape[1:] or control.shape[1:] != truth.shape:
        raise ValueError("control, repulsion, and truth shapes do not match")

    variables = list(metadata["variables"])
    required = [
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    missing = [name for name in required if name not in variables]
    if missing:
        raise ValueError(f"missing tracker variables: {missing}")
    lat = np.load(bundle / "data" / "source_root" / "lat.npy")
    lon = np.load(bundle / "data" / "source_root" / "lon.npy")
    tracker = Tracker(
        lat,
        lon,
        variables.index(required[0]),
        variables.index(required[1]),
        variables.index(required[2]),
    )
    initial_truth = tracker.track(truth)
    seed = (initial_truth["lat"][OBS_FIRST], initial_truth["lon"][OBS_FIRST])
    truth_track = tracker.track(truth, seed=seed, seed_frame=OBS_FIRST)
    control_tracks = [tracker.track(member, seed=seed, seed_frame=OBS_FIRST) for member in control]
    repulsion_tracks = [tracker.track(member, seed=seed, seed_frame=OBS_FIRST) for member in repulsion]
    mean_tracks = {
        "control": tracker.track(control.mean(0), seed=seed, seed_frame=OBS_FIRST),
        "repulsion": tracker.track(repulsion.mean(0), seed=seed, seed_frame=OBS_FIRST),
    }

    timesteps = truth.shape[1]
    times = START + np.arange(timesteps) * np.timedelta64(6, "h")
    landfall = int((LANDFALL_TIME - START) / np.timedelta64(6, "h"))
    tend = min(landfall + 3, timesteps)
    inferred = np.arange(2, OBS_FIRST)
    observed = np.arange(OBS_FIRST, tend)
    alpha = float(metadata["repulsion_strength"])
    repulsion_guidance = (
        "full-DPS" if metadata.get("guide_through_net", False) else "decoder-only"
    )
    control_metrics = ensemble_metrics(control_tracks, truth_track, inferred, observed, landfall)
    repulsion_metrics = ensemble_metrics(repulsion_tracks, truth_track, inferred, observed, landfall)
    result = {
        "configuration": {
            "source_index": args.source_index,
            "observed_frames": OBS_FRAMES,
            "members": 8,
            "control": "independent posterior samples",
            "control_guidance": "full-DPS",
            "repulsion": "joint-repulsion posterior samples",
            "repulsion_guidance": repulsion_guidance,
            "repulsion_strength": alpha,
            "repulsion_sigma_band": [metadata["repulsion_min_sigma"], metadata["repulsion_max_sigma"]],
            "repulsion_schedule": metadata["repulsion_schedule"],
            "repulsion_unobserved_only": metadata["repulsion_unobserved_only"],
            "repulsion_guide_through_net": metadata["guide_through_net"],
        },
        "control_k8": control_metrics,
        "repulsion_k8": repulsion_metrics,
        "relative_change": {
            "inferred_track_error": relative(repulsion_metrics["track_error_km_inferred_mean"], control_metrics["track_error_km_inferred_mean"]),
            "landfall_track_error": relative(repulsion_metrics["landfall_error_km_mean"], control_metrics["landfall_error_km_mean"]),
            "inferred_track_diversity": relative(repulsion_metrics["pairwise_track_distance_km_inferred"], control_metrics["pairwise_track_distance_km_inferred"]),
        },
        "interpretation_note": (
            "Positive track-diversity change means trajectories spread farther apart. "
            "Negative track-error change means improved trajectory skill."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs_path = args.observations_pkl
    if obs_path is None:
        candidate = bundle / "data" / "observations" / "global_joint_2020_year_superob.pkl"
        obs_path = candidate if candidate.exists() else None
    obs_points = load_observation_locations(obs_path, args.source_index + OBS_FIRST)
    plot_tracks(
        args.output_dir / "laura_track_comparison.png",
        control_tracks,
        repulsion_tracks,
        truth_track,
        mean_tracks,
        obs_points,
        times,
        tend,
        args.plain_map,
        alpha,
    )
    plot_track_error(
        args.output_dir / "laura_track_error.png",
        control_tracks,
        repulsion_tracks,
        truth_track,
        times,
        tend,
        alpha,
    )
    plot_intensity(
        args.output_dir / "laura_intensity_comparison.png",
        control_tracks,
        repulsion_tracks,
        truth_track,
        times,
        alpha,
    )
    (args.output_dir / "laura_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"wrote Laura comparison products to {args.output_dir}")


if __name__ == "__main__":
    main()
