#!/usr/bin/env python3
"""Compare independent, post-hoc-selected, and joint-repulsion K=4 ensembles.

The post-hoc selector receives eight independently generated candidates and
returns four.  Selection never reads ERA5 ground truth: it maximizes pairwise
distance over unobserved standardized trajectories, subject to an
observation-space RMSE constraint.  Ground truth is loaded only after the
member indices have been fixed and is used solely for evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from evaluate_baseline_ensemble import (
    DEFAULT_PLOT_VARIABLES,
    OBS_FRAMES,
    area_mean,
    mean_frame_area_rmse,
    resolve_variables,
    scalar_stat,
)
from evaluate_repulsion_smoke import baseline_members, evaluate
from evaluate_repulsion_smoke_k8 import smoke_checks


CANDIDATE_K = 8
RETURN_K = 4
DEFAULT_SEEDS = list(range(17, 25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-repulsion-run", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/sr/baseline_independent"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument(
        "--candidate-seeds", type=int, nargs="+", default=DEFAULT_SEEDS
    )
    parser.add_argument(
        "--independent-seeds", type=int, nargs="+", default=DEFAULT_SEEDS[:RETURN_K]
    )
    parser.add_argument(
        "--selection-spatial-pool",
        type=int,
        default=8,
        help="Block-average factor used only by the truth-free diversity selector.",
    )
    parser.add_argument(
        "--max-relative-observation-increase",
        type=float,
        default=0.02,
        help="Allowed selector observation-RMSE increase over ordinary K=4.",
    )
    parser.add_argument(
        "--max-absolute-observation-increase",
        type=float,
        default=0.001,
        help="Absolute observation-RMSE tolerance in standardized units.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_joint_run(run_dir: Path, window: int):
    metadata_path = run_dir / "metadata.json"
    sample_dir = run_dir / f"sample_{window:04d}"
    paths = {
        "metadata": metadata_path,
        "samples": sample_dir / "samples.npy",
        "truth": sample_dir / "ground_truth.npy",
        "observation": sample_dir / "observation_lowres.npy",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = np.load(paths["samples"], mmap_mode="r")
    if samples.ndim != 5 or samples.shape[0] != RETURN_K:
        raise ValueError(f"expected joint-repulsion K=4, found {samples.shape}")
    if float(metadata.get("repulsion_strength", 0.0)) <= 0.0:
        raise ValueError("joint run metadata has no active repulsion strength")
    return (
        metadata,
        samples,
        np.load(paths["truth"], mmap_mode="r"),
        np.load(paths["observation"], mmap_mode="r"),
    )


def pooled_fields(fields: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return fields
    height, width = fields.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(
            f"spatial shape {(height, width)} is not divisible by pool factor {factor}"
        )
    shape = fields.shape[:-2] + (
        height // factor,
        factor,
        width // factor,
        factor,
    )
    return fields.reshape(shape).mean(axis=(-3, -1), dtype=np.float64)


def observation_rmse(
    predicted: np.ndarray,
    measured: np.ndarray,
    indices: tuple[int, ...],
    latitude_weights: np.ndarray,
) -> float:
    ensemble_mean = np.mean(predicted[list(indices)], axis=0)
    per_channel = [
        float(mean_frame_area_rmse(ensemble_mean[channel] - measured[channel], latitude_weights))
        for channel in range(ensemble_mean.shape[0])
    ]
    return float(np.mean(per_channel))


def subset_diversity(distance_matrix: np.ndarray, indices: tuple[int, ...]) -> float:
    pairs = list(itertools.combinations(indices, 2))
    return float(np.mean([distance_matrix[left, right] for left, right in pairs]))


def truth_free_selection(
    arrays: list[np.ndarray],
    variables: list[str],
    observation: np.ndarray,
    means,
    stds,
    latitude_weights: np.ndarray,
    observation_latitude_weights: np.ndarray,
    independent_indices: tuple[int, ...],
    pool_factor: int,
    relative_observation_tolerance: float,
    absolute_observation_tolerance: float,
) -> tuple[tuple[int, ...], dict]:
    """Select members without accessing the truth array."""
    member_count = len(arrays)
    pairwise = np.zeros((member_count, member_count), dtype=np.float64)
    predicted_observations = []
    unobserved = np.ones(arrays[0].shape[2], dtype=bool)
    unobserved[list(OBS_FRAMES)] = False

    if pool_factor == 1:
        pooled_latitude_weights = latitude_weights
    else:
        if latitude_weights.size % pool_factor:
            raise ValueError("latitude count is not divisible by selection pool factor")
        pooled_latitude_weights = latitude_weights.reshape(-1, pool_factor).mean(axis=1)

    for channel, variable in enumerate(variables):
        values = np.stack(
            [np.asarray(array[0, channel], dtype=np.float32) for array in arrays],
            axis=0,
        )
        mean = scalar_stat(means, variable)
        std = scalar_stat(stds, variable)
        standardized = (values - mean) / std
        predicted_observations.append(
            standardized[:, list(OBS_FRAMES), ::8, ::8]
        )
        features = pooled_fields(standardized[:, unobserved], pool_factor)
        for left, right in itertools.combinations(range(member_count), 2):
            distance = float(
                area_mean(
                    np.square(features[left] - features[right]),
                    pooled_latitude_weights,
                )
            )
            pairwise[left, right] += distance
            pairwise[right, left] += distance

    pairwise /= len(variables)
    predicted = np.stack(predicted_observations, axis=1)
    measured = np.asarray(observation[0], dtype=np.float32)
    reference_observation_rmse = observation_rmse(
        predicted, measured, independent_indices, observation_latitude_weights
    )
    observation_limit = max(
        reference_observation_rmse * (1.0 + relative_observation_tolerance),
        reference_observation_rmse + absolute_observation_tolerance,
    )

    candidates = []
    for indices in itertools.combinations(range(member_count), RETURN_K):
        obs_rmse = observation_rmse(
            predicted, measured, indices, observation_latitude_weights
        )
        candidates.append(
            {
                "indices": indices,
                "diversity": subset_diversity(pairwise, indices),
                "observation_rmse": obs_rmse,
                "eligible": obs_rmse <= observation_limit,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:  # The reference subset should make this unreachable.
        raise RuntimeError("no observation-consistent K=4 subset was found")
    selected = max(eligible, key=lambda item: item["diversity"])
    reference_diversity = subset_diversity(pairwise, independent_indices)
    diagnostics = {
        "uses_ground_truth": False,
        "candidate_members": member_count,
        "returned_members": RETURN_K,
        "subsets_evaluated": len(candidates),
        "eligible_subsets": len(eligible),
        "spatial_pool": pool_factor,
        "reference_indices": list(independent_indices),
        "selected_indices": list(selected["indices"]),
        "reference_proxy_pairwise_mse": reference_diversity,
        "selected_proxy_pairwise_mse": selected["diversity"],
        "proxy_diversity_gain": (
            selected["diversity"] / max(reference_diversity, 1e-30) - 1.0
        ),
        "reference_observation_rmse": reference_observation_rmse,
        "selected_observation_rmse": selected["observation_rmse"],
        "observation_rmse_limit": observation_limit,
    }
    return selected["indices"], diagnostics


def relative(new: float, old: float) -> float:
    return new / max(old, 1e-30) - 1.0


def arm_checks(reference: dict, arm: dict) -> dict[str, bool]:
    checks = smoke_checks(reference, arm)
    checks.update(
        {
            "skill_rmse_preserved_2pct": relative(
                arm["skill_rmse_unobserved"], reference["skill_rmse_unobserved"]
            )
            <= 0.02,
            "observation_rmse_preserved_5pct": relative(
                arm["observation_rmse"], reference["observation_rmse"]
            )
            <= 0.05,
            "temporal_rmse_preserved_2pct": relative(
                arm["temporal_increment_rmse"], reference["temporal_increment_rmse"]
            )
            <= 0.02,
            "not_overdispersed": arm[
                "spread_skill_ratio_unobserved_finite_ensemble_corrected"
            ]
            <= 1.20,
        }
    )
    return checks


def plot_window_comparison(path: Path, arms: dict[str, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["independent", "post-hoc", "joint repulsion"]
    keys = ["independent_k4", "posthoc_k4", "joint_repulsion_k4"]
    colors = ["#777777", "#2b83ba", "#d7191c"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    plots = [
        ("pairwise_mse_unobserved", "Unobserved pairwise MSE", None),
        (
            "spread_skill_ratio_unobserved_finite_ensemble_corrected",
            "Corrected spread–skill ratio",
            1.0,
        ),
        ("coverage_2sigma_unobserved", "Unobserved 2-sigma coverage", None),
    ]
    for axis, (metric, title, target) in zip(axes, plots):
        axis.bar(labels, [arms[key][metric] for key in keys], color=colors)
        if target is not None:
            axis.axhline(target, color="0.2", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.candidate_seeds) != CANDIDATE_K or len(set(args.candidate_seeds)) != CANDIDATE_K:
        raise ValueError("--candidate-seeds must contain eight unique seeds")
    if len(args.independent_seeds) != RETURN_K:
        raise ValueError("--independent-seeds must contain exactly four seeds")
    if not set(args.independent_seeds).issubset(args.candidate_seeds):
        raise ValueError("independent seeds must be part of the candidate pool")
    if args.selection_spatial_pool < 1:
        raise ValueError("--selection-spatial-pool must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, joint_samples, truth, observation = load_joint_run(
        args.joint_repulsion_run, args.window
    )
    candidate_paths, source = baseline_members(
        args.baseline_root, args.data_root, args.window, args.candidate_seeds
    )
    candidate_arrays = [np.load(path, mmap_mode="r") for path in candidate_paths]
    for path, array in zip(candidate_paths, candidate_arrays):
        if array.ndim != 5 or array.shape[0] != 1:
            raise ValueError(f"expected one member in {path}, found {array.shape}")

    variables = list(metadata["variables"])
    baseline_metadata_path = candidate_paths[0].parents[1] / "metadata.json"
    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    if list(baseline_metadata["variables"]) != variables:
        raise ValueError("baseline and joint-repulsion variable lists differ")

    lat = np.asarray(np.load(args.data_root / "era5" / "lat.npy")).squeeze()
    lon = np.asarray(np.load(args.data_root / "era5" / "lon.npy")).squeeze()
    latitude_weights = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    observation_latitude_weights = latitude_weights[::8]
    means = np.load(args.data_root / "era5" / "normalize_mean.npz")
    stds = np.load(args.data_root / "era5" / "normalize_std.npz")

    independent_indices = tuple(
        args.candidate_seeds.index(seed) for seed in args.independent_seeds
    )
    selected_indices, selection = truth_free_selection(
        candidate_arrays,
        variables,
        observation,
        means,
        stds,
        latitude_weights,
        observation_latitude_weights,
        independent_indices,
        args.selection_spatial_pool,
        args.max_relative_observation_increase,
        args.max_absolute_observation_increase,
    )
    selected_seeds = [args.candidate_seeds[index] for index in selected_indices]

    plot_variables = resolve_variables(DEFAULT_PLOT_VARIABLES, variables)

    def independent_loader(channel: int) -> np.ndarray:
        return np.stack(
            [
                np.asarray(candidate_arrays[index][0, channel], dtype=np.float32)
                for index in independent_indices
            ],
            axis=0,
        )

    def posthoc_loader(channel: int) -> np.ndarray:
        return np.stack(
            [
                np.asarray(candidate_arrays[index][0, channel], dtype=np.float32)
                for index in selected_indices
            ],
            axis=0,
        )

    def joint_loader(channel: int) -> np.ndarray:
        return np.asarray(joint_samples[:, channel], dtype=np.float32)

    common = (
        truth,
        observation,
        variables,
        means,
        stds,
        lat,
        lon,
        latitude_weights,
        observation_latitude_weights,
        plot_variables,
    )
    independent, _ = evaluate(independent_loader, *common)
    posthoc, _ = evaluate(posthoc_loader, *common)
    joint, _ = evaluate(joint_loader, *common)
    arms = {
        "independent_k4": independent,
        "posthoc_k4": posthoc,
        "joint_repulsion_k4": joint,
    }

    scalar_keys = [
        key
        for key, value in independent.items()
        if isinstance(value, (int, float)) and key != "members"
    ]
    posthoc_checks = arm_checks(independent, posthoc)
    joint_checks = arm_checks(independent, joint)
    result = {
        "configuration": {
            "window": args.window,
            "source_index": source,
            "candidate_seeds": args.candidate_seeds,
            "independent_seeds": args.independent_seeds,
            "selected_seeds": selected_seeds,
            "candidate_generation_count": CANDIDATE_K,
            "returned_members": RETURN_K,
            "joint_repulsion_generation_count": RETURN_K,
            "joint_repulsion_run": str(args.joint_repulsion_run),
            "repulsion_strength": float(metadata["repulsion_strength"]),
        },
        "selection": selection,
        **arms,
        "posthoc_relative_to_independent": {
            key: relative(posthoc[key], independent[key]) for key in scalar_keys
        },
        "joint_repulsion_relative_to_independent": {
            key: relative(joint[key], independent[key]) for key in scalar_keys
        },
        "posthoc_validity_checks": posthoc_checks,
        "joint_repulsion_validity_checks": joint_checks,
        "posthoc_validity_passed": bool(all(posthoc_checks.values())),
        "joint_repulsion_validity_passed": bool(all(joint_checks.values())),
        "note": (
            "Post-hoc selection uses eight independent generations to return four; "
            "joint repulsion directly generates four. Selection is truth-free."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_window_comparison(args.output_dir / "three_way_comparison.png", arms)
    print(json.dumps(result, indent=2))
    print(f"Wrote three-way K=4 comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
