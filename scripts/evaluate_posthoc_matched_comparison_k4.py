#!/usr/bin/env python3
"""Matched three-way K=4 comparison using an existing K=8 control pool.

The first four particles of a decoder-only, no-repulsion K=8 run form the
ordinary K=4 reference.  A truth-free selector chooses four particles from all
eight control candidates.  Both are compared with an existing joint-repulsion
K=4 run produced with the same posterior-sampler settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_baseline_ensemble import DEFAULT_PLOT_VARIABLES, resolve_variables
from evaluate_posthoc_comparison_k4 import (
    CANDIDATE_K,
    RETURN_K,
    arm_checks,
    load_joint_run,
    plot_window_comparison,
    relative,
    truth_free_selection,
)
from evaluate_repulsion_smoke import evaluate
from evaluate_repulsion_smoke_k8 import load_batched_run


MATCHED_METADATA_FIELDS = (
    "sampler",
    "sampler_family",
    "num_steps",
    "seed",
    "sigma_data",
    "sigma_min",
    "sigma_max",
    "rho",
    "split",
    "sample_index",
    "obs_frames_spec",
    "obs_frames",
    "obs_downscale",
    "obs_mode",
    "obs_noise_std",
    "guidance_scale",
    "gamma",
    "meas_std",
    "grad_clip",
    "guide_through_net",
    "glue_weight",
    "weighting",
    "corrector_steps",
    "corrector_snr",
    "corrector_max_sigma",
    "corrector_min_sigma",
    "corrector_steps2",
    "corrector_max_sigma2",
    "guide_max_sigma",
    "guide_min_sigma",
    "guide_reuse",
    "guide_momentum",
    "ens_weight",
    "restart_k",
    "restart_sigma",
    "restart_steps",
    "guide_norm",
    "noise_scale",
    "warm_start",
    "sigma_start",
    "use_pp",
    "checkpoint",
    "config",
    "variables",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-k8-summary",
        type=Path,
        required=True,
        help="Completed per-window matched K=8 metric summary; supplies control_run.",
    )
    parser.add_argument("--joint-repulsion-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument(
        "--selection-spatial-pool",
        type=int,
        default=8,
        help="Block-average factor used only by the truth-free selector.",
    )
    parser.add_argument(
        "--max-relative-observation-increase", type=float, default=0.02
    )
    parser.add_argument(
        "--max-absolute-observation-increase", type=float, default=0.001
    )
    parser.add_argument("--expected-repulsion-strength", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def assert_matching_sampler(control: dict, repulsion: dict) -> dict:
    differences = {}
    for field in MATCHED_METADATA_FIELDS:
        control_value = control.get(field)
        repulsion_value = repulsion.get(field)
        if control_value != repulsion_value:
            differences[field] = {
                "control": control_value,
                "joint_repulsion": repulsion_value,
            }
    if differences:
        rendered = json.dumps(differences, indent=2)
        raise ValueError(f"control and joint sampler settings differ:\n{rendered}")
    if bool(control.get("guide_through_net", True)):
        raise ValueError("matched control is not the expected decoder-only guide")
    if float(control.get("repulsion_strength", 0.0)) != 0.0:
        raise ValueError("K=8 candidate run has nonzero repulsion strength")
    return {field: control.get(field) for field in MATCHED_METADATA_FIELDS}


def main() -> None:
    args = parse_args()
    if args.selection_spatial_pool < 1:
        raise ValueError("--selection-spatial-pool must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_result = json.loads(args.full_k8_summary.read_text(encoding="utf-8"))
    full_configuration = full_result["configuration"]
    if int(full_configuration["window"]) != args.window:
        raise ValueError("full-K8 summary window does not match --window")
    control_run = Path(full_configuration["control_run"])

    control_metadata, candidates, truth, observation = load_batched_run(
        control_run, args.window, "matched control candidate pool"
    )
    joint_metadata, joint_samples, joint_truth, joint_observation = load_joint_run(
        args.joint_repulsion_run, args.window
    )
    if candidates.shape[0] != CANDIDATE_K:
        raise ValueError(f"expected K=8 candidate pool, found {candidates.shape}")
    if joint_samples.shape[0] != RETURN_K:
        raise ValueError(f"expected joint K=4, found {joint_samples.shape}")
    if not np.isclose(
        float(joint_metadata["repulsion_strength"]),
        args.expected_repulsion_strength,
    ):
        raise ValueError(
            "joint run repulsion strength does not match the requested comparison"
        )
    if truth.shape != joint_truth.shape:
        raise ValueError("control and joint ground-truth shapes differ")
    if observation.shape != joint_observation.shape or not np.array_equal(
        observation, joint_observation
    ):
        raise ValueError("control and joint observations differ")

    matched_settings = assert_matching_sampler(control_metadata, joint_metadata)
    variables = list(control_metadata["variables"])
    lat = np.asarray(np.load(args.data_root / "era5" / "lat.npy")).squeeze()
    lon = np.asarray(np.load(args.data_root / "era5" / "lon.npy")).squeeze()
    latitude_weights = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    observation_latitude_weights = latitude_weights[::8]
    means = np.load(args.data_root / "era5" / "normalize_mean.npz")
    stds = np.load(args.data_root / "era5" / "normalize_std.npz")

    candidate_arrays = [candidates[index : index + 1] for index in range(CANDIDATE_K)]
    independent_indices = tuple(range(RETURN_K))
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
    plot_variables = resolve_variables(DEFAULT_PLOT_VARIABLES, variables)

    def independent_loader(channel: int) -> np.ndarray:
        return np.asarray(candidates[list(independent_indices), channel], dtype=np.float32)

    def posthoc_loader(channel: int) -> np.ndarray:
        return np.asarray(candidates[list(selected_indices), channel], dtype=np.float32)

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
            "source_index": full_configuration["source_index"],
            "candidate_member_indices": list(range(CANDIDATE_K)),
            "independent_member_indices": list(independent_indices),
            "selected_member_indices": list(selected_indices),
            "candidate_generation_count": CANDIDATE_K,
            "returned_members": RETURN_K,
            "joint_repulsion_generation_count": RETURN_K,
            "full_k8_summary": str(args.full_k8_summary),
            "matched_control_run": str(control_run),
            "joint_repulsion_run": str(args.joint_repulsion_run),
            "repulsion_strength": float(joint_metadata["repulsion_strength"]),
            "sampler_match_verified": True,
        },
        "matched_sampler_settings": matched_settings,
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
            "All three returned ensembles have K=4. Independent and post-hoc "
            "members come from the same decoder-only K=8 control pool; the "
            "joint run has identical non-repulsion sampler settings. Post-hoc "
            "uses eight generations to return four; joint repulsion generates "
            "four directly. Selection is truth-free."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_window_comparison(args.output_dir / "matched_three_way_comparison.png", arms)
    print(json.dumps(result, indent=2))
    print(f"Wrote matched three-way K=4 comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
