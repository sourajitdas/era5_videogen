#!/usr/bin/env python3
"""Aggregate the three-window K=4 post-hoc method comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("independent_k4", "posthoc_k4", "joint_repulsion_k4")
METRICS = (
    "pairwise_mse_unobserved",
    "spread_unobserved",
    "spread_skill_ratio_unobserved_finite_ensemble_corrected",
    "coverage_range_unobserved",
    "coverage_2sigma_unobserved",
    "skill_rmse_unobserved",
    "observation_rmse",
    "temporal_increment_rmse",
    "temporal_increment_energy_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/metrics/posthoc_comparison_k4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/posthoc_comparison_k4/summary.json"),
    )
    parser.add_argument("--expected-windows", type=int, nargs="+", default=[0, 3, 7])
    return parser.parse_args()


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def relative(new: float, old: float) -> float:
    return new / max(old, 1e-30) - 1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict], aggregate: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    windows = [row["window"] for row in rows]
    x = np.arange(len(windows))
    width = 0.36
    colors = {"posthoc": "#2b83ba", "joint": "#d7191c"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    axes[0, 0].bar(
        x - width / 2,
        [100 * row["posthoc_pairwise_gain"] for row in rows],
        width,
        label="post-hoc K=4",
        color=colors["posthoc"],
    )
    axes[0, 0].bar(
        x + width / 2,
        [100 * row["joint_pairwise_gain"] for row in rows],
        width,
        label="joint repulsion K=4",
        color=colors["joint"],
    )
    axes[0, 0].axhline(0, color="0.25", linewidth=1)
    axes[0, 0].set_title("Unobserved pairwise-diversity gain")
    axes[0, 0].set_ylabel("percent vs independent K=4")
    axes[0, 0].legend(fontsize=8)

    for arm, label, color, marker in (
        ("independent_k4", "independent", "#777777", "o"),
        ("posthoc_k4", "post-hoc", colors["posthoc"], "s"),
        ("joint_repulsion_k4", "joint repulsion", colors["joint"], "^"),
    ):
        axes[0, 1].plot(
            x,
            [row[f"{arm}_spread_skill_ratio"] for row in rows],
            marker=marker,
            label=label,
            color=color,
        )
    axes[0, 1].axhline(1.0, color="0.25", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Corrected unobserved spread–skill ratio")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        x - width / 2,
        [100 * row["posthoc_two_sigma_gain"] for row in rows],
        width,
        label="post-hoc K=4",
        color=colors["posthoc"],
    )
    axes[1, 0].bar(
        x + width / 2,
        [100 * row["joint_two_sigma_gain"] for row in rows],
        width,
        label="joint repulsion K=4",
        color=colors["joint"],
    )
    axes[1, 0].axhline(0, color="0.25", linewidth=1)
    axes[1, 0].set_title("Unobserved 2-sigma coverage gain")
    axes[1, 0].set_ylabel("percentage points vs independent K=4")
    axes[1, 0].legend(fontsize=8)

    error_metrics = [
        ("skill_rmse_unobserved", "skill RMSE"),
        ("observation_rmse", "observation RMSE"),
        ("temporal_increment_rmse", "temporal RMSE"),
    ]
    error_x = np.arange(len(error_metrics))
    independent = aggregate["independent_k4"]
    axes[1, 1].bar(
        error_x - width / 2,
        [
            100 * relative(aggregate["posthoc_k4"][metric], independent[metric])
            for metric, _ in error_metrics
        ],
        width,
        label="post-hoc K=4",
        color=colors["posthoc"],
    )
    axes[1, 1].bar(
        error_x + width / 2,
        [
            100
            * relative(aggregate["joint_repulsion_k4"][metric], independent[metric])
            for metric, _ in error_metrics
        ],
        width,
        label="joint repulsion K=4",
        color=colors["joint"],
    )
    axes[1, 1].axhline(0, color="0.25", linewidth=1)
    axes[1, 1].set_xticks(error_x, [label for _, label in error_metrics])
    axes[1, 1].set_title("Mean validity-metric changes")
    axes[1, 1].set_ylabel("percent vs independent K=4")
    axes[1, 1].legend(fontsize=8)

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(x, [str(window) for window in windows])
        axis.set_xlabel("window")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "K=4 diversity methods (post-hoc selects 4 of 8; joint generates 4 directly)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.glob("w??/summary.json"))
    if len(paths) != len(args.expected_windows):
        raise SystemExit(
            f"expected {len(args.expected_windows)} summaries under {args.root}, "
            f"found {len(paths)}"
        )

    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results.sort(key=lambda result: int(result["configuration"]["window"]))
    found = [int(result["configuration"]["window"]) for result in results]
    if found != args.expected_windows:
        raise SystemExit(f"expected windows {args.expected_windows}, found {found}")

    rows = []
    for result in results:
        independent = result["independent_k4"]
        posthoc = result["posthoc_k4"]
        joint = result["joint_repulsion_k4"]
        row = {
            "window": int(result["configuration"]["window"]),
            "source_index": result["configuration"]["source_index"],
            "selected_seeds": " ".join(map(str, result["configuration"]["selected_seeds"])),
            "posthoc_validity_passed": bool(result["posthoc_validity_passed"]),
            "joint_validity_passed": bool(result["joint_repulsion_validity_passed"]),
            "posthoc_pairwise_gain": relative(
                posthoc["pairwise_mse_unobserved"], independent["pairwise_mse_unobserved"]
            ),
            "joint_pairwise_gain": relative(
                joint["pairwise_mse_unobserved"], independent["pairwise_mse_unobserved"]
            ),
            "posthoc_two_sigma_gain": (
                posthoc["coverage_2sigma_unobserved"]
                - independent["coverage_2sigma_unobserved"]
            ),
            "joint_two_sigma_gain": (
                joint["coverage_2sigma_unobserved"]
                - independent["coverage_2sigma_unobserved"]
            ),
        }
        for arm in ARMS:
            row[f"{arm}_spread_skill_ratio"] = result[arm][
                "spread_skill_ratio_unobserved_finite_ensemble_corrected"
            ]
        rows.append(row)

    aggregate = {
        arm: {
            metric: mean(result[arm][metric] for result in results)
            for metric in METRICS
        }
        for arm in ARMS
    }
    independent = aggregate["independent_k4"]
    changes = {
        arm: {
            metric: relative(aggregate[arm][metric], independent[metric])
            for metric in METRICS
        }
        for arm in ("posthoc_k4", "joint_repulsion_k4")
    }
    checks = {
        "all_posthoc_windows_valid": all(
            result["posthoc_validity_passed"] for result in results
        ),
        "all_joint_repulsion_windows_valid": all(
            result["joint_repulsion_validity_passed"] for result in results
        ),
        "posthoc_diversity_increased_every_window": all(
            row["posthoc_pairwise_gain"] > 0 for row in rows
        ),
        "joint_diversity_increased_every_window": all(
            row["joint_pairwise_gain"] > 0 for row in rows
        ),
        "selection_was_truth_free_every_window": all(
            result["selection"]["uses_ground_truth"] is False for result in results
        ),
    }
    output = {
        "configuration": {
            "windows": args.expected_windows,
            "returned_members": RETURNED_MEMBERS,
            "independent_generation_count": RETURNED_MEMBERS,
            "posthoc_candidate_generation_count": 8,
            "joint_repulsion_generation_count": RETURNED_MEMBERS,
            "posthoc_selection_ratio": "select 4 of 8",
            "repulsion_strength": 0.20,
        },
        **aggregate,
        "relative_to_independent": changes,
        "comparison_checks": checks,
        "comparison_complete": True,
        "comparison_passed": bool(all(checks.values())),
        "per_window": rows,
        "interpretation_note": (
            "Post-hoc and joint results have the same returned K=4, but post-hoc "
            "requires twice as many generated candidates. Ground truth is never "
            "used for member selection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_name("per_window.csv"), rows)
    plot_summary(args.output.with_name("posthoc_full_comparison.png"), rows, aggregate)
    print(json.dumps(output, indent=2))
    print(f"Wrote post-hoc comparison aggregate to {args.output}")


RETURNED_MEMBERS = 4


if __name__ == "__main__":
    main()
