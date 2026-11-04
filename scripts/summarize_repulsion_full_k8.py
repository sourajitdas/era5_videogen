#!/usr/bin/env python3
"""Aggregate the final eight-window matched K=8 repulsion evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
        default=Path("outputs/metrics/repulsion_full_k8"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/repulsion_full_k8/summary.json"),
    )
    return parser.parse_args()


def relative(new: float, old: float) -> float:
    return new / max(old, 1e-30) - 1.0


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    windows = [row["window"] for row in rows]
    x = np.arange(len(windows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    axes[0, 0].bar(x, [100 * row["pairwise_gain"] for row in rows])
    axes[0, 0].axhline(0, color="0.25", linewidth=1)
    axes[0, 0].set_title("Unobserved pairwise-diversity gain")
    axes[0, 0].set_ylabel("percent")

    axes[0, 1].plot(
        x, [row["control_spread_skill_ratio"] for row in rows], "o-", label="control"
    )
    axes[0, 1].plot(
        x,
        [row["repulsion_spread_skill_ratio"] for row in rows],
        "o-",
        label="repulsion",
    )
    axes[0, 1].axhline(1.0, color="0.25", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Corrected unobserved spread–skill ratio")
    axes[0, 1].legend()

    axes[1, 0].bar(
        x - 0.18,
        [100 * row["range_coverage_gain"] for row in rows],
        width=0.36,
        label="range",
    )
    axes[1, 0].bar(
        x + 0.18,
        [100 * row["two_sigma_coverage_gain"] for row in rows],
        width=0.36,
        label="2 sigma",
    )
    axes[1, 0].set_title("Unobserved coverage gain")
    axes[1, 0].set_ylabel("percentage points")
    axes[1, 0].legend()

    axes[1, 1].bar(
        x - 0.27,
        [100 * row["skill_change"] for row in rows],
        width=0.27,
        label="skill RMSE",
    )
    axes[1, 1].bar(
        x,
        [100 * row["observation_change"] for row in rows],
        width=0.27,
        label="observation RMSE",
    )
    axes[1, 1].bar(
        x + 0.27,
        [100 * row["temporal_change"] for row in rows],
        width=0.27,
        label="temporal RMSE",
    )
    axes[1, 1].axhline(0, color="0.25", linewidth=1)
    axes[1, 1].set_title("Validity changes")
    axes[1, 1].set_ylabel("percent")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xticks(x, [str(window) for window in windows])
        ax.set_xlabel("window")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.glob("w??_alpha0p20/summary.json"))
    if len(paths) != 8:
        raise SystemExit(f"expected 8 window summaries under {args.root}, found {len(paths)}")

    rows = []
    summaries = []
    for path in paths:
        result = json.loads(path.read_text())
        cfg = result["configuration"]
        control = result["control_k8"]
        repulsion = result["repulsion_k8"]
        row = {
            "window": int(cfg["window"]),
            "source_index": cfg["source_index"],
            "validity_passed": bool(result["validity_passed"]),
            "pairwise_gain": relative(
                repulsion["pairwise_mse_unobserved"],
                control["pairwise_mse_unobserved"],
            ),
            "spread_gain": relative(
                repulsion["spread_unobserved"], control["spread_unobserved"]
            ),
            "control_spread_skill_ratio": control[
                "spread_skill_ratio_unobserved_finite_ensemble_corrected"
            ],
            "repulsion_spread_skill_ratio": repulsion[
                "spread_skill_ratio_unobserved_finite_ensemble_corrected"
            ],
            "range_coverage_gain": (
                repulsion["coverage_range_unobserved"]
                - control["coverage_range_unobserved"]
            ),
            "two_sigma_coverage_gain": (
                repulsion["coverage_2sigma_unobserved"]
                - control["coverage_2sigma_unobserved"]
            ),
            "skill_change": relative(
                repulsion["skill_rmse_unobserved"],
                control["skill_rmse_unobserved"],
            ),
            "observation_change": relative(
                repulsion["observation_rmse"], control["observation_rmse"]
            ),
            "temporal_change": relative(
                repulsion["temporal_increment_rmse"],
                control["temporal_increment_rmse"],
            ),
        }
        rows.append(row)
        summaries.append(result)

    rows.sort(key=lambda row: row["window"])
    found = [row["window"] for row in rows]
    if found != list(range(8)):
        raise SystemExit(f"expected windows 0..7, found {found}")

    control_aggregate = {
        metric: mean(result["control_k8"][metric] for result in summaries)
        for metric in METRICS
    }
    repulsion_aggregate = {
        metric: mean(result["repulsion_k8"][metric] for result in summaries)
        for metric in METRICS
    }
    relative_change = {
        metric: relative(repulsion_aggregate[metric], control_aggregate[metric])
        for metric in METRICS
    }
    mean_skill_change = mean(row["skill_change"] for row in rows)
    mean_observation_change = mean(row["observation_change"] for row in rows)
    aggregate_checks = {
        "all_windows_valid": all(row["validity_passed"] for row in rows),
        "diversity_increased_every_window": all(
            row["pairwise_gain"] > 0 for row in rows
        ),
        "mean_skill_increase_at_most_1pct": mean_skill_change <= 0.01,
        "max_skill_increase_at_most_2pct": max(
            row["skill_change"] for row in rows
        )
        <= 0.02,
        "mean_observation_increase_at_most_2pct": mean_observation_change <= 0.02,
        "max_observation_increase_at_most_5pct": max(
            row["observation_change"] for row in rows
        )
        <= 0.05,
        "mean_corrected_spread_skill_in_0p8_to_1p2": (
            0.8
            <= repulsion_aggregate[
                "spread_skill_ratio_unobserved_finite_ensemble_corrected"
            ]
            <= 1.2
        ),
    }
    output = {
        "configuration": {
            "windows": list(range(8)),
            "members": 8,
            "repulsion_strength": 0.20,
            "repulsion_sigma_band": [0.05, 5.0],
            "repulsion_schedule": "sine",
            "particle_coupling": "global_all_particles",
        },
        "control_k8": control_aggregate,
        "repulsion_k8": repulsion_aggregate,
        "relative_change": relative_change,
        "aggregate_checks": aggregate_checks,
        "experiment_passed": bool(all(aggregate_checks.values())),
        "per_window": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_name("per_window.csv"), rows)
    plot_summary(args.output.with_name("full_k8_comparison.png"), rows)
    print(json.dumps(output, indent=2))
    print(f"Wrote final K=8 aggregate to {args.output}")


if __name__ == "__main__":
    main()
