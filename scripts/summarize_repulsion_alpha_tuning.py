#!/usr/bin/env python3
"""Aggregate multi-window K=4 alpha-tuning summaries and select a candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/metrics/repulsion_tuning_k4"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/metrics/repulsion_tuning_k4/alpha_tuning_summary.json"),
    )
    parser.add_argument(
        "--expected-windows", type=int, nargs="+", default=[0, 3, 7]
    )
    parser.add_argument(
        "--max-skill-increase", type=float, default=0.01,
        help="Maximum allowed mean relative unobserved skill-RMSE increase.",
    )
    parser.add_argument(
        "--max-observation-increase", type=float, default=0.02,
        help="Maximum allowed mean relative observation-RMSE increase.",
    )
    parser.add_argument(
        "--max-window-skill-increase", type=float, default=0.02,
        help="Maximum allowed relative skill-RMSE increase on any one window.",
    )
    parser.add_argument(
        "--max-window-observation-increase", type=float, default=0.05,
        help="Maximum allowed relative observation-RMSE increase on any one window.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(new: float, old: float) -> float:
    return new / max(old, 1e-30) - 1.0


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def nanmean(values: list[float]) -> float:
    usable = finite(values)
    return sum(usable) / len(usable) if usable else math.nan


def nanmaxabs(values: list[float]) -> float:
    usable = finite(values)
    return max(map(abs, usable)) if usable else math.nan


def main() -> None:
    args = parse_args()
    summaries = sorted(args.root.glob("w??_alpha*/summary.json"))
    if not summaries:
        raise SystemExit(f"no tuning summaries found under {args.root}")

    rows = []
    grouped: dict[float, list[dict]] = defaultdict(list)
    for path in summaries:
        result = json.loads(path.read_text())
        cfg = result["configuration"]
        alpha = float(cfg["repulsion_strength"])
        window = int(cfg["window"])
        control = result["baseline_k4"]
        repulsion = result["repulsion_k4"]
        physical_control = control.get("physical_consistency", {})
        physical_repulsion = repulsion.get("physical_consistency", {})
        row = {
            "window": window,
            "source_index": cfg["source_index"],
            "alpha": alpha,
            "smoke_passed": bool(result["smoke_passed"]),
            "pairwise_gain": relative(
                repulsion["pairwise_mse_unobserved"],
                control["pairwise_mse_unobserved"],
            ),
            "spread_gain": relative(
                repulsion["spread_unobserved"], control["spread_unobserved"]
            ),
            "spread_skill_ratio": repulsion[
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
            "geostrophic_cosine_change": (
                physical_repulsion.get("geostrophic_cosine", math.nan)
                - physical_control.get("geostrophic_cosine", math.nan)
            ),
            "hypsometric_rmse_change": (
                physical_repulsion.get("hypsometric_rmse_k", math.nan)
                - physical_control.get("hypsometric_rmse_k", math.nan)
            ),
        }
        rows.append(row)
        grouped[alpha].append(row)

    expected = set(args.expected_windows)
    aggregate = []
    for alpha, alpha_rows in sorted(grouped.items()):
        found = {int(row["window"]) for row in alpha_rows}
        if found != expected:
            raise SystemExit(
                f"alpha={alpha} has windows {sorted(found)}, expected {sorted(expected)}"
            )
        record = {"alpha": alpha, "windows": sorted(found)}
        for key in (
            "pairwise_gain", "spread_gain", "spread_skill_ratio",
            "range_coverage_gain", "two_sigma_coverage_gain", "skill_change",
            "observation_change", "temporal_change", "geostrophic_cosine_change",
            "hypsometric_rmse_change",
        ):
            values = [float(row[key]) for row in alpha_rows]
            record[f"mean_{key}"] = nanmean(values)
            record[f"max_abs_{key}"] = nanmaxabs(values)
        record["all_smoke_checks_passed"] = bool(
            all(row["smoke_passed"] for row in alpha_rows)
        )
        record["max_skill_increase"] = max(
            float(row["skill_change"]) for row in alpha_rows
        )
        record["max_observation_increase"] = max(
            float(row["observation_change"]) for row in alpha_rows
        )
        record["eligible"] = bool(
            record["all_smoke_checks_passed"]
            and record["mean_skill_change"] <= args.max_skill_increase
            and record["mean_observation_change"] <= args.max_observation_increase
            and record["max_skill_increase"] <= args.max_window_skill_increase
            and record["max_observation_increase"]
            <= args.max_window_observation_increase
        )
        record["distance_from_unit_spread_skill"] = abs(
            record["mean_spread_skill_ratio"] - 1.0
        )
        aggregate.append(record)

    eligible = [record for record in aggregate if record["eligible"]]
    selected = min(
        eligible,
        key=lambda record: record["distance_from_unit_spread_skill"],
        default=None,
    )
    output = {
        "configuration": {
            "root": str(args.root),
            "expected_windows": args.expected_windows,
            "selection_rule": (
                "among strengths passing every smoke check with mean skill increase "
                f"<= {args.max_skill_increase:.3f} and mean observation increase <= "
                f"{args.max_observation_increase:.3f}, with no window exceeding "
                f"{args.max_window_skill_increase:.3f} skill increase or "
                f"{args.max_window_observation_increase:.3f} observation increase, "
                "choose mean corrected "
                "unobserved spread-skill ratio closest to 1"
            ),
        },
        "per_alpha": aggregate,
        "selected_alpha": None if selected is None else selected["alpha"],
        "selection_complete": selected is not None,
        "next_step": (
            "Use selected_alpha for the eight-window matched K=8 evaluation. "
            "Run a sigma-band search only if no alpha is eligible or calibration "
            "remains materially underdispersed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps(output, indent=2))
    print(f"Wrote alpha tuning summary to {args.output}")


if __name__ == "__main__":
    main()
