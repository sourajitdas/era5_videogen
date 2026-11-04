#!/usr/bin/env python3
"""Aggregate the matched three-window K=4 method comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_posthoc_comparison_k4 import ARMS, METRICS, mean, plot_summary, relative


RETURNED_MEMBERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/metrics/posthoc_matched_comparison_k4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/posthoc_matched_comparison_k4/summary.json"),
    )
    parser.add_argument("--expected-windows", type=int, nargs="+", default=[0, 3, 7])
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
            "selected_member_indices": " ".join(
                map(str, result["configuration"]["selected_member_indices"])
            ),
            "sampler_match_verified": bool(
                result["configuration"]["sampler_match_verified"]
            ),
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
        "sampler_match_verified_every_window": all(
            result["configuration"]["sampler_match_verified"] for result in results
        ),
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
            "matched_sampler": "decoder-only posterior guidance",
        },
        **aggregate,
        "relative_to_independent": changes,
        "comparison_checks": checks,
        "comparison_complete": True,
        "comparison_passed": bool(all(checks.values())),
        "per_window": rows,
        "interpretation_note": (
            "This is the corrected apples-to-apples comparison. All candidates "
            "and joint particles use matching decoder-only posterior-sampler "
            "settings; only the diversity mechanism and candidate budget differ."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_name("per_window.csv"), rows)
    plot_summary(
        args.output.with_name("posthoc_matched_full_comparison.png"), rows, aggregate
    )
    print(json.dumps(output, indent=2))
    print(f"Wrote matched post-hoc aggregate to {args.output}")


if __name__ == "__main__":
    main()
