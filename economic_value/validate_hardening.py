"""Validate saved hardening outcomes without rerunning experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results" / "hardening_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    outcomes = pd.read_csv(args.results / "outcomes.csv")
    errors: list[str] = []

    pairs = outcomes[outcomes["experiment"].isin([3, 4])]
    grouped = defaultdict(list)
    for row in pairs.to_dict(orient="records"):
        grouped[row["case_id"]].append(row)
    for case_id, rows in grouped.items():
        methods = Counter(row["method"] for row in rows)
        if methods != Counter({"Catalog + geography": 1, "Current feasibility": 1}):
            errors.append(f"{case_id}: incomplete information pair")
        for row in rows:
            if row["total_expenditure"] > row["budget"] + 1e-4 * max(1.0, row["budget"]):
                errors.append(f"{case_id}/{row['method']}: budget exceeded")
            if not -1e-8 <= row["qualifying_coverage"] <= 1.0 + 1e-8:
                errors.append(f"{case_id}/{row['method']}: invalid coverage")
            if row["method"] == "Current feasibility" and abs(row["planning_error"]) > 1e-6:
                errors.append(f"{case_id}: current-feasibility planning and execution disagree")

    correction = outcomes[outcomes["experiment"] == 5]
    for row in correction.to_dict(orient="records"):
        if row["total_expenditure"] > row["budget"] + 1e-4 * max(1.0, row["budget"]):
            errors.append(f"{row['case_id']}: correction budget exceeded")
        if not -1e-8 <= row["qualifying_coverage"] <= 1.0 + 1e-8:
            errors.append(f"{row['case_id']}: invalid corrected coverage")
    null = correction[
        (correction["detection_fraction"] == 0.0)
        & (correction["refund_fraction"] == 1.0)
    ]
    if len(null) != correction["environment_id"].nunique():
        errors.append("Correction null control is incomplete")
    if len(null) and null["profit_difference"].abs().max() > 1e-6:
        errors.append("Immediate fully refundable correction does not remove the advantage")

    expected_counts = {
        3: 20 * 3 * 4 * 3 * 2,
        4: 20 * 3 * 3 * 3 * 2,
        5: 20 * 3 * 4 * 3,
    }
    observed_counts = outcomes.groupby("experiment").size().to_dict()
    for experiment, expected in expected_counts.items():
        if observed_counts.get(experiment) != expected:
            errors.append(
                f"Experiment {experiment}: expected {expected} rows, "
                f"found {observed_counts.get(experiment)}"
            )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "outcome_rows": int(len(outcomes)),
        "counts_by_experiment": {str(key): int(value) for key, value in observed_counts.items()},
        "null_control_max_absolute_difference": (
            float(null["profit_difference"].abs().max()) if len(null) else None
        ),
        "checks": [
            "same greedy method pair per economic and resource case",
            "fixed plus variable spending remains within budget",
            "coverage remains within [0,1]",
            "current-feasibility planning equals realized execution",
            "correction grid is complete",
            "immediate fully refundable correction removes the timing advantage",
        ],
    }
    (args.results / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
