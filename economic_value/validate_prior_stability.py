"""Validate the nested-sample prior benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results" / "prior_stability_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    frame = pd.read_csv(args.results / "outcomes.csv")
    errors = []
    if len(frame) != 90:
        errors.append(f"expected 90 rows, found {len(frame)}")
    counts = set(frame["sample_count"].unique())
    if counts != {3, 6, 12}:
        errors.append(f"unexpected sample counts: {sorted(counts)}")
    per_case = frame.groupby("case_id")["sample_count"].apply(set)
    if any(values != {3, 6, 12} for values in per_case):
        errors.append("nested sample grid is incomplete")
    if set(frame["status"]) != {"optimal"}:
        errors.append("not every prior benchmark is certified optimal")
    if frame["mip_gap"].fillna(0).max() > 1e-6:
        errors.append("prior benchmark optimality gap exceeds tolerance")
    if (frame["total_expenditure"] > frame["budget"] + 1e-3).any():
        errors.append("prior benchmark execution exceeds budget")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": int(len(frame)),
        "environments": int(frame["environment_id"].nunique()),
        "sample_counts": sorted(int(value) for value in counts),
        "maximum_gap": float(frame["mip_gap"].fillna(0).max()),
    }
    (args.results / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
