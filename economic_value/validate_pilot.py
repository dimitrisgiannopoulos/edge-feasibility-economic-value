"""Validate saved pilot outcomes without rerunning an experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results" / "pilot_v1"
EXPECTED_METHODS = {"G-Catalog", "M-Catalog", "M-Prior", "G-EDFS", "M-EDFS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    with (args.results / "outcomes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    bounded_prior_cases: list[dict[str, str]] = []
    for row in rows:
        grouped[row["case_id"]].append(row)
    errors: list[str] = []
    for case, case_rows in grouped.items():
        by_method = {row["method"]: row for row in case_rows}
        if set(by_method) != EXPECTED_METHODS or len(case_rows) != len(EXPECTED_METHODS):
            errors.append(f"{case}: incomplete or duplicate method set")
            continue
        reference = by_method["M-EDFS"]
        optimum = float(reference["realized_profit"])
        common_fields = ("budget", "max_revenue", "support_density", "seed")
        for field in common_fields:
            if len({row[field] for row in case_rows}) != 1:
                errors.append(f"{case}: inconsistent {field}")
        for row in case_rows:
            method = row["method"]
            profit = float(row["realized_profit"])
            spending = float(row["total_expenditure"])
            budget = float(row["budget"])
            coverage = float(row["qualifying_coverage"])
            if profit > optimum + 0.02:
                errors.append(f"{case}/{method}: exceeds full-information optimum")
            if spending > budget + 1e-3 * max(1.0, budget):
                errors.append(f"{case}/{method}: spending exceeds budget")
            if not -1e-8 <= coverage <= 1.0 + 1e-8:
                errors.append(f"{case}/{method}: invalid coverage")
            if method in {"M-Catalog", "M-EDFS"}:
                if row["planning_status"] != "optimal":
                    errors.append(f"{case}/{method}: nonoptimal reference")
                if row["planning_mip_gap"] and float(row["planning_mip_gap"]) > 1e-6:
                    errors.append(f"{case}/{method}: excessive MIP gap")
            elif method == "M-Prior" and row["planning_status"] != "optimal":
                if not row["planning_mip_gap"]:
                    errors.append(f"{case}/{method}: bounded incumbent lacks a reported gap")
                else:
                    bounded_prior_cases.append({
                        "case_id": case, "status": row["planning_status"],
                        "mip_gap": row["planning_mip_gap"],
                    })
        if int(reference["experiment"]) == 2:
            alternatives = float(reference["alternatives"])
            observed = float(reference["mean_supported_alternatives"])
            if abs(alternatives - observed) > 1e-8:
                errors.append(f"{case}: support count is not matched")
    replica_counts = Counter()
    for case_rows in grouped.values():
        row = case_rows[0]
        if int(row["experiment"]) == 1:
            key = (1, row["topology"], row["budget_index"])
        else:
            key = (2, row["study"], row["overlap"], row["support_pattern"], row["alternatives"])
        replica_counts[key] += 1
    report = {
        "status": "PASS" if not errors else "FAIL", "outcome_rows": len(rows),
        "cases": len(grouped), "errors": errors,
        "bounded_prior_cases": bounded_prior_cases,
        "replicas_per_configuration": {str(key): value for key, value in sorted(replica_counts.items(), key=str)},
        "checks": [
            "five methods exactly once per case",
            "common environment and budget within each paired case",
            "full-information MILP weakly dominates every realized footprint",
            "fixed plus variable spending respects the budget",
            "coverage remains within [0,1]",
            "catalog and full-information MILP references are certified optimal",
            "time-limited prior-informed comparators retain their reported MIP gaps",
            "Experiment 2 preserves supported alternatives per demand group",
        ],
    }
    (args.results / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
