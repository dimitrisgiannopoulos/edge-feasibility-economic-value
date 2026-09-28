"""Derive footprint and accounting diagnostics from final paired outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .run_diagnostics import DEFAULT_CALIBRATION, _write_worked_example


ROOT = Path(__file__).resolve().parent.parent


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "final_v2" / "outcomes.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "final_decision_diagnostics_v1")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        outcomes = list(csv.DictReader(handle))
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in outcomes:
        grouped.setdefault(row["case_id"], {})[row["method"]] = row
    pairs = []
    for case_id, methods in grouped.items():
        catalog = methods["Catalog + geography"]
        informed = methods["Current feasibility"]
        catalog_sites = set(json.loads(catalog["selected_sites"]))
        informed_sites = set(json.loads(informed["selected_sites"]))
        retained = catalog_sites & informed_sites
        union = catalog_sites | informed_sites
        row: dict[str, Any] = {
            "kind": "decision", "case_id": case_id,
            "environment_id": informed["environment_id"], "seed": int(informed["seed"]),
            "topology": informed["topology"], "budget_index": int(informed["budget_index"]),
            "budget": float(informed["budget"]),
            "footprints_differ": int(catalog_sites != informed_sites),
            "catalog_sites": len(catalog_sites), "informed_sites": len(informed_sites),
            "sites_added": len(informed_sites - catalog_sites),
            "sites_removed": len(catalog_sites - informed_sites),
            "sites_retained": len(retained),
            "footprint_jaccard": len(retained) / len(union) if union else 1.0,
            "added_site_ids": json.dumps(sorted(informed_sites - catalog_sites)),
            "removed_site_ids": json.dumps(sorted(catalog_sites - informed_sites)),
            "retained_site_ids": json.dumps(sorted(retained)),
            "catalog_selected_site_ids": json.dumps(sorted(catalog_sites)),
            "informed_selected_site_ids": json.dumps(sorted(informed_sites)),
            "catalog_realized_profit": float(catalog["profit"]),
            "informed_realized_profit": float(informed["profit"]),
            "profit_difference": float(informed["profit"]) - float(catalog["profit"]),
            "revenue_difference": float(informed["revenue"]) - float(catalog["revenue"]),
            "fixed_cost_difference": float(informed["fixed_cost"]) - float(catalog["fixed_cost"]),
            "variable_cost_difference": float(informed["variable_cost"]) - float(catalog["variable_cost"]),
            "catalog_coverage": float(catalog["qualifying_coverage"]),
            "informed_coverage": float(informed["qualifying_coverage"]),
            "max_revenue": float(informed["max_revenue"]),
        }
        row["revenue_normalized_profit_difference"] = row["profit_difference"] / row["max_revenue"]
        row["accounting_residual"] = (
            row["profit_difference"] - row["revenue_difference"]
            + row["fixed_cost_difference"] + row["variable_cost_difference"]
        )
        for metric in ("revenue", "fixed_cost", "variable_cost", "total_expenditure",
                       "stranded_expenditure", "local_sites", "regional_sites", "central_sites"):
            row[f"catalog_{metric}"] = float(catalog[metric])
            row[f"informed_{metric}"] = float(informed[metric])
        pairs.append(row)
    pairs.sort(key=lambda row: row["case_id"])
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "decision_pairs.csv", pairs)
    calibration = json.loads(args.calibration.read_text())
    _write_worked_example(
        args.output, pairs, calibration,
        "all 300 independent final middle-budget environments",
    )
    manifest = {
        "paired_cases": len(pairs),
        "environments": len({row["environment_id"] for row in pairs}),
        "maximum_accounting_residual": max(abs(row["accounting_residual"]) for row in pairs),
        "worked_example_rule": "positive middle-budget case nearest the final-sample median gain",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
