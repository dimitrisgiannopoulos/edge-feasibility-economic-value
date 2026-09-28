"""Derive exact correction accounting from saved hardening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .environment import build_environment


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardening", type=Path,
                        default=ROOT / "results" / "hardening_v1" / "outcomes.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "diagnostics_v1" / "correction_accounting.csv")
    args = parser.parse_args()
    rows = pd.read_csv(args.hardening)
    correction = rows[rows.experiment == 5].copy()
    baseline = rows[
        (rows.experiment == 3)
        & (rows.fixed_cost_multiplier == 1.0)
        & (rows.variable_cost_multiplier == 1.0)
    ].copy()
    lookup = {
        (int(row.seed), row.topology, row.method): row
        for row in baseline.itertuples(index=False)
    }
    patterns = {"regular": "uniform", "multi_hotspot": "hotspots", "asymmetric": "hotspots"}
    environments = {}
    outputs = []
    for row in correction.to_dict("records"):
        key = (int(row["seed"]), row["topology"])
        if key not in environments:
            environments[key] = build_environment(
                key[0], topology=key[1], profile="moderate",
                demand_pattern=patterns[key[1]], compute_ratio=3.0, throughput_ratio=3.0,
            )
        environment = environments[key]
        costs = dict(zip(environment.instance.site_ids, environment.instance.fixed_cost))
        reference = lookup[(key[0], key[1], "Current feasibility")]
        catalog = lookup[(key[0], key[1], "Catalog + geography")]
        initial = set(json.loads(row["selected_sites_initial"]))
        final = set(json.loads(row["selected_sites_final"]))
        removed = initial - final
        added = final - initial
        retained = initial & final
        initial_fixed = float(catalog.fixed_cost)
        removed_fixed = float(sum(costs[site] for site in removed))
        refunded = float(row["refund_fraction"] * removed_fixed)
        pre_variable = float(row["detection_fraction"] * catalog.variable_cost)
        pre_revenue = float(row["detection_fraction"] * catalog.revenue)
        remaining_budget = float(
            row["budget"] - (1.0 - row["refund_fraction"]) * initial_fixed - pre_variable
        )
        row.update({
            "retained_sites": len(retained),
            "initial_fixed_commitment": initial_fixed,
            "refunded_fixed_cost": refunded,
            "remaining_budget_at_correction": max(0.0, remaining_budget),
            "pre_detection_revenue": pre_revenue,
            "post_detection_revenue": float(row["revenue"] - pre_revenue),
            "pre_detection_variable_cost": pre_variable,
            "post_detection_variable_cost": float(row["variable_cost"] - pre_variable),
            "reference_profit": float(reference.profit),
            "reference_revenue": float(reference.revenue),
            "reference_fixed_cost": float(reference.fixed_cost),
            "reference_variable_cost": float(reference.variable_cost),
            "reference_total_expenditure": float(reference.total_expenditure),
            "reference_qualifying_coverage": float(reference.qualifying_coverage),
            "reference_sites": int(reference.sites),
        })
        row["profit_advantage_from_early_information"] = reference.profit - row["profit"]
        row["revenue_difference"] = reference.revenue - row["revenue"]
        row["fixed_cost_difference"] = reference.fixed_cost - row["fixed_cost"]
        row["variable_cost_difference"] = reference.variable_cost - row["variable_cost"]
        row["accounting_residual"] = (
            row["profit_advantage_from_early_information"] - row["revenue_difference"]
            + row["fixed_cost_difference"] + row["variable_cost_difference"]
        )
        outputs.append(row)
    result = pd.DataFrame(outputs).sort_values(
        ["seed", "topology", "detection_fraction", "refund_fraction"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    residual = float(result.accounting_residual.abs().max())
    if residual > 1e-7:
        raise AssertionError(f"Correction accounting does not reconcile: {residual}")
    decision = pd.read_csv(args.output.parent / "decision_pairs.csv")
    manifest = {
        "decision_pairs": int((decision.kind == "decision").sum()),
        "overlap_pairs": int((decision.kind == "overlap").sum()),
        "correction_conditions": len(result),
        "maximum_decision_accounting_residual": float(decision.accounting_residual.abs().max()),
        "maximum_correction_accounting_residual": residual,
        "correction_source": str(args.hardening),
        "worked_example_rule": (
            "nearest positive middle-budget profit gain to the median across frozen Experiment 1"
        ),
    }
    (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(result)} correction conditions; max residual={residual:.3e}")


if __name__ == "__main__":
    main()
