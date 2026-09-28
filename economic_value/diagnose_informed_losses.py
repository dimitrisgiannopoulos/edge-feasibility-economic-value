"""Reconstruct informed-greedy losses and compare with exact site selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .environment import build_environment
from .model import ProfitModel


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", type=Path,
        default=ROOT / "results/final_decision_diagnostics_v1/decision_pairs.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/evidence_summary_v1/informed_loss_diagnostics.csv",
    )
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs)
    losses = pairs[(pairs.kind == "decision") & (pairs.profit_difference < -1e-7)]
    records = []
    for pair in losses.itertuples(index=False):
        environment = build_environment(
            int(pair.seed), topology=pair.topology, profile="moderate",
            demand_pattern="uniform" if pair.topology == "regular" else "hotspots",
            compute_ratio=3.0, throughput_ratio=3.0,
        )
        budget = float(environment.metadata["budget_levels"][int(pair.budget_index)])
        model = ProfitModel(environment.instance.with_budget(budget))
        ids = environment.instance.site_ids
        catalog_ids = set(json.loads(pair.catalog_selected_site_ids))
        informed_ids = set(json.loads(pair.informed_selected_site_ids))
        catalog = model.solve_fixed([site in catalog_ids for site in ids])
        informed = model.solve_fixed([site in informed_ids for site in ids])
        exact = model.solve_exact(time_limit=120.0)
        if abs(catalog.metrics["profit"] - pair.catalog_realized_profit) > 1e-5:
            raise AssertionError(f"Catalog replay disagrees for {pair.case_id}")
        if abs(informed.metrics["profit"] - pair.informed_realized_profit) > 1e-5:
            raise AssertionError(f"Informed replay disagrees for {pair.case_id}")
        if exact.metrics["profit"] + 1e-5 < catalog.metrics["profit"]:
            raise AssertionError(f"Exact incumbent is worse than catalog for {pair.case_id}")
        records.append({
            "case_id": pair.case_id,
            "topology": pair.topology,
            "budget_index": pair.budget_index,
            "catalog_actual_profit": catalog.metrics["profit"],
            "informed_greedy_profit": informed.metrics["profit"],
            "exact_informed_profit": exact.metrics["profit"],
            "catalog_minus_greedy": catalog.metrics["profit"] - informed.metrics["profit"],
            "exact_minus_greedy": exact.metrics["profit"] - informed.metrics["profit"],
            "exact_minus_catalog": exact.metrics["profit"] - catalog.metrics["profit"],
            "exact_status": exact.status,
            "exact_mip_gap": exact.mip_gap,
            "exact_seconds": exact.seconds,
            "catalog_sites": json.dumps(sorted(catalog_ids)),
            "informed_greedy_sites": json.dumps(sorted(informed_ids)),
            "exact_sites": json.dumps([site for site, active in zip(ids, exact.selected) if active]),
        })
        print(f"{pair.case_id}: exact={exact.status}, gain={records[-1]['exact_minus_greedy']:.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
