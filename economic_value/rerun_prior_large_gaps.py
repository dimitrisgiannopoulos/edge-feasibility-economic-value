"""Rerun prior-informed outcomes whose exact-solver gap is not useful."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .environment import build_environment, resample_support
from .model import ProfitModel, ScenarioAverageModel, greedy_local_search


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "prior_stability_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-threshold", type=float, default=0.05)
    parser.add_argument("--time-limit", type=float, default=1200.0)
    args = parser.parse_args()
    patterns = {"regular": "uniform", "multi_hotspot": "hotspots", "asymmetric": "hotspots"}
    targets = []
    for path in sorted((OUTPUT / "checkpoints").glob("*.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            if float(row["mip_gap"] or 0) > args.gap_threshold:
                targets.append((path, row))
    print(f"rerunning {len(targets)} prior outcomes at {args.time_limit:g} seconds", flush=True)
    for completed, (path, target) in enumerate(targets, start=1):
        seed = int(target["seed"])
        topology = str(target["topology"])
        count = int(target["sample_count"])
        environment = build_environment(
            seed, topology=topology, profile="moderate", demand_pattern=patterns[topology],
            compute_ratio=3.0, throughput_ratio=3.0,
        )
        budget = float(environment.metadata["budget_levels"][2])
        actual_instance = environment.instance.with_budget(budget)
        actual_model = ProfitModel(actual_instance)
        informed = greedy_local_search(actual_model)
        sampled = np.stack([
            resample_support(environment, 90_000_007 + environment.seed * 101 + index)
            for index in range(count)
        ])
        model = ScenarioAverageModel(actual_instance, sampled)
        selected, expected, status, gap, seconds = model.solve_exact(
            time_limit=args.time_limit
        )
        realized = actual_model.solve_fixed(selected)
        replacement = dict(target)
        replacement.update({
            "expected_profit": expected,
            "realized_profit": realized.metrics["profit"],
            "reference_profit": informed.metrics["profit"],
            "difference_from_current": informed.metrics["profit"] - realized.metrics["profit"],
            "revenue_normalized_profit": realized.metrics["profit"] / actual_instance.max_revenue,
            "revenue_normalized_difference_from_current": (
                informed.metrics["profit"] - realized.metrics["profit"]
            ) / actual_instance.max_revenue,
            "qualifying_coverage": realized.metrics["qualifying_coverage"],
            "total_expenditure": realized.metrics["total_expenditure"],
            "sites": realized.metrics["sites"],
            "selection_seconds": seconds,
            "status": status,
            "mip_gap": gap,
        })
        rows = json.loads(path.read_text())
        rows = [replacement if int(row["sample_count"]) == count else row for row in rows]
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2) + "\n")
        temporary.replace(path)
        print(f"completed {completed}/{len(targets)}: status={status}, gap={gap}", flush=True)


if __name__ == "__main__":
    main()
