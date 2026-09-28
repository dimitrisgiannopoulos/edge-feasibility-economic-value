"""Fit the catalog planner's distance-to-delay rule on development environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .environment import build_environment, _distance


def calibrate(seed_start: int = 10_000, seeds: int = 30) -> dict[str, float]:
    rows, targets = [], []
    for seed in range(seed_start, seed_start + seeds):
        topology = ("regular", "multi_hotspot", "asymmetric")[seed % 3]
        environment = build_environment(seed, topology=topology)
        for path in environment.paths:
            if not path["allowed"] or path["delay_ms"] is None:
                continue
            area = environment.areas[path["area"]]
            site = environment.sites[path["site"]]
            distance = _distance((area["x"], area["y"]), (site["x"], site["y"]))
            rows.append([1.0, distance, site["tier"] == "REGIONAL",
                         site["tier"] == "CENTRAL"])
            targets.append(path["delay_ms"])
    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    return {"intercept": float(coefficients[0]), "distance": float(coefficients[1]),
            "tier_local": 0.0, "tier_regional": float(coefficients[2]),
            "tier_central": float(coefficients[3]),
            "development_seed_start": seed_start, "development_seeds": seeds,
            "observations": len(target),
            "rmse_ms": float(np.sqrt(np.mean((predicted - target) ** 2))),
            "method": "ordinary least squares over finite synthetic path delays",
            "uses_evaluation_environments": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "config" / "catalog_delay_model.json")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = calibrate(args.seed_start, args.seeds)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
