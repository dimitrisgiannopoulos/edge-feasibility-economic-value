"""Add capacity-utilization and affordability diagnostics to saved outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .environment import build_environment


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results" / "hardening_v1" / "outcomes.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "hardening_diagnostics_v1")
    args = parser.parse_args()
    rows = pd.read_csv(args.input)
    patterns = {"regular": "uniform", "multi_hotspot": "hotspots", "asymmetric": "hotspots"}
    cache = {}
    augmented = []
    for row in rows.to_dict("records"):
        if int(row["experiment"]) not in {3, 4}:
            augmented.append(row)
            continue
        key = (int(row["seed"]), row["topology"], float(row["compute_ratio"]),
               float(row["throughput_ratio"]))
        if key not in cache:
            cache[key] = build_environment(
                key[0], topology=key[1], profile="moderate",
                demand_pattern=patterns[key[1]], compute_ratio=key[2], throughput_ratio=key[3],
            )
        environment = cache[key]
        selected_ids = set(json.loads(row["selected_sites"]))
        selected = np.asarray([site in selected_ids for site in environment.instance.site_ids])
        selected_compute = float(environment.instance.compute_capacity[selected].sum())
        selected_interface = float(environment.instance.interface_capacity[selected].sum())
        scaled_fixed = environment.instance.fixed_cost * float(row["fixed_cost_multiplier"])
        row.update({
            "no_deployment": int(not selected.any()),
            "individually_affordable_sites": int((scaled_fixed <= float(row["budget"])).sum()),
            "selected_compute_capacity": selected_compute,
            "selected_interface_capacity": selected_interface,
            "compute_utilization": (
                float(row["compute_used"]) / selected_compute if selected_compute else 0.0
            ),
            "interface_utilization": (
                float(row["throughput_used"]) / selected_interface if selected_interface else 0.0
            ),
        })
        augmented.append(row)
    result = pd.DataFrame(augmented)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "outcomes.csv", index=False)
    for experiment in (3, 4, 5):
        result[result.experiment == experiment].to_csv(
            args.output / f"experiment{experiment}_outcomes.csv", index=False
        )
    manifest = {
        "source": str(args.input),
        "rows": len(result),
        "augmented_experiment3_rows": int((result.experiment == 3).sum()),
        "augmented_experiment4_rows": int((result.experiment == 4).sum()),
        "derived_fields": ["no_deployment", "individually_affordable_sites",
                           "selected_compute_capacity", "selected_interface_capacity",
                           "compute_utilization", "interface_utilization"],
        "note": "No planner was rerun; fields are exact deterministic post-processing.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
