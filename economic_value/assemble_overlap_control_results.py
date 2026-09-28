"""Assemble controlled-overlap checkpoint results after longer-limit retries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "overlap_control_v1"


def _write(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    pairs, sites = [], []
    checkpoints = OUTPUT / "checkpoints"
    for offset in range(20):
        for overlap in ("aligned", "intermediate", "disjoint"):
            for pattern in ("clustered", "dispersed"):
                case = f"controlled:{overlap}:{pattern}:r{offset:03d}"
                path = checkpoints / (case.replace(":", "__") + ".json")
                if not path.exists():
                    raise RuntimeError(f"Missing controlled-overlap checkpoint: {path}")
                result = json.loads(path.read_text())
                pairs.append(result["pair"])
                sites.extend(result["sites"])
    pairs.sort(key=lambda row: row["case_id"])
    sites.sort(key=lambda row: (row["case_id"], row["site_id"]))
    _write(OUTPUT / "pairs.csv", pairs)
    _write(OUTPUT / "selected_site_diagnostics.csv", sites)
    statuses = {
        "catalog_optimal": sum(row["catalog_exact_status"] == "optimal" for row in pairs),
        "informed_optimal": sum(row["informed_exact_status"] == "optimal" for row in pairs),
        "maximum_catalog_gap": max(float(row["catalog_exact_gap"] or 0) for row in pairs),
        "maximum_informed_gap": max(float(row["informed_exact_gap"] or 0) for row in pairs),
    }
    manifest = {
        "cases": len(pairs), "outcome_selection_runs": 4 * len(pairs),
        "replicas_per_cell": 20,
        "tier_composition_per_group": {"LOCAL": 4, "REGIONAL": 1, "CENTRAL": 1},
        "normalized_within_tier": ["fixed_cost", "compute_capacity", "interface_capacity",
                                   "contracted_quota", "base_variable_cost"],
        "exact_solver_status": statuses,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
