"""Assemble the 15-environment longer-limit prior validation subset."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "prior_stability_v3"


def main() -> None:
    rows = []
    expected = []
    for offset in range(5):
        for topology in ("regular", "multi_hotspot", "asymmetric"):
            case = f"prior:{topology}:r{offset:03d}"
            path = OUTPUT / "checkpoints" / (case.replace(":", "__") + ".json")
            expected.append(path.name)
            if not path.exists():
                raise RuntimeError(f"Missing prior-validation checkpoint: {path}")
            rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda row: (row["case_id"], row["sample_count"]))
    fields = sorted({key for row in rows for key in row})
    with (OUTPUT / "outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "environments": 15,
        "environments_per_topology": 5,
        "outcomes": len(rows),
        "sample_counts": [12, 24, 48],
        "solver_time_limit_seconds": 600,
        "optimal_outcomes": sum(row["status"] == "optimal" for row in rows),
        "maximum_mip_gap": max(float(row["mip_gap"] or 0) for row in rows),
        "completed_checkpoints": expected,
        "purpose": "bounded stronger-comparator validation subset",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
