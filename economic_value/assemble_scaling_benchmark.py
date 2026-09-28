"""LEGACY: assemble the superseded, fixed-total-demand scaling benchmark.

The authoritative benchmark is generated directly by ``run_robustness`` with
``robustness_scaling_v3.json`` and written to ``results/scaling_clean_v5``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source = ROOT / "results" / "robustness_v1" / "checkpoints"
    output = ROOT / "results" / "scaling_v1"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    expected = []
    for topology in ("regular", "multi_hotspot", "asymmetric"):
        for areas, sites in ((24, 48), (100, 200), (250, 500)):
            name = f"scaling__{topology}__a{areas}__s{sites}__r000.json"
            expected.append(name)
            path = source / name
            if not path.exists():
                raise RuntimeError(f"Missing completed scaling checkpoint: {path}")
            rows.extend(json.loads(path.read_text()))
    fields = sorted({key for row in rows for key in row})
    with (output / "scaling_outcomes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["case_id"], row["method"])))
    manifest = {
        "cases": 9,
        "outcomes": 18,
        "replication": "one independent environment per topology and size",
        "purpose": "computational scaling benchmark, not statistical economic inference",
        "completed_checkpoints": expected,
        "note": (
            "Additional 500-site seeds exceeded 50 minutes during exhaustive swap search; "
            "the final paper should report this as a heuristic scalability boundary."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
