"""Run composition-controlled cross-operator support experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .environment import build_composition_controlled_overlap_environment, catalog_support
from .model import ProfitModel, greedy_local_search
from .run_diagnostics import _composition, _pair


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "results" / "overlap_control_v1"
DEFAULT_CALIBRATION = ROOT / "config" / "catalog_delay_model.json"


def _evaluate(task: dict[str, Any], calibration: dict[str, float]) -> dict[str, Any]:
    environment = build_composition_controlled_overlap_environment(
        task["seed"], task["overlap"], task["support_pattern"]
    )
    budget = float(environment.metadata["budget_levels"][2])
    row, site_rows = _pair(environment, budget, calibration)
    actual = environment.instance.with_budget(budget)
    actual_model = ProfitModel(actual)
    catalog_model = ProfitModel(actual.with_support(catalog_support(environment, calibration)))
    exact_time_limit = float(task.get("exact_time_limit", 180.0))
    catalog_exact_plan = catalog_model.solve_exact(time_limit=exact_time_limit)
    catalog_exact_actual = actual_model.solve_fixed(catalog_exact_plan.selected)
    informed_exact = actual_model.solve_exact(time_limit=exact_time_limit)
    greedy_informed = greedy_local_search(actual_model)
    row.update({
        **task,
        **_composition(environment, environment.resolved_support),
        "catalog_exact_profit": catalog_exact_actual.metrics["profit"],
        "informed_exact_profit": informed_exact.metrics["profit"],
        "exact_information_points": 100 * (
            informed_exact.metrics["profit"] - catalog_exact_actual.metrics["profit"]
        ) / actual.max_revenue,
        "informed_greedy_gap_points": 100 * (
            informed_exact.metrics["profit"] - greedy_informed.metrics["profit"]
        ) / actual.max_revenue,
        "catalog_exact_status": catalog_exact_plan.status,
        "informed_exact_status": informed_exact.status,
        "catalog_exact_gap": catalog_exact_plan.mip_gap,
        "informed_exact_gap": informed_exact.mip_gap,
    })
    for entry in site_rows:
        entry.update(case_id=task["case_id"], overlap=task["overlap"],
                     support_pattern=task["support_pattern"])
    return {"pair": row, "sites": site_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--replicas", type=int, default=20)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        {"seed": 31_000 + offset, "overlap": overlap, "support_pattern": pattern,
         "case_id": f"controlled:{overlap}:{pattern}:r{offset:03d}"}
        for offset in range(args.replicas)
        for overlap in ("aligned", "intermediate", "disjoint")
        for pattern in ("clustered", "dispersed")
    ]
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    identity = {
        "protocol": "overlap_control_v1",
        "implementation_sha256": hashlib.sha256(
            b"".join((ROOT / name).read_bytes() for name in
                     ("environment.py", "model.py", "run_diagnostics.py",
                      "run_overlap_control.py"))
        ).hexdigest(),
        "replicas": args.replicas,
    }
    checkpoint_manifest = args.output / "checkpoint_manifest.json"
    if checkpoint_manifest.exists() and json.loads(checkpoint_manifest.read_text()) != identity:
        raise RuntimeError("Existing overlap-control checkpoints use different inputs")
    if not checkpoint_manifest.exists():
        checkpoint_manifest.write_text(json.dumps(identity, indent=2) + "\n")

    def checkpoint(task: dict[str, Any]) -> Path:
        return checkpoints / (task["case_id"].replace(":", "__") + ".json")

    pending = [task for task in tasks if not checkpoint(task).exists()]
    print(f"loaded {len(tasks) - len(pending)} checkpoints; {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate, task, calibration): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            result = future.result()
            target = checkpoint(task)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(target)
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(f"completed {completed}/{len(pending)} pending overlap cases", flush=True)
    pairs, sites = [], []
    for task in tasks:
        result = json.loads(checkpoint(task).read_text())
        pairs.append(result["pair"])
        sites.extend(result["sites"])
    pairs.sort(key=lambda row: row["case_id"])
    sites.sort(key=lambda row: (row["case_id"], row["site_id"]))
    _write_csv(args.output / "pairs.csv", pairs)
    _write_csv(args.output / "selected_site_diagnostics.csv", sites)
    manifest = {
        "cases": len(pairs), "outcome_selection_runs": 4 * len(pairs),
        "replicas_per_cell": args.replicas,
        "tier_composition_per_group": {"LOCAL": 4, "REGIONAL": 1, "CENTRAL": 1},
        "normalized_within_tier": ["fixed_cost", "compute_capacity", "interface_capacity",
                                   "contracted_quota", "base_variable_cost"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote controlled overlap results to {args.output}")


if __name__ == "__main__":
    main()
