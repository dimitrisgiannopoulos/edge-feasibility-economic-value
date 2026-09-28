"""Run the independently seeded principal information-value evaluation."""

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

from .environment import build_environment
from .hardening import evaluate_greedy_information_pair


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "final_v1.json"
DEFAULT_CALIBRATION = ROOT / "config" / "catalog_delay_model.json"
DEFAULT_OUTPUT = ROOT.parent / "results" / "final_v1"


def _evaluate(task: dict[str, Any], calibration: dict[str, float]) -> list[dict[str, Any]]:
    environment = build_environment(
        task["seed"], topology=task["topology"], profile=task["profile"],
        demand_pattern=task["demand_pattern"], compute_ratio=task["compute_ratio"],
        throughput_ratio=task["throughput_ratio"],
    )
    outputs = []
    for budget_index in task["budget_indices"]:
        budget = float(environment.metadata["budget_levels"][budget_index])
        rows = evaluate_greedy_information_pair(
            environment, environment.instance.with_budget(budget), calibration
        )
        case_id = f"final:{task['topology']}:b{budget_index}:r{task['offset']:03d}"
        for row in rows:
            row.update(case_id=case_id, environment_id=task["environment_id"],
                       seed=task["seed"], topology=task["topology"],
                       budget_index=budget_index, budget=budget)
            outputs.append(row)
    return outputs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--replicas", type=int)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    calibration = json.loads(args.calibration.read_text())
    count = args.replicas or config["replicas"]
    tasks = []
    for offset in range(count):
        for topology_index, topology in enumerate(config["topologies"]):
            tasks.append({
                "offset": offset,
                "seed": config["seed_start"] + 10_000 * topology_index + offset,
                "topology": topology, "profile": config["profile"],
                "demand_pattern": config["demand_patterns"][topology],
                "compute_ratio": config["compute_ratio"],
                "throughput_ratio": config["throughput_ratio"],
                "budget_indices": config["budget_indices"],
                "environment_id": f"final:{topology}:r{offset:03d}",
            })
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    identity = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            b"".join((ROOT / name).read_bytes() for name in
                     ("environment.py", "model.py", "hardening.py", "run_final_evaluation.py"))
        ).hexdigest(),
    }
    checkpoint_manifest = args.output / "checkpoint_manifest.json"
    if checkpoint_manifest.exists() and json.loads(checkpoint_manifest.read_text()) != identity:
        raise RuntimeError("Existing final checkpoints use different inputs")
    if not checkpoint_manifest.exists():
        checkpoint_manifest.write_text(json.dumps(identity, indent=2) + "\n")

    def path(task: dict[str, Any]) -> Path:
        return checkpoints / (task["environment_id"].replace(":", "__") + ".json")

    pending = [task for task in tasks if not path(task).exists()]
    print(f"loaded {len(tasks) - len(pending)} checkpoints; {len(pending)} pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate, task, calibration): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            target = path(task)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(future.result(), indent=2) + "\n")
            temporary.replace(target)
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(f"completed {completed}/{len(pending)} environments", flush=True)
    rows = []
    for task in tasks:
        rows.extend(json.loads(path(task).read_text()))
    rows.sort(key=lambda row: (row["case_id"], row["method"]))
    _write_csv(args.output / "outcomes.csv", rows)
    pairs = []
    for index in range(0, len(rows), 2):
        methods = {row["method"]: row for row in rows[index:index + 2]}
        catalog = methods["Catalog + geography"]
        informed = methods["Current feasibility"]
        pairs.append({
            "case_id": informed["case_id"], "environment_id": informed["environment_id"],
            "seed": informed["seed"], "topology": informed["topology"],
            "budget_index": informed["budget_index"], "budget": informed["budget"],
            "max_revenue": informed["max_revenue"],
            "profit_difference": informed["profit"] - catalog["profit"],
            "revenue_normalized_profit_difference": (
                informed["profit"] - catalog["profit"]
            ) / informed["max_revenue"],
            "coverage_difference": informed["qualifying_coverage"] - catalog["qualifying_coverage"],
            "spending_difference": informed["total_expenditure"] - catalog["total_expenditure"],
            "revenue_difference": informed["revenue"] - catalog["revenue"],
            "fixed_cost_difference": informed["fixed_cost"] - catalog["fixed_cost"],
            "variable_cost_difference": informed["variable_cost"] - catalog["variable_cost"],
        })
    _write_csv(args.output / "paired_contrasts.csv", pairs)
    manifest = {**identity, "config": config, "environments": len(tasks),
                "paired_cases": len(pairs), "outcomes": len(rows),
                "note": "Independent final seeds; baseline calibration and generator frozen."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(pairs)} paired cases to {args.output}")


if __name__ == "__main__":
    main()
