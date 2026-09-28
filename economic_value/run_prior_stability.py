"""Evaluate stability of the prior-informed benchmark across nested samples."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy

from .environment import build_environment, resample_support
from .model import ProfitModel, ScenarioAverageModel, greedy_local_search


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "prior_stability_v1.json"
DEFAULT_OUTPUT = ROOT.parent / "results" / "prior_stability_v1"


def _evaluate(task: dict[str, Any]) -> list[dict[str, Any]]:
    environment = build_environment(
        task["seed"],
        topology=task["topology"],
        profile=task["profile"],
        demand_pattern=task["demand_pattern"],
        compute_ratio=task["compute_ratio"],
        throughput_ratio=task["throughput_ratio"],
    )
    budget = float(environment.metadata["budget_levels"][task["budget_index"]])
    actual_instance = environment.instance.with_budget(budget)
    actual_model = ProfitModel(actual_instance)
    informed = greedy_local_search(actual_model)
    maximum_count = max(task["sample_counts"])
    sampled = np.stack([
        resample_support(environment, 90_000_007 + environment.seed * 101 + index)
        for index in range(maximum_count)
    ])
    rows = []
    for count in task["sample_counts"]:
        model = ScenarioAverageModel(actual_instance, sampled[:count])
        selected, expected, status, gap, seconds = model.solve_exact(
            time_limit=float(task["solver_time_limit_seconds"])
        )
        realized = actual_model.solve_fixed(selected)
        rows.append({
            "case_id": task["case_id"],
            "environment_id": task["environment_id"],
            "seed": task["seed"],
            "topology": task["topology"],
            "profile": task["profile"],
            "sample_count": count,
            "budget": budget,
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
            "sample_seed_formula": "90000007 + environment_seed * 101 + sample_index",
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--replicas", type=int, help="Override replica count for a smoke run")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    count = args.replicas or int(config["replicas"])
    tasks = []
    profiles = config.get("profiles", [config.get("profile", "moderate")])
    for offset in range(count):
        seed = int(config["seed_start"]) + offset
        for topology in config["topologies"]:
            for profile in profiles:
                tasks.append({
                    "seed": seed,
                    "topology": topology,
                    "profile": profile,
                    "demand_pattern": config["demand_patterns"][topology],
                    "compute_ratio": config["compute_ratio"],
                    "throughput_ratio": config["throughput_ratio"],
                    "budget_index": config["budget_index"],
                    "sample_counts": config["sample_counts"],
                    "solver_time_limit_seconds": config.get("solver_time_limit_seconds", 120.0),
                    "case_id": f"prior:{topology}:{profile}:r{offset:03d}",
                    "environment_id": f"prior:{topology}:{profile}:r{offset:03d}",
                })
    implementation = b"".join(
        (ROOT / name).read_bytes()
        for name in ("environment.py", "model.py", "run_prior_stability.py")
    )
    identity = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = args.output / "checkpoint_manifest.json"
    if checkpoint_manifest.exists() and json.loads(checkpoint_manifest.read_text()) != identity:
        raise RuntimeError("Existing checkpoints were created with different inputs")
    if not checkpoint_manifest.exists():
        checkpoint_manifest.write_text(json.dumps(identity, indent=2) + "\n")

    def checkpoint_path(case_id: str) -> Path:
        return checkpoints / (case_id.replace(":", "__") + ".json")

    pending = [task for task in tasks if not checkpoint_path(task["case_id"]).exists()]
    print(f"loaded {len(tasks) - len(pending)} checkpoints; {len(pending)} tasks pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate, task): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                rows = future.result()
            except Exception as error:
                raise RuntimeError(f"Task failed: {task['case_id']}") from error
            target = checkpoint_path(task["case_id"])
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows, indent=2) + "\n")
            temporary.replace(target)
            print(f"completed {completed}/{len(pending)} pending tasks", flush=True)

    rows = []
    for task in tasks:
        rows.extend(json.loads(checkpoint_path(task["case_id"]).read_text()))
    rows.sort(key=lambda row: (row["case_id"], row["sample_count"]))
    _write_csv(args.output / "outcomes.csv", rows)
    manifest = {
        **identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": len(tasks),
        "outcome_rows": len(rows),
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} outcomes to {args.output}")


if __name__ == "__main__":
    main()
