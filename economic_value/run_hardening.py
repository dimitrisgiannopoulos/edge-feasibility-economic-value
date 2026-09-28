"""Run economic sensitivity, resource sensitivity, and correction experiments.

This module writes numerical outputs only. Plotting remains in a separate
module so presentation changes never require rerunning the experiments.
"""

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

from .environment import build_environment
from .hardening import (
    evaluate_correction_grid,
    evaluate_greedy_information_pair,
    evaluate_informed_from_start,
    scaled_economic_instance,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "hardening_v1.json"
DEFAULT_CALIBRATION = ROOT / "config" / "catalog_delay_model.json"
DEFAULT_OUTPUT = ROOT.parent / "results" / "hardening_v1"


def _base_environment(task: dict[str, Any]):
    return build_environment(
        task["seed"],
        topology=task["topology"],
        profile=task["profile"],
        demand_pattern=task["demand_pattern"],
        compute_ratio=task["compute_ratio"],
        throughput_ratio=task["throughput_ratio"],
    )


def _evaluate(task: dict[str, Any], calibration: dict[str, float]) -> list[dict[str, Any]]:
    environment = _base_environment(task)
    budget = float(environment.metadata["budget_levels"][task["budget_index"]])
    common = {
        "experiment": task["experiment"],
        "case_id": task["case_id"],
        "environment_id": task["environment_id"],
        "seed": task["seed"],
        "topology": task["topology"],
        "budget": budget,
        "budget_index": task["budget_index"],
        "profile": task["profile"],
        "compute_ratio": task["compute_ratio"],
        "throughput_ratio": task["throughput_ratio"],
        "fixed_cost_multiplier": task.get("fixed_cost_multiplier", 1.0),
        "variable_cost_multiplier": task.get("variable_cost_multiplier", 1.0),
    }
    if task["experiment"] in {3, 4}:
        instance = scaled_economic_instance(
            environment,
            budget,
            task.get("fixed_cost_multiplier", 1.0),
            task.get("variable_cost_multiplier", 1.0),
        )
        rows = evaluate_greedy_information_pair(environment, instance, calibration)
        for row in rows:
            row.update(common)
        return rows

    informed = evaluate_informed_from_start(environment, budget)
    rows = evaluate_correction_grid(
        environment,
        budget,
        calibration,
        task["detection_fractions"],
        task["refund_fractions"],
    )
    for row in rows:
        row.update(common)
        row["reference_profit"] = informed["profit"]
        row["reference_coverage"] = informed["qualifying_coverage"]
        row["profit_difference"] = informed["profit"] - row["profit"]
        row["revenue_normalized_profit_difference"] = (
            row["profit_difference"] / row["max_revenue"] if row["max_revenue"] else 0.0
        )
    return rows


def _tasks(config: dict[str, Any], experiment: str, replicas: int | None) -> list[dict[str, Any]]:
    count = replicas or int(config["replicas"])
    tasks: list[dict[str, Any]] = []
    for offset in range(count):
        seed = int(config["seed_start"]) + offset
        for topology in config["topologies"]:
            base = {
                "seed": seed,
                "topology": topology,
                "profile": config["profile"],
                "demand_pattern": config["demand_patterns"][topology],
                "budget_index": config["budget_index"],
            }
            if experiment in {"3", "all"}:
                spec = config["economic_sensitivity"]
                for fixed in spec["fixed_cost_multipliers"]:
                    for variable in spec["variable_cost_multipliers"]:
                        tasks.append({
                            **base,
                            "experiment": 3,
                            "compute_ratio": spec["compute_ratio"],
                            "throughput_ratio": spec["throughput_ratio"],
                            "fixed_cost_multiplier": fixed,
                            "variable_cost_multiplier": variable,
                            "case_id": f"exp3:{topology}:f{fixed}:v{variable}:r{offset:03d}",
                            "environment_id": f"hardening:{topology}:r{offset:03d}",
                        })
            if experiment in {"4", "all"}:
                spec = config["resource_sensitivity"]
                for compute in spec["compute_ratios"]:
                    for throughput in spec["throughput_ratios"]:
                        tasks.append({
                            **base,
                            "experiment": 4,
                            "compute_ratio": compute,
                            "throughput_ratio": throughput,
                            "case_id": f"exp4:{topology}:c{compute}:t{throughput}:r{offset:03d}",
                            "environment_id": f"hardening:{topology}:r{offset:03d}",
                        })
            if experiment in {"5", "all"}:
                spec = config["correction"]
                tasks.append({
                    **base,
                    "experiment": 5,
                    "compute_ratio": spec["compute_ratio"],
                    "throughput_ratio": spec["throughput_ratio"],
                    "detection_fractions": spec["detection_fractions"],
                    "refund_fractions": spec["refund_fractions"],
                    "case_id": f"exp5:{topology}:r{offset:03d}",
                    "environment_id": f"hardening:{topology}:r{offset:03d}",
                })
    return tasks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["experiment"] not in {3, 4}:
            continue
        grouped.setdefault(row["case_id"], {})[row["method"]] = row
    contrasts = []
    for case_id, methods in grouped.items():
        if set(methods) != {"Catalog + geography", "Current feasibility"}:
            raise AssertionError(f"Incomplete information pair for {case_id}")
        informed = methods["Current feasibility"]
        catalog = methods["Catalog + geography"]
        contrast = {
            key: informed[key]
            for key in (
                "experiment", "case_id", "environment_id", "seed", "topology",
                "budget", "budget_index", "profile", "compute_ratio", "throughput_ratio",
                "fixed_cost_multiplier", "variable_cost_multiplier", "max_revenue",
            )
        }
        contrast.update(
            profit_difference=informed["realized_profit"] - catalog["realized_profit"],
            revenue_normalized_profit_difference=(
                (informed["realized_profit"] - catalog["realized_profit"])
                / informed["max_revenue"]
            ),
            coverage_difference=(
                informed["qualifying_coverage"] - catalog["qualifying_coverage"]
            ),
            spending_difference=(
                informed["total_expenditure"] - catalog["total_expenditure"]
            ),
        )
        contrasts.append(contrast)
    return contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("3", "4", "5", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--replicas", type=int, help="Override replica count for smoke runs")
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    calibration_bytes = args.calibration.read_bytes()
    config = json.loads(config_bytes)
    calibration = json.loads(calibration_bytes)
    implementation = b"".join(
        name.encode() + b"\0" + (ROOT / name).read_bytes()
        for name in ("environment.py", "model.py", "hardening.py", "run_hardening.py")
    )
    identity = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
    }
    tasks = _tasks(config, args.experiment, args.replicas)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "checkpoint_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise RuntimeError("Existing checkpoints were created with different inputs")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n")

    def checkpoint_path(case_id: str) -> Path:
        return checkpoints / (case_id.replace(":", "__") + ".json")

    pending = [task for task in tasks if not checkpoint_path(task["case_id"]).exists()]
    print(f"loaded {len(tasks) - len(pending)} checkpoints; {len(pending)} tasks pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate, task, calibration): task for task in pending}
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
            if completed == 1 or completed % 20 == 0 or completed == len(pending):
                print(f"completed {completed}/{len(pending)} pending tasks", flush=True)

    all_rows: list[dict[str, Any]] = []
    for task in tasks:
        all_rows.extend(json.loads(checkpoint_path(task["case_id"]).read_text()))
    all_rows.sort(key=lambda row: (
        row["experiment"], row["case_id"], row.get("detection_fraction", -1),
        row.get("refund_fraction", -1), row.get("method", ""),
    ))
    _write_csv(args.output / "outcomes.csv", all_rows)
    for number in (3, 4, 5):
        selected = [row for row in all_rows if row["experiment"] == number]
        if selected:
            _write_csv(args.output / f"experiment{number}_outcomes.csv", selected)
    contrasts = _pair_contrasts(all_rows)
    if contrasts:
        _write_csv(args.output / "paired_contrasts.csv", contrasts)

    manifest = {
        **identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "tasks": len(tasks),
        "outcome_rows": len(all_rows),
        "workers": args.workers,
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "note": "Hardening pilot. Final evaluation must use independent frozen seeds.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(all_rows)} outcomes to {args.output}")


if __name__ == "__main__":
    main()
