"""Run application-requirement and large-instance robustness experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import resource
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from .environment import build_environment
from .hardening import evaluate_greedy_information_pair


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "robustness_v1.json"
DEFAULT_CALIBRATION = ROOT / "config" / "catalog_delay_model.json"
DEFAULT_OUTPUT = ROOT.parent / "results" / "robustness_v1"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(task: dict[str, Any], calibration: dict[str, float]) -> list[dict[str, Any]]:
    started = time.perf_counter()
    environment = build_environment(
        task["seed"], topology=task["topology"], n_areas=task["n_areas"],
        n_sites=task["n_sites"], profile=task["profile"],
        demand_pattern=task["demand_pattern"], compute_ratio=task["compute_ratio"],
        throughput_ratio=task["throughput_ratio"],
        total_demand=task["total_demand"],
    )
    budget = float(environment.metadata["budget_levels"][task["budget_index"]])
    rows = evaluate_greedy_information_pair(
        environment, environment.instance.with_budget(budget), calibration,
        time_limit_seconds=task.get("planner_time_limit_seconds"),
    )
    elapsed = time.perf_counter() - started
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    for row in rows:
        row.update({
            **task,
            "budget": budget,
            "case_runtime_seconds": elapsed,
            "worker_peak_rss_mb": rss_mb,
            "no_deployment": int(row["sites"] == 0),
        })
    return rows


def _evaluate_payload(payload: tuple[dict[str, Any], dict[str, float]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one task in a replaceable worker process."""
    task, calibration = payload
    return task, _evaluate(task, calibration)


def _tasks(config: dict[str, Any], experiment: str, replicas: int | None) -> list[dict[str, Any]]:
    tasks = []
    if experiment in {"requirements", "all"}:
        spec = config["application_requirements"]
        count = replicas or spec["replicas"]
        for offset in range(count):
            for topology in config["topologies"]:
                for profile in spec["profiles"]:
                    tasks.append({
                        "experiment": "requirements", "seed": spec["seed_start"] + offset,
                        "topology": topology, "demand_pattern": config["demand_patterns"][topology],
                        "profile": profile, "n_areas": 24, "n_sites": 48,
                        "compute_ratio": config["compute_ratio"],
                        "throughput_ratio": config["throughput_ratio"],
                        "total_demand": 10_000.0,
                        "planner_time_limit_seconds": None,
                        "budget_index": config["budget_index"],
                        "case_id": f"requirements:{topology}:{profile}:r{offset:03d}",
                    })
    if experiment in {"scaling", "all"}:
        spec = config["scaling"]
        count = replicas or spec["replicas"]
        for offset in range(count):
            # Submit the largest independent cases together so the scale-only
            # benchmark does not serialize its dominant runtimes.
            for size in reversed(spec["sizes"]):
                for topology in config["topologies"]:
                    tasks.append({
                        "experiment": "scaling", "seed": spec["seed_start"] + offset,
                        "topology": topology, "demand_pattern": config["demand_patterns"][topology],
                        "profile": spec["profile"], "n_areas": size["areas"],
                        "n_sites": size["sites"],
                        "compute_ratio": config["compute_ratio"],
                        "throughput_ratio": config["throughput_ratio"],
                        # Preserve demand per area so scale does not change the
                        # deployment economics into an empty-footprint test.
                        "total_demand": 10_000.0 * size["areas"] / 24.0,
                        "planner_time_limit_seconds": spec.get("planner_time_limit_seconds"),
                        "budget_index": spec.get("budget_index", config["budget_index"]),
                        "case_id": f"scaling:{topology}:a{size['areas']}:s{size['sites']}:r{offset:03d}",
                    })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("requirements", "scaling", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--fresh-workers", action="store_true",
                        help="Use one fresh process per case for attributable peak-memory results")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    calibration = json.loads(args.calibration.read_text())
    tasks = _tasks(config, args.experiment, args.replicas)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    identity = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            b"".join((ROOT / name).read_bytes() for name in
                     ("environment.py", "model.py", "hardening.py", "run_robustness.py"))
        ).hexdigest(),
    }
    manifest_path = args.output / "checkpoint_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise RuntimeError("Existing robustness checkpoints use different inputs")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n")

    def path(task: dict[str, Any]) -> Path:
        return checkpoints / (task["case_id"].replace(":", "__") + ".json")

    pending = [task for task in tasks if not path(task).exists()]
    print(f"loaded {len(tasks) - len(pending)} checkpoints; {len(pending)} pending", flush=True)
    def checkpoint(task: dict[str, Any], rows: list[dict[str, Any]], completed: int) -> None:
        target = path(task)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2) + "\n")
        temporary.replace(target)
        if completed == 1 or completed % 20 == 0 or completed == len(pending):
            print(f"completed {completed}/{len(pending)} pending cases", flush=True)

    if args.fresh_workers:
        payloads = [(task, calibration) for task in pending]
        with get_context("spawn").Pool(
            processes=args.workers, maxtasksperchild=1
        ) as pool:
            for completed, (task, rows) in enumerate(
                pool.imap_unordered(_evaluate_payload, payloads), start=1
            ):
                checkpoint(task, rows, completed)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_evaluate, task, calibration): task for task in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                checkpoint(task, future.result(), completed)

    rows = []
    for task in tasks:
        rows.extend(json.loads(path(task).read_text()))
    rows.sort(key=lambda row: (row["experiment"], row["case_id"], row["method"]))
    _write_csv(args.output / "outcomes.csv", rows)
    for name in ("requirements", "scaling"):
        selected = [row for row in rows if row["experiment"] == name]
        if selected:
            _write_csv(args.output / f"{name}_outcomes.csv", selected)
    manifest = {**identity, "config": config, "tasks": len(tasks), "outcomes": len(rows)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} outcomes to {args.output}")


if __name__ == "__main__":
    main()
