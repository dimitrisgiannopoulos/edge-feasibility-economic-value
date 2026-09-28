"""Run the frozen pilot for information value and operator fragmentation.

This module writes numerical outputs only. Plotting is deliberately separate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
from scipy.stats import t

from .environment import build_environment, build_overlap_environment
from .planners import evaluate_methods


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "pilot_v1.json"
DEFAULT_CALIBRATION = ROOT / "config" / "catalog_delay_model.json"
DEFAULT_OUTPUT = ROOT.parent / "results" / "pilot_v1"


def _overlap_score(environment) -> float:
    """Mean pairwise Jaccard overlap of MNO support sets within each area."""
    n_mnos = int(environment.metadata["n_mnos"])
    values = []
    for area in range(len(environment.areas)):
        supports = [set(np.flatnonzero(environment.resolved_support[area * n_mnos + m]))
                    for m in range(n_mnos)]
        for left in range(n_mnos):
            for right in range(left + 1, n_mnos):
                union = supports[left] | supports[right]
                values.append(len(supports[left] & supports[right]) / len(union) if union else 1.0)
    return float(np.mean(values))


def _evaluate(task: dict[str, Any], calibration: dict[str, float]) -> list[dict[str, Any]]:
    if task["experiment"] == 1:
        environment = build_environment(
            task["seed"], topology=task["topology"], profile="moderate",
            demand_pattern=task["demand_pattern"], compute_ratio=3.0,
            throughput_ratio=3.0,
        )
    else:
        environment = build_overlap_environment(
            task["seed"], task["overlap"], task["support_pattern"],
            alternatives=task["alternatives"],
        )
    budget_index = task["budget_index"]
    budget = float(environment.metadata["budget_levels"][budget_index])
    rows = evaluate_methods(environment, budget, calibration, task["prior_scenarios"])
    raw_counts = {status: int((environment.raw_status == status).sum())
                  for status in ("VIABLE", "PARTIAL", "UNKNOWN", "UNVIABLE")}
    common = {
        "experiment": task["experiment"], "case_id": task["case_id"],
        "environment_id": task["environment_id"], "seed": task["seed"],
        "topology": environment.topology, "budget_index": budget_index,
        "budget": budget, "support_density": float(environment.resolved_support.mean()),
        "mean_supported_alternatives": float(environment.resolved_support.sum(axis=1).mean()),
        "operator_support_overlap": _overlap_score(environment),
        **{f"status_{key.lower()}": value for key, value in raw_counts.items()},
        "study": task.get("study", "information_vs_optimization"),
        "overlap": task.get("overlap", "natural"),
        "support_pattern": task.get("support_pattern", task.get("demand_pattern", "")),
        "alternatives": task.get("alternatives", ""),
    }
    for row in rows:
        row.update(common)
    return rows


def _tasks(config: dict[str, Any], experiment: str, replicas: int | None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if experiment in {"1", "all"}:
        spec = config["experiment_1"]
        count = replicas or spec["replicas"]
        for offset in range(count):
            seed = spec["seed_start"] + offset
            for topology in spec["topologies"]:
                for budget_index in spec["budget_indices"]:
                    case = f"exp1:{topology}:b{budget_index}:r{offset:03d}"
                    tasks.append({
                        "experiment": 1, "seed": seed, "topology": topology,
                        "demand_pattern": spec["demand_patterns"][topology],
                        "budget_index": budget_index, "prior_scenarios": spec["prior_scenarios"],
                        "case_id": case,
                        "environment_id": f"exp1:{topology}:r{offset:03d}",
                    })
    if experiment in {"2", "all"}:
        spec = config["experiment_2"]
        count = replicas or spec["replicas"]
        combinations = {
            ("overlap", overlap, pattern, spec["matched_alternatives"])
            for overlap in spec["overlap_levels"] for pattern in spec["support_patterns"]
        }
        combinations |= {
            ("scarcity", spec["scarcity_overlap"], pattern, alternatives)
            for pattern in spec["support_patterns"]
            for alternatives in spec["scarcity_alternatives"]
            if alternatives != spec["matched_alternatives"]
        }
        for offset in range(count):
            seed = spec["seed_start"] + offset
            for study, overlap, pattern, alternatives in sorted(combinations):
                case = f"exp2:{study}:{overlap}:{pattern}:k{alternatives}:r{offset:03d}"
                tasks.append({
                    "experiment": 2, "seed": seed, "study": study, "overlap": overlap,
                    "support_pattern": pattern, "alternatives": alternatives,
                    "budget_index": spec["budget_index"],
                    "prior_scenarios": spec["prior_scenarios"], "case_id": case,
                    "environment_id": f"exp2:{study}:{overlap}:{pattern}:k{alternatives}:r{offset:03d}",
                })
    return tasks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["case_id"]][row["method"]] = row
    definitions = {
        "information_greedy": ("G-EDFS", "G-Catalog"),
        "information_milp": ("M-EDFS", "M-Catalog"),
        "optimization_with_information": ("M-EDFS", "G-EDFS"),
        "instance_information_vs_prior": ("M-EDFS", "M-Prior"),
    }
    outputs = []
    for case_id, methods in grouped.items():
        if set(methods) != {"G-Catalog", "M-Catalog", "M-Prior", "G-EDFS", "M-EDFS"}:
            raise AssertionError(f"Incomplete method set for {case_id}")
        exemplar = methods["M-EDFS"]
        for name, (treatment, comparison) in definitions.items():
            left, right = methods[treatment], methods[comparison]
            outputs.append({
                "case_id": case_id, "experiment": exemplar["experiment"],
                "environment_id": exemplar["environment_id"], "seed": exemplar["seed"],
                "topology": exemplar["topology"], "budget_index": exemplar["budget_index"],
                "budget": exemplar["budget"], "study": exemplar["study"],
                "overlap": exemplar["overlap"],
                "support_pattern": exemplar["support_pattern"],
                "alternatives": exemplar["alternatives"], "contrast": name,
                "treatment": treatment, "comparison": comparison,
                "profit_difference": left["realized_profit"] - right["realized_profit"],
                "revenue_normalized_profit_difference": (
                    (left["realized_profit"] - right["realized_profit"]) / left["max_revenue"]
                    if left["max_revenue"] else 0.0
                ),
                "coverage_difference": left["qualifying_coverage"] - right["qualifying_coverage"],
                "spending_difference": left["total_expenditure"] - right["total_expenditure"],
                "stranded_expenditure_difference": (
                    left["stranded_expenditure"] - right["stranded_expenditure"]
                ),
                "site_difference": left["sites"] - right["sites"],
                "multi_operator_site_difference": (
                    left["multi_operator_sites"] - right["multi_operator_sites"]
                ),
                "operator_support_overlap": exemplar["operator_support_overlap"],
                "support_density": exemplar["support_density"],
            })
    return outputs


def _summaries(contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in contrasts:
        if row["experiment"] == 1:
            key = (1, row["topology"], row["budget_index"], "", "", "", row["contrast"])
        else:
            key = (2, "", row["budget_index"], row["study"], row["overlap"],
                   f"{row['support_pattern']}:{row['alternatives']}", row["contrast"])
        groups[key].append(float(row["profit_difference"]))
    outputs = []
    for key, values in sorted(groups.items(), key=str):
        array = np.asarray(values)
        mean = float(array.mean())
        if len(array) > 1:
            half = float(t.ppf(.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array)))
        else:
            half = 0.0
        outputs.append({
            "experiment": key[0], "topology": key[1], "budget_index": key[2],
            "study": key[3], "overlap": key[4], "configuration": key[5],
            "contrast": key[6], "replicas": len(array), "mean": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "median": float(np.median(array)), "q25": float(np.quantile(array, .25)),
            "q75": float(np.quantile(array, .75)), "minimum": float(array.min()),
            "maximum": float(array.max()),
        })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("1", "2", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--replicas", type=int, help="Override replica count for a smoke run")
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    calibration_bytes = args.calibration.read_bytes()
    calibration = json.loads(calibration_bytes)
    implementation = b"".join(
        name.encode() + b"\0" + (ROOT / name).read_bytes()
        for name in ("environment.py", "model.py", "planners.py", "run_pilot.py")
    )
    tasks = _tasks(config, args.experiment, args.replicas)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = args.output / "checkpoint_manifest.json"
    checkpoint_identity = {
        "protocol": config["protocol"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
    }
    if checkpoint_manifest.exists():
        existing = json.loads(checkpoint_manifest.read_text())
        if existing != checkpoint_identity:
            raise RuntimeError("Existing checkpoints were created with different inputs")
    else:
        checkpoint_manifest.write_text(json.dumps(checkpoint_identity, indent=2) + "\n")

    def checkpoint_path(case_id: str) -> Path:
        return checkpoints / (case_id.replace(":", "__") + ".json")

    pending = [task for task in tasks if not checkpoint_path(task["case_id"]).exists()]
    print(f"loaded {len(tasks) - len(pending)} checkpoints; {len(pending)} cases pending", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_evaluate, task, calibration): task for task in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                case_rows = future.result()
            except Exception as error:
                raise RuntimeError(f"Case failed: {task['case_id']}") from error
            target = checkpoint_path(task["case_id"])
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(case_rows, indent=2) + "\n")
            temporary.replace(target)
            if completed == 1 or completed % 10 == 0 or completed == len(pending):
                print(f"completed {completed}/{len(pending)} pending cases", flush=True)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        path = checkpoint_path(task["case_id"])
        if not path.exists():
            raise RuntimeError(f"Missing checkpoint for {task['case_id']}")
        rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda row: (row["case_id"], row["method"]))
    _write_csv(args.output / "outcomes.csv", rows)
    for number in (1, 2):
        selected = [row for row in rows if row["experiment"] == number]
        if selected:
            _write_csv(args.output / f"experiment{number}_outcomes.csv", selected)
    contrasts = _contrasts(rows)
    _write_csv(args.output / "paired_contrasts.csv", contrasts)
    _write_csv(args.output / "contrast_summaries.csv", _summaries(contrasts))
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        revision = "unavailable"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"], "experiment": args.experiment,
        "cases": len(tasks), "outcome_rows": len(rows), "workers": args.workers,
        "config_sha256": checkpoint_identity["config_sha256"],
        "calibration_sha256": checkpoint_identity["calibration_sha256"],
        "implementation_sha256": checkpoint_identity["implementation_sha256"],
        "config": config, "calibration": calibration, "git_revision": revision,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__,
        "note": "Pilot seeds are for range diagnosis and are not the final evaluation sample.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(rows)} outcomes to {args.output}")


if __name__ == "__main__":
    main()
