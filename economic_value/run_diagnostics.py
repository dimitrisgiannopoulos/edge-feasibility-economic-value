"""Replay frozen experiments with decision-level and accounting diagnostics.

Numerical diagnostics are written here. Plot generation remains separate so
visual revisions never require rerunning deployment planning.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from .environment import (
    build_environment,
    build_overlap_environment,
    catalog_support,
    save_environment,
)
from .hardening import evaluate_correction_grid, evaluate_informed_from_start
from .model import ProfitModel, Solution, greedy_local_search


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "results" / "diagnostics_v1"
DEFAULT_CALIBRATION = ROOT / "config" / "catalog_delay_model.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_ids(solution: Solution, site_ids: tuple[str, ...]) -> set[str]:
    return {site for site, active in zip(site_ids, solution.selected) if active}


def _operator_service(instance, solution: Solution, mno: int) -> float:
    return float(solution.assignment[instance.group_mno == mno].sum())


def _multi_operator_volume(instance, solution: Solution) -> float:
    total = 0.0
    for site in range(len(instance.site_ids)):
        operators = [
            mno for mno in range(instance.contracted_quota.shape[0])
            if solution.assignment[instance.group_mno == mno, site].sum() > 1e-7
        ]
        if len(operators) >= 2:
            total += float(solution.assignment[:, site].sum())
    return total


def _composition(environment, support: np.ndarray) -> dict[str, float]:
    fixed = environment.instance.fixed_cost
    compute = environment.instance.compute_capacity
    interface = environment.instance.interface_capacity
    tiers = np.asarray(environment.instance.site_tiers)
    counts = support.sum(axis=1)
    if np.any(counts <= 0):
        raise AssertionError("Composition diagnostics require at least one supported site per group")
    return {
        "supported_mean_fixed_cost": float(np.mean((support * fixed).sum(axis=1) / counts)),
        "supported_mean_compute_capacity": float(
            np.mean((support * compute).sum(axis=1) / counts)
        ),
        "supported_mean_interface_capacity": float(
            np.mean((support * interface).sum(axis=1) / counts)
        ),
        **{
            f"supported_{tier.lower()}_share": float(
                np.mean((support[:, tiers == tier].sum(axis=1) / counts))
            )
            for tier in ("LOCAL", "REGIONAL", "CENTRAL")
        },
    }


def _pair(environment, budget: float, calibration: dict[str, float]) -> tuple[dict, list[dict]]:
    actual_instance = environment.instance.with_budget(budget)
    actual_model = ProfitModel(actual_instance)
    estimated_support = catalog_support(environment, calibration)
    catalog_model = ProfitModel(actual_instance.with_support(estimated_support))

    catalog_plan = greedy_local_search(catalog_model)
    catalog_actual = actual_model.solve_fixed(catalog_plan.selected)
    informed = greedy_local_search(actual_model)

    catalog_sites = _selected_ids(catalog_plan, actual_instance.site_ids)
    informed_sites = _selected_ids(informed, actual_instance.site_ids)
    retained = catalog_sites & informed_sites
    added = informed_sites - catalog_sites
    removed = catalog_sites - informed_sites
    union = catalog_sites | informed_sites

    delta_revenue = informed.metrics["revenue"] - catalog_actual.metrics["revenue"]
    delta_fixed = informed.metrics["fixed_cost"] - catalog_actual.metrics["fixed_cost"]
    delta_variable = informed.metrics["variable_cost"] - catalog_actual.metrics["variable_cost"]
    delta_profit = informed.metrics["profit"] - catalog_actual.metrics["profit"]
    row: dict[str, Any] = {
        "budget": budget,
        "footprints_differ": int(catalog_sites != informed_sites),
        "catalog_sites": len(catalog_sites),
        "informed_sites": len(informed_sites),
        "sites_added": len(added),
        "sites_removed": len(removed),
        "sites_retained": len(retained),
        "footprint_jaccard": len(retained) / len(union) if union else 1.0,
        "added_site_ids": json.dumps(sorted(added)),
        "removed_site_ids": json.dumps(sorted(removed)),
        "retained_site_ids": json.dumps(sorted(retained)),
        "catalog_selected_site_ids": json.dumps(sorted(catalog_sites)),
        "informed_selected_site_ids": json.dumps(sorted(informed_sites)),
        "catalog_predicted_profit": catalog_plan.metrics["profit"],
        "catalog_realized_profit": catalog_actual.metrics["profit"],
        "informed_realized_profit": informed.metrics["profit"],
        "profit_difference": delta_profit,
        "revenue_normalized_profit_difference": delta_profit / actual_instance.max_revenue,
        "revenue_difference": delta_revenue,
        "fixed_cost_difference": delta_fixed,
        "variable_cost_difference": delta_variable,
        "accounting_residual": delta_profit - (delta_revenue - delta_fixed - delta_variable),
        "catalog_predicted_service": catalog_plan.metrics["qualifying_demand"],
        "catalog_actual_service": catalog_actual.metrics["qualifying_demand"],
        "catalog_support_overestimate": (
            catalog_plan.metrics["qualifying_demand"]
            - catalog_actual.metrics["qualifying_demand"]
        ),
        "informed_actual_service": informed.metrics["qualifying_demand"],
        "catalog_coverage": catalog_actual.metrics["qualifying_coverage"],
        "informed_coverage": informed.metrics["qualifying_coverage"],
        "catalog_multi_operator_volume": _multi_operator_volume(actual_instance, catalog_actual),
        "informed_multi_operator_volume": _multi_operator_volume(actual_instance, informed),
        "max_revenue": actual_instance.max_revenue,
    }
    for metric in ("revenue", "fixed_cost", "variable_cost", "total_expenditure",
                   "stranded_expenditure", "local_sites", "regional_sites", "central_sites"):
        row[f"catalog_{metric}"] = catalog_actual.metrics[metric]
        row[f"informed_{metric}"] = informed.metrics[metric]
    for mno in range(actual_instance.contracted_quota.shape[0]):
        row[f"catalog_mno_{mno}_service"] = _operator_service(actual_instance, catalog_actual, mno)
        row[f"informed_mno_{mno}_service"] = _operator_service(actual_instance, informed, mno)

    site_rows: list[dict[str, Any]] = []
    for site, site_id in enumerate(actual_instance.site_ids):
        if site_id not in union:
            continue
        entry: dict[str, Any] = {
            "site_id": site_id,
            "tier": actual_instance.site_tiers[site],
            "fixed_cost": actual_instance.fixed_cost[site],
            "compute_capacity": actual_instance.compute_capacity[site],
            "interface_capacity": actual_instance.interface_capacity[site],
            "catalog_selected": int(site_id in catalog_sites),
            "informed_selected": int(site_id in informed_sites),
            "catalog_actual_service": float(catalog_actual.assignment[:, site].sum()),
            "informed_actual_service": float(informed.assignment[:, site].sum()),
            "supported_groups": int(environment.resolved_support[:, site].sum()),
            "catalog_estimated_supported_groups": int(estimated_support[:, site].sum()),
        }
        for mno in range(actual_instance.contracted_quota.shape[0]):
            mask = actual_instance.group_mno == mno
            entry[f"catalog_mno_{mno}_service"] = float(catalog_actual.assignment[mask, site].sum())
            entry[f"informed_mno_{mno}_service"] = float(informed.assignment[mask, site].sum())
        site_rows.append(entry)
    return row, site_rows


def _evaluate_task(task: dict[str, Any], calibration: dict[str, float]) -> dict[str, Any]:
    if task["kind"] == "decision":
        environment = build_environment(
            task["seed"], topology=task["topology"], profile="moderate",
            demand_pattern=task["demand_pattern"], compute_ratio=3.0,
            throughput_ratio=3.0,
        )
    else:
        environment = build_overlap_environment(
            task["seed"], task["overlap"], task["support_pattern"], alternatives=6
        )
    budget = float(environment.metadata["budget_levels"][task["budget_index"]])
    pair, sites = _pair(environment, budget, calibration)
    pair.update({key: value for key, value in task.items() if key != "demand_pattern"})
    if task["kind"] == "overlap":
        pair.update(_composition(environment, environment.resolved_support))
    for row in sites:
        row.update({"case_id": task["case_id"], "kind": task["kind"]})
    return {"pair": pair, "sites": sites}


def _tasks() -> list[dict[str, Any]]:
    tasks = []
    patterns = {"regular": "uniform", "multi_hotspot": "hotspots", "asymmetric": "hotspots"}
    for offset in range(20):
        for topology, pattern in patterns.items():
            for budget_index in range(5):
                tasks.append({
                    "kind": "decision", "seed": 20_000 + offset,
                    "topology": topology, "demand_pattern": pattern,
                    "budget_index": budget_index,
                    "case_id": f"decision:{topology}:b{budget_index}:r{offset:03d}",
                })
        for overlap in ("aligned", "intermediate", "disjoint"):
            for pattern in ("clustered", "dispersed"):
                tasks.append({
                    "kind": "overlap", "seed": 30_000 + offset,
                    "topology": "matched", "overlap": overlap,
                    "support_pattern": pattern, "budget_index": 2,
                    "case_id": f"overlap:{overlap}:{pattern}:r{offset:03d}",
                })
    return tasks


def _correction_case(task: dict[str, Any], calibration: dict[str, float]) -> list[dict[str, Any]]:
    environment = build_environment(
        task["seed"], topology=task["topology"], profile="moderate",
        demand_pattern=task["demand_pattern"], compute_ratio=3.0, throughput_ratio=3.0,
    )
    budget = float(environment.metadata["budget_levels"][2])
    reference = evaluate_informed_from_start(environment, budget)
    corrected = evaluate_correction_grid(
        environment, budget, calibration, [0.0, 0.01, 0.10, 0.25], [0.0, 0.5, 1.0]
    )
    for row in corrected:
        row.update(seed=environment.seed, topology=task["topology"], budget=budget)
        for key in ("profit", "revenue", "fixed_cost", "variable_cost",
                    "total_expenditure", "qualifying_coverage", "sites"):
            row[f"reference_{key}"] = reference[key]
        row["profit_advantage_from_early_information"] = reference["profit"] - row["profit"]
        row["revenue_difference"] = reference["revenue"] - row["revenue"]
        row["fixed_cost_difference"] = reference["fixed_cost"] - row["fixed_cost"]
        row["variable_cost_difference"] = reference["variable_cost"] - row["variable_cost"]
        row["accounting_residual"] = (
            row["profit_advantage_from_early_information"] - row["revenue_difference"]
            + row["fixed_cost_difference"] + row["variable_cost_difference"]
        )
    return corrected


def _correction_rows(calibration: dict[str, float], workers: int) -> list[dict[str, Any]]:
    patterns = {"regular": "uniform", "multi_hotspot": "hotspots", "asymmetric": "hotspots"}
    tasks = [
        {"seed": 40_000 + offset, "topology": topology, "demand_pattern": pattern}
        for offset in range(20) for topology, pattern in patterns.items()
    ]
    outputs = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_correction_case, task, calibration): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            outputs.extend(future.result())
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)} correction environments", flush=True)
    return outputs


def _write_worked_example(output: Path, pair_rows: list[dict[str, Any]],
                           calibration: dict[str, float],
                           selection_scope: str = "all 60 frozen Experiment 1 environments") -> None:
    eligible = [row for row in pair_rows if row["kind"] == "decision"
                and row["budget_index"] == 2 and row["profit_difference"] > 0]
    median = float(np.median([row["profit_difference"] for row in eligible]))
    chosen = min(eligible, key=lambda row: (abs(row["profit_difference"] - median), row["case_id"]))
    pattern = "uniform" if chosen["topology"] == "regular" else "hotspots"
    environment = build_environment(
        int(chosen["seed"]), topology=str(chosen["topology"]), profile="moderate",
        demand_pattern=pattern, compute_ratio=3.0, throughput_ratio=3.0,
    )
    budget = float(environment.metadata["budget_levels"][2])
    actual = environment.instance.with_budget(budget)
    actual_model = ProfitModel(actual)
    catalog_model = ProfitModel(actual.with_support(catalog_support(environment, calibration)))
    catalog_plan = greedy_local_search(catalog_model)
    catalog_actual = actual_model.solve_fixed(catalog_plan.selected)
    informed = greedy_local_search(actual_model)

    directory = output / "worked_example"
    directory.mkdir(parents=True, exist_ok=True)
    save_environment(environment, directory / "environment.json")
    summary = dict(chosen)
    summary["selection_rule"] = (
        "Middle-budget positive case with absolute profit gain nearest the median "
        f"across {selection_scope}."
    )
    summary["median_positive_profit_difference"] = median
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    site_rows = []
    for index, site in enumerate(environment.sites):
        site_rows.append({
            **site,
            "catalog_selected": int(catalog_plan.selected[index]),
            "informed_selected": int(informed.selected[index]),
            "actual_supported_groups": int(environment.resolved_support[:, index].sum()),
            "catalog_estimated_supported_groups": int(
                catalog_model.data.support[:, index].sum()
            ),
            "catalog_actual_service": float(catalog_actual.assignment[:, index].sum()),
            "informed_service": float(informed.assignment[:, index].sum()),
        })
    _write_csv(directory / "sites.csv", site_rows)

    assignment_rows = []
    for group, group_info in enumerate(environment.groups):
        for site, site_id in enumerate(actual.site_ids):
            catalog_value = float(catalog_actual.assignment[group, site])
            informed_value = float(informed.assignment[group, site])
            if max(catalog_value, informed_value) <= 1e-7:
                continue
            assignment_rows.append({
                "group_id": group_info["id"], "area": group_info["area"],
                "mno": group_info["mno"], "site_id": site_id,
                "demand": group_info["demand"],
                "actual_support": int(environment.resolved_support[group, site]),
                "catalog_assignment": catalog_value,
                "informed_assignment": informed_value,
            })
    _write_csv(directory / "assignments.csv", assignment_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--correction-only", action="store_true")
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    if args.correction_only:
        with (args.output / "decision_pairs.csv").open(newline="") as handle:
            pair_rows = list(csv.DictReader(handle))
    else:
        tasks = _tasks()
        pair_rows = []
        site_rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_evaluate_task, task, calibration): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                pair_rows.append(result["pair"])
                site_rows.extend(result["sites"])
                if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)} diagnostic pairs", flush=True)
        pair_rows.sort(key=lambda row: row["case_id"])
        site_rows.sort(key=lambda row: (row["case_id"], row["site_id"]))
        _write_csv(args.output / "decision_pairs.csv", pair_rows)
        _write_csv(args.output / "selected_site_diagnostics.csv", site_rows)
        _write_worked_example(args.output, pair_rows, calibration)

    correction = _correction_rows(calibration, args.workers)
    _write_csv(args.output / "correction_accounting.csv", correction)
    maximum_residual = max(abs(float(row["accounting_residual"])) for row in pair_rows + correction)
    manifest = {
        "decision_pairs": sum(row["kind"] == "decision" for row in pair_rows),
        "overlap_pairs": sum(row["kind"] == "overlap" for row in pair_rows),
        "correction_conditions": len(correction),
        "maximum_accounting_residual": maximum_residual,
        "worked_example_rule": (
            "nearest positive middle-budget profit gain to the median across frozen Experiment 1"
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote diagnostic outputs to {args.output}")


if __name__ == "__main__":
    main()
