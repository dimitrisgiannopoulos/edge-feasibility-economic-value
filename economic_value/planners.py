"""Fair information-versus-algorithm planner comparisons."""

from __future__ import annotations

import json
import time

import numpy as np

from .environment import (Environment, catalog_support, resample_overlap_support,
                          resample_support)
from .model import ProfitModel, ScenarioAverageModel, greedy_local_search


METHODS = ("G-Catalog", "M-Catalog", "M-Prior", "G-EDFS", "M-EDFS")


def _record(method: str, information: str, algorithm: str, planning_profit: float,
            actual, elapsed: float, instance, planning_status: str = "heuristic",
            planning_gap: float | None = None, planning_upper_bound: float | None = None) -> dict:
    selected_ids = [site for site, active in zip(instance.site_ids, actual.selected) if active]
    row = {"method": method, "information": information, "algorithm": algorithm,
           "planning_profit": planning_profit, "realized_profit": actual.metrics["profit"],
           "planning_error": planning_profit - actual.metrics["profit"],
           "selected_sites": json.dumps(selected_ids), "selection_seconds": elapsed,
           "planning_status": planning_status, "planning_mip_gap": planning_gap,
           "planning_profit_upper_bound": planning_upper_bound}
    row.update(actual.metrics)
    used_by_operator = np.zeros((instance.contracted_quota.shape[0], len(instance.site_ids)),
                                dtype=bool)
    for mno in range(used_by_operator.shape[0]):
        used_by_operator[mno] = actual.assignment[instance.group_mno == mno].sum(axis=0) > 1e-7
    row["multi_operator_sites"] = int((used_by_operator.sum(axis=0) >= 2).sum())
    row["revenue_normalized_profit"] = (actual.metrics["profit"] / instance.max_revenue
                                         if instance.max_revenue else 0.0)
    return row


def evaluate_methods(environment: Environment, budget: float,
                     calibration: dict[str, float], prior_scenarios: int = 6) -> list[dict]:
    """Plan under each permitted view, then reassign all footprints under truth."""
    actual_instance = environment.instance.with_budget(budget)
    actual_model = ProfitModel(actual_instance)
    catalog_instance = actual_instance.with_support(catalog_support(environment, calibration))
    catalog_model = ProfitModel(catalog_instance)
    outputs: list[dict] = []

    started = time.perf_counter()
    planned = greedy_local_search(catalog_model)
    actual = actual_model.solve_fixed(planned.selected)
    outputs.append(_record("G-Catalog", "catalog/geography", "greedy_local_search",
                           planned.metrics["profit"], actual, time.perf_counter() - started,
                           actual_instance))

    started = time.perf_counter()
    planned = catalog_model.solve_exact()
    actual = actual_model.solve_fixed(planned.selected)
    outputs.append(_record("M-Catalog", "catalog/geography", "MILP",
                           planned.metrics["profit"], actual, time.perf_counter() - started,
                           actual_instance, planned.status, planned.mip_gap,
                           planned.profit_upper_bound))
    outputs[-1]["planning_milp_recourse_adjustment"] = planned.metrics.get(
        "milp_recourse_adjustment", 0.0
    )

    sampler = (resample_overlap_support
               if environment.metadata.get("matched_support_count") else resample_support)
    sampled = np.stack([
        sampler(environment, 50_000_003 + environment.seed * 101 + index)
        for index in range(prior_scenarios)
    ])
    started = time.perf_counter()
    prior_model = ScenarioAverageModel(actual_instance, sampled)
    prior_selected, prior_profit, prior_status, prior_gap, _ = prior_model.solve_exact()
    actual = actual_model.solve_fixed(prior_selected)
    outputs.append(_record("M-Prior", "population prior", "scenario_MILP",
                           prior_profit, actual, time.perf_counter() - started,
                           actual_instance, prior_status, prior_gap))

    started = time.perf_counter()
    planned = greedy_local_search(actual_model)
    actual = actual_model.solve_fixed(planned.selected)
    outputs.append(_record("G-EDFS", "truthful resolved feasibility", "greedy_local_search",
                           planned.metrics["profit"], actual, time.perf_counter() - started,
                           actual_instance))

    started = time.perf_counter()
    planned = actual_model.solve_exact()
    actual = actual_model.solve_fixed(planned.selected)
    outputs.append(_record("M-EDFS", "truthful resolved feasibility", "MILP",
                           planned.metrics["profit"], actual, time.perf_counter() - started,
                           actual_instance, planned.status, planned.mip_gap,
                           planned.profit_upper_bound))
    outputs[-1]["planning_milp_recourse_adjustment"] = planned.metrics.get(
        "milp_recourse_adjustment", 0.0
    )
    optimum = outputs[-1]["realized_profit"]
    for row in outputs:
        row["planning_loss"] = optimum - row["realized_profit"]
        row["revenue_normalized_profit_difference_vs_m_edfs"] = (
            (row["realized_profit"] - optimum) / actual_instance.max_revenue
            if actual_instance.max_revenue else 0.0)
        if row["planning_loss"] < -0.02:
            raise AssertionError("A footprint exceeds the full-information optimum")
    return outputs
