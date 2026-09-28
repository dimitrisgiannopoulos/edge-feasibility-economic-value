"""Economic-sensitivity and post-deployment-correction helpers.

The main treatment remains information, not optimization. Both catalog and
current-feasibility conditions use the same greedy local-search planner. Exact
optimization remains confined to the existing validation results.
"""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import numpy as np

from .environment import Environment, catalog_support
from .model import EconomicInstance, ProfitModel, Solution, greedy_local_search


def scaled_economic_instance(
    environment: Environment,
    budget: float,
    fixed_cost_multiplier: float = 1.0,
    variable_cost_multiplier: float = 1.0,
) -> EconomicInstance:
    """Return one common economic scenario without changing its support map."""
    if fixed_cost_multiplier <= 0 or variable_cost_multiplier <= 0:
        raise ValueError("Cost multipliers must be positive")
    data = environment.instance
    return replace(
        data,
        fixed_cost=data.fixed_cost * fixed_cost_multiplier,
        variable_cost=data.variable_cost * variable_cost_multiplier,
        budget=float(budget),
    )


def _multi_operator_sites(instance: EconomicInstance, solution: Solution) -> int:
    used = np.zeros((instance.contracted_quota.shape[0], len(instance.site_ids)), dtype=bool)
    for mno in range(used.shape[0]):
        used[mno] = solution.assignment[instance.group_mno == mno].sum(axis=0) > 1e-7
    return int((used.sum(axis=0) >= 2).sum())


def _record(
    method: str,
    information: str,
    planning_solution: Solution,
    realized_solution: Solution,
    instance: EconomicInstance,
) -> dict[str, Any]:
    selected = [site for site, active in zip(instance.site_ids, realized_solution.selected) if active]
    selected_mask = realized_solution.selected.astype(bool)
    selected_compute = float(instance.compute_capacity[selected_mask].sum())
    selected_interface = float(instance.interface_capacity[selected_mask].sum())
    row: dict[str, Any] = {
        "method": method,
        "information": information,
        "algorithm": "greedy_local_search",
        "planning_status": planning_solution.status,
        "planning_seconds": planning_solution.seconds,
        "planning_profit": planning_solution.metrics["profit"],
        "realized_profit": realized_solution.metrics["profit"],
        "planning_error": planning_solution.metrics["profit"] - realized_solution.metrics["profit"],
        "selected_sites": json.dumps(selected),
        "multi_operator_sites": _multi_operator_sites(instance, realized_solution),
        "no_deployment": int(not selected_mask.any()),
        "individually_affordable_sites": int((instance.fixed_cost <= instance.budget).sum()),
        "selected_compute_capacity": selected_compute,
        "selected_interface_capacity": selected_interface,
        "compute_utilization": (
            realized_solution.metrics["compute_used"] / selected_compute
            if selected_compute else 0.0
        ),
        "interface_utilization": (
            realized_solution.metrics["throughput_used"] / selected_interface
            if selected_interface else 0.0
        ),
        "revenue_normalized_profit": (
            realized_solution.metrics["profit"] / instance.max_revenue
            if instance.max_revenue else 0.0
        ),
    }
    row.update(realized_solution.metrics)
    return row


def evaluate_greedy_information_pair(
    environment: Environment,
    instance: EconomicInstance,
    calibration: dict[str, float],
    time_limit_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Apply the same greedy planner under catalog and current support views."""
    actual_model = ProfitModel(instance.with_support(environment.resolved_support))
    catalog_instance = instance.with_support(catalog_support(environment, calibration))
    catalog_model = ProfitModel(catalog_instance)

    catalog_plan = greedy_local_search(catalog_model, time_limit_seconds)
    catalog_realized = actual_model.solve_fixed(catalog_plan.selected)
    current_plan = greedy_local_search(actual_model, time_limit_seconds)
    current_realized = actual_model.solve_fixed(current_plan.selected)
    return [
        _record(
            "Catalog + geography",
            "catalog/geography",
            catalog_plan,
            catalog_realized,
            instance,
        ),
        _record(
            "Current feasibility",
            "truthful resolved feasibility",
            current_plan,
            current_realized,
            instance,
        ),
    ]


def _weighted_phase_metrics(
    initial: Solution,
    final: Solution,
    detection_fraction: float,
) -> tuple[float, float, float]:
    initial_weight = detection_fraction
    final_weight = 1.0 - detection_fraction
    revenue = initial_weight * initial.metrics["revenue"] + final.metrics["revenue"]
    variable = initial_weight * initial.metrics["variable_cost"] + final.metrics["variable_cost"]
    coverage = (
        initial_weight * initial.metrics["qualifying_coverage"]
        + final_weight * final.metrics["qualifying_coverage"]
    )
    return float(revenue), float(variable), float(coverage)


def evaluate_post_deployment_correction(
    environment: Environment,
    budget: float,
    calibration: dict[str, float],
    detection_fraction: float,
    refund_fraction: float,
) -> dict[str, Any]:
    """Evaluate catalog deployment followed by informed correction.

    Fixed commitments are charged once. Removing an initial site returns the
    configured share of its fixed commitment. Added sites pay their full fixed
    cost. Variable cost and service revenue accrue in proportion to the time
    spent before and after correction. The original total budget remains in
    force over the whole horizon.
    """
    return evaluate_correction_grid(
        environment,
        budget,
        calibration,
        [detection_fraction],
        [refund_fraction],
    )[0]


def evaluate_correction_grid(
    environment: Environment,
    budget: float,
    calibration: dict[str, float],
    detection_fractions: list[float],
    refund_fractions: list[float],
) -> list[dict[str, Any]]:
    """Evaluate a correction grid while reusing the initial deployment solve."""
    if any(not 0.0 <= value <= 1.0 for value in detection_fractions):
        raise ValueError("Detection fractions must lie in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in refund_fractions):
        raise ValueError("Refund fractions must lie in [0, 1]")

    base = environment.instance.with_budget(float(budget))
    actual = base.with_support(environment.resolved_support)
    actual_model = ProfitModel(actual)
    catalog_model = ProfitModel(actual.with_support(catalog_support(environment, calibration)))
    initial_plan = greedy_local_search(catalog_model)
    initial_actual = actual_model.solve_fixed(initial_plan.selected)
    outputs = []
    for detection_fraction in detection_fractions:
        for refund_fraction in refund_fractions:
            outputs.append(_correct_one(
                base,
                initial_plan,
                initial_actual,
                budget,
                detection_fraction,
                refund_fraction,
            ))
    return outputs


def _correct_one(
    base: EconomicInstance,
    initial_plan: Solution,
    initial_actual: Solution,
    budget: float,
    detection_fraction: float,
    refund_fraction: float,
) -> dict[str, Any]:
    """Apply one delay/refund correction condition to a frozen initial plan."""

    initial_selected = initial_plan.selected.astype(bool)
    initial_fixed = float(base.fixed_cost[initial_selected].sum())
    base_nonrefundable_fixed = (1.0 - refund_fraction) * initial_fixed
    pre_variable = detection_fraction * initial_actual.metrics["variable_cost"]
    remaining_budget = budget - base_nonrefundable_fixed - pre_variable
    if remaining_budget < -1e-6:
        raise AssertionError("Initial phase exceeds the horizon budget")

    # The constant term assumes every initial site is refunded. Retaining an
    # initial site pays back its refundable share. A new site pays full cost.
    adjusted_fixed = np.where(
        initial_selected,
        refund_fraction * base.fixed_cost,
        base.fixed_cost,
    )
    remaining_fraction = 1.0 - detection_fraction
    correction_instance = replace(
        base,
        revenue=base.revenue * remaining_fraction,
        variable_cost=base.variable_cost * remaining_fraction,
        fixed_cost=adjusted_fixed,
        budget=max(0.0, float(remaining_budget)),
    )
    final_solution = greedy_local_search(ProfitModel(correction_instance))

    total_revenue, total_variable, coverage = _weighted_phase_metrics(
        initial_actual, final_solution, detection_fraction
    )
    total_fixed = base_nonrefundable_fixed + final_solution.metrics["fixed_cost"]
    total_expenditure = total_fixed + total_variable
    profit = total_revenue - total_expenditure
    if total_expenditure > budget + 1e-5 * max(1.0, budget):
        raise AssertionError("Correction accounting exceeds the horizon budget")

    removed = initial_selected & ~final_solution.selected
    added = ~initial_selected & final_solution.selected
    retained = initial_selected & final_solution.selected
    finally_used = final_solution.assignment.sum(axis=0) > 1e-7
    initial_unused = initial_selected & ~finally_used
    unrecovered = float((1.0 - refund_fraction) * base.fixed_cost[initial_unused].sum())
    nonrefundable_initial = float((1.0 - refund_fraction) * initial_fixed)
    added_fixed = float(base.fixed_cost[added].sum())
    refunded_fixed = float(refund_fraction * base.fixed_cost[removed].sum())
    pre_revenue = float(detection_fraction * initial_actual.metrics["revenue"])
    pre_variable = float(detection_fraction * initial_actual.metrics["variable_cost"])
    post_revenue = float(final_solution.metrics["revenue"])
    post_variable = float(final_solution.metrics["variable_cost"])
    return {
        "method": "Catalog then correct",
        "detection_fraction": detection_fraction,
        "refund_fraction": refund_fraction,
        "profit": float(profit),
        "revenue": total_revenue,
        "fixed_cost": float(total_fixed),
        "variable_cost": total_variable,
        "total_expenditure": float(total_expenditure),
        "qualifying_coverage": coverage,
        "initial_coverage": initial_actual.metrics["qualifying_coverage"],
        "final_coverage": final_solution.metrics["qualifying_coverage"],
        "initial_sites": int(initial_selected.sum()),
        "final_sites": int(final_solution.selected.sum()),
        "added_sites": int(added.sum()),
        "removed_sites": int(removed.sum()),
        "retained_sites": int(retained.sum()),
        "pre_detection_revenue": pre_revenue,
        "post_detection_revenue": post_revenue,
        "pre_detection_variable_cost": pre_variable,
        "post_detection_variable_cost": post_variable,
        "initial_fixed_commitment": initial_fixed,
        "refunded_fixed_cost": refunded_fixed,
        "remaining_budget_at_correction": float(max(0.0, remaining_budget)),
        "unrecovered_expenditure": unrecovered,
        "nonrefundable_initial_commitment": nonrefundable_initial,
        "added_fixed_cost": added_fixed,
        "correction_cost": unrecovered + added_fixed,
        "selected_sites_initial": json.dumps(
            [site for site, active in zip(base.site_ids, initial_selected) if active]
        ),
        "selected_sites_final": json.dumps(
            [site for site, active in zip(base.site_ids, final_solution.selected) if active]
        ),
        "max_revenue": base.max_revenue,
        "revenue_normalized_profit": float(profit / base.max_revenue),
    }


def evaluate_informed_from_start(
    environment: Environment,
    budget: float,
) -> dict[str, Any]:
    """Reference outcome when current feasibility is known before commitment."""
    instance = environment.instance.with_budget(float(budget)).with_support(
        environment.resolved_support
    )
    solution = greedy_local_search(ProfitModel(instance))
    row: dict[str, Any] = {
        "method": "Current feasibility from start",
        "profit": solution.metrics["profit"],
        "revenue_normalized_profit": solution.metrics["profit"] / instance.max_revenue,
        "selected_sites": json.dumps(
            [site for site, active in zip(instance.site_ids, solution.selected) if active]
        ),
    }
    row.update(solution.metrics)
    return row
