"""Correctness tests for economic hardening and correction accounting."""

from __future__ import annotations

import unittest

import numpy as np

from .environment import Environment
from .hardening import (
    evaluate_correction_grid,
    evaluate_greedy_information_pair,
    evaluate_informed_from_start,
    scaled_economic_instance,
)
from .model import EconomicInstance
from .model import ProfitModel, greedy_local_search


def _environment() -> Environment:
    areas = ({"id": "A0", "x": 0.0, "y": 0.0, "demand_weight": 1.0},)
    sites = (
        {"id": "S0", "tier": "LOCAL", "ecsp": 0, "x": 1.0, "y": 0.0,
         "fixed_cost": 2.0, "variable_cost": 0.2},
        {"id": "S1", "tier": "LOCAL", "ecsp": 0, "x": 10.0, "y": 0.0,
         "fixed_cost": 2.0, "variable_cost": 0.2},
    )
    groups = ({"id": "G0", "area": 0, "mno": 0, "profile": "moderate",
               "demand": 10.0, "revenue": 2.0, "compute": 1.0,
               "throughput": 1.0},)
    support = np.array([[0.0, 1.0]])
    instance = EconomicInstance(
        site_ids=("S0", "S1"),
        group_ids=("G0",),
        group_mno=np.array([0]),
        demand=np.array([10.0]),
        revenue=np.array([2.0]),
        compute_per_unit=np.array([1.0]),
        throughput_per_unit=np.array([1.0]),
        fixed_cost=np.array([2.0, 2.0]),
        variable_cost=np.array([[0.2, 0.2]]),
        compute_capacity=np.array([10.0, 10.0]),
        interface_capacity=np.array([10.0, 10.0]),
        contracted_quota=np.array([[10.0, 10.0]]),
        support=support,
        budget=25.0,
        site_tiers=("LOCAL", "LOCAL"),
    )
    return Environment(
        seed=1,
        topology="regular",
        areas=areas,
        sites=sites,
        groups=groups,
        operator_anchors=(),
        paths=(),
        raw_status=np.array([["UNVIABLE", "VIABLE"]], dtype=object),
        resolved_support=support,
        delay_ms=np.array([[100.0, 10.0]]),
        instance=instance,
        commercial_package={"MNO_0": True},
        metadata={"delay_budget_ms": 5.0, "n_mnos": 1, "budget_levels": (25.0,)},
    )


CALIBRATION = {
    "intercept": 0.0,
    "distance": 1.0,
    "tier_local": 0.0,
    "tier_regional": 0.0,
    "tier_central": 0.0,
}


class HardeningCorrectness(unittest.TestCase):
    def test_time_bounded_greedy_returns_a_valid_incumbent(self) -> None:
        environment = _environment()
        solution = greedy_local_search(ProfitModel(environment.instance), time_limit_seconds=0.0)
        self.assertEqual(solution.status, "time_limit")
        self.assertGreaterEqual(solution.metrics["profit"], 0.0)

    def test_cost_scaling_preserves_support_and_budget(self) -> None:
        environment = _environment()
        scaled = scaled_economic_instance(environment, 17.0, 2.0, 3.0)
        np.testing.assert_array_equal(scaled.support, environment.resolved_support)
        np.testing.assert_allclose(scaled.fixed_cost, [4.0, 4.0])
        np.testing.assert_allclose(scaled.variable_cost, [[0.6, 0.6]])
        self.assertEqual(scaled.budget, 17.0)

    def test_catalog_nonnegative_prediction_can_realize_negative_profit(self) -> None:
        environment = _environment()
        rows = evaluate_greedy_information_pair(environment, environment.instance, CALIBRATION)
        catalog = next(row for row in rows if row["method"] == "Catalog + geography")
        self.assertGreaterEqual(catalog["planning_profit"], 0.0)
        self.assertLess(catalog["realized_profit"], 0.0)

    def test_immediate_full_refund_removes_information_timing_advantage(self) -> None:
        environment = _environment()
        informed = evaluate_informed_from_start(environment, 25.0)
        corrected = evaluate_correction_grid(
            environment, 25.0, CALIBRATION, [0.0], [1.0]
        )[0]
        self.assertAlmostEqual(corrected["profit"], informed["profit"], places=7)
        self.assertAlmostEqual(
            corrected["qualifying_coverage"], informed["qualifying_coverage"], places=7
        )
        self.assertLessEqual(corrected["total_expenditure"], 25.0)

    def test_delayed_nonrefundable_correction_retains_a_loss(self) -> None:
        environment = _environment()
        informed = evaluate_informed_from_start(environment, 25.0)
        corrected = evaluate_correction_grid(
            environment, 25.0, CALIBRATION, [0.25], [0.0]
        )[0]
        self.assertLess(corrected["profit"], informed["profit"])
        self.assertGreater(corrected["unrecovered_expenditure"], 0.0)


if __name__ == "__main__":
    unittest.main()
