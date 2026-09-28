"""Experiment 0: hand-checkable correctness tests for the economic model."""

from __future__ import annotations

from itertools import product
import unittest

import numpy as np

from .environment import (
    _resolve,
    build_composition_controlled_overlap_environment,
    build_environment,
    build_overlap_environment,
)
from .edfs_adapter import response_to_support
from .model import EconomicInstance, ProfitModel, ScenarioAverageModel


def instance(
    support: list[list[int]],
    demand: list[float],
    revenue: list[float],
    fixed: list[float],
    variable: list[list[float]],
    compute_capacity: list[float],
    *,
    group_mno: list[int] | None = None,
    compute_per_unit: list[float] | None = None,
    interface_capacity: list[float] | None = None,
    quota: list[list[float]] | None = None,
    budget: float = 1_000.0,
    tiers: list[str] | None = None,
) -> EconomicInstance:
    """Construct a compact deterministic instance for one assertion."""
    support_array = np.asarray(support, dtype=float)
    groups, sites = support_array.shape
    group_mno = group_mno or [0] * groups
    compute_per_unit = compute_per_unit or [1.0] * groups
    interface_capacity = interface_capacity or [1_000.0] * sites
    mno_count = max(group_mno) + 1
    quota = quota or [[1_000.0] * sites for _ in range(mno_count)]
    tiers = tiers or ["LOCAL"] * sites
    return EconomicInstance(
        site_ids=tuple(f"S{i}" for i in range(sites)),
        group_ids=tuple(f"G{i}" for i in range(groups)),
        group_mno=np.asarray(group_mno),
        demand=np.asarray(demand),
        revenue=np.asarray(revenue),
        compute_per_unit=np.asarray(compute_per_unit),
        throughput_per_unit=np.ones(groups),
        fixed_cost=np.asarray(fixed),
        variable_cost=np.asarray(variable),
        compute_capacity=np.asarray(compute_capacity),
        interface_capacity=np.asarray(interface_capacity),
        contracted_quota=np.asarray(quota),
        support=support_array,
        budget=budget,
        site_tiers=tuple(tiers),
    )


class EconomicModelCorrectness(unittest.TestCase):
    def test_total_demand_parameter_preserves_requested_scale(self) -> None:
        baseline = build_environment(1234, n_areas=24, n_sites=48, total_demand=10_000.0)
        scaled = build_environment(1234, n_areas=24, n_sites=48, total_demand=25_000.0)
        self.assertAlmostEqual(float(baseline.instance.demand.sum()), 10_000.0)
        self.assertAlmostEqual(float(scaled.instance.demand.sum()), 25_000.0)
        np.testing.assert_array_equal(scaled.resolved_support, baseline.resolved_support)
        np.testing.assert_allclose(
            scaled.instance.compute_capacity / baseline.instance.compute_capacity, 2.5
        )

    def test_exact_milp_matches_exhaustive_footprints(self) -> None:
        data = instance(
            support=[[1, 0, 1], [0, 1, 1]], demand=[8, 7], revenue=[3, 3],
            fixed=[5, 4, 8], variable=[[1, 2, 1.2], [2, 1, 1.2]],
            compute_capacity=[8, 7, 15], budget=30,
        )
        model = ProfitModel(data)
        exhaustive = []
        for bits in product((False, True), repeat=3):
            if data.fixed_cost[np.asarray(bits)].sum() <= data.budget:
                exhaustive.append(model.solve_fixed(bits).metrics["profit"])
        exact = model.solve_exact()
        self.assertAlmostEqual(exact.metrics["profit"], max(exhaustive), places=5)
        self.assertEqual(exact.status, "optimal")

    def test_operator_specific_support_controls_assignment(self) -> None:
        data = instance(
            support=[[0, 1], [1, 0]], demand=[5, 5], revenue=[3, 3],
            fixed=[2, 2], variable=[[1, 1], [1, 1]], compute_capacity=[10, 10],
            group_mno=[0, 1], budget=20,
        )
        solution = ProfitModel(data).solve_exact()
        self.assertAlmostEqual(solution.assignment[0, 0], 0.0)
        self.assertAlmostEqual(solution.assignment[1, 1], 0.0)
        self.assertAlmostEqual(solution.metrics["qualifying_demand"], 10.0)

    def test_one_regional_site_can_replace_two_local_sites(self) -> None:
        data = instance(
            support=[[1, 0, 1], [0, 1, 1]], demand=[5, 5], revenue=[5, 5],
            fixed=[8, 8, 11], variable=[[1, 9, 1], [9, 1, 1]],
            compute_capacity=[5, 5, 10], budget=30,
            tiers=["LOCAL", "LOCAL", "REGIONAL"],
        )
        solution = ProfitModel(data).solve_exact()
        self.assertEqual(solution.selected.tolist(), [False, False, True])
        self.assertAlmostEqual(solution.metrics["qualifying_demand"], 10.0)

    def test_shared_compute_capacity_is_not_repeated_per_operator(self) -> None:
        data = instance(
            support=[[1], [1]], demand=[10, 10], revenue=[3, 3], fixed=[1],
            variable=[[1], [1]], compute_capacity=[12], group_mno=[0, 1], budget=100,
        )
        solution = ProfitModel(data).solve_exact()
        self.assertAlmostEqual(solution.metrics["qualifying_demand"], 12.0, places=5)
        self.assertAlmostEqual(solution.metrics["compute_used"], 12.0, places=5)

    def test_interface_and_operator_quota_are_separate_shared_limits(self) -> None:
        data = instance(
            support=[[1], [1]], demand=[10, 10], revenue=[3, 3], fixed=[1],
            variable=[[1], [1]], compute_capacity=[100], group_mno=[0, 1],
            interface_capacity=[11], quota=[[4], [9]], budget=100,
        )
        solution = ProfitModel(data).solve_exact()
        self.assertAlmostEqual(solution.metrics["qualifying_demand"], 11.0, places=5)
        self.assertLessEqual(solution.assignment[0].sum(), 4.0 + 1e-6)
        self.assertLessEqual(solution.assignment[1].sum(), 9.0 + 1e-6)

    def test_budget_charges_fixed_and_variable_expenditure(self) -> None:
        data = instance(
            support=[[1]], demand=[20], revenue=[3], fixed=[2], variable=[[1]],
            compute_capacity=[100], budget=12,
        )
        solution = ProfitModel(data).solve_exact()
        self.assertAlmostEqual(solution.metrics["qualifying_demand"], 10.0, places=5)
        self.assertAlmostEqual(solution.metrics["total_expenditure"], 12.0, places=5)

    def test_partial_condition_resolves_to_binary_support(self) -> None:
        paths = (
            {"group": 0, "site": 0, "allowed": True, "kind": "LOCAL_BREAKOUT",
             "condition": "premium_breakout", "condition_satisfied": True},
            {"group": 0, "site": 1, "allowed": True, "kind": "LOCAL_BREAKOUT",
             "condition": "premium_breakout", "condition_satisfied": False},
        )
        raw, support = _resolve(paths, np.array([[10.0, 10.0]]), 20.0, 1, 2)
        self.assertEqual(raw.tolist(), [["PARTIAL", "PARTIAL"]])
        self.assertEqual(support.tolist(), [[1.0, 0.0]])

    def test_unprofitable_instance_selects_no_sites(self) -> None:
        data = instance(
            support=[[1, 1]], demand=[10], revenue=[1], fixed=[20, 25],
            variable=[[.9, .8]], compute_capacity=[10, 10], budget=100,
        )
        solution = ProfitModel(data).solve_exact()
        self.assertFalse(solution.selected.any())
        self.assertAlmostEqual(solution.metrics["profit"], 0.0)

    def test_exact_solver_is_repeatable(self) -> None:
        data = instance(
            support=[[1, 1], [1, 0]], demand=[4, 6], revenue=[3, 3],
            fixed=[4, 5], variable=[[1, 1.1], [1, 2]], compute_capacity=[10, 10],
        )
        first = ProfitModel(data).solve_exact()
        second = ProfitModel(data).solve_exact()
        self.assertEqual(first.selected.tolist(), second.selected.tolist())
        np.testing.assert_allclose(first.assignment, second.assignment, atol=1e-7)

    def test_prior_scenario_milp_matches_exhaustive_expected_profit(self) -> None:
        data = instance(
            support=[[1, 1], [1, 0]], demand=[5, 7], revenue=[3, 3],
            fixed=[3, 4], variable=[[1, 1.2], [1, 2]], compute_capacity=[10, 10],
            budget=20,
        )
        supports = np.asarray(([[1, 1], [1, 0]], [[0, 1], [1, 1]]), dtype=float)
        model = ScenarioAverageModel(data, supports)
        selected, objective, status, gap, _ = model.solve_exact()
        exhaustive = []
        for bits in product((False, True), repeat=2):
            exhaustive.append((model.objective(bits), list(bits)))
        self.assertAlmostEqual(objective, max(value for value, _ in exhaustive), places=5)
        self.assertEqual(status, "optimal")
        self.assertLessEqual(gap or 0.0, 1e-8)
        self.assertAlmostEqual(model.objective(selected), objective, places=5)

    def test_matched_fragmentation_uses_global_operator_site_policies(self) -> None:
        reference = None
        for overlap in ("aligned", "intermediate", "disjoint"):
            environment = build_overlap_environment(30_000, overlap, "clustered", 6)
            if reference is None:
                reference = environment.instance
            else:
                np.testing.assert_allclose(environment.instance.demand, reference.demand)
                np.testing.assert_allclose(environment.instance.fixed_cost, reference.fixed_cost)
                np.testing.assert_allclose(environment.instance.variable_cost, reference.variable_cost)
                np.testing.assert_allclose(
                    environment.instance.compute_capacity, reference.compute_capacity
                )
                np.testing.assert_allclose(
                    environment.instance.interface_capacity, reference.interface_capacity
                )
                self.assertEqual(environment.instance.site_tiers, reference.site_tiers)
            support = environment.resolved_support.reshape(len(environment.areas), 3,
                                                           len(environment.sites))
            np.testing.assert_allclose(support.sum(axis=2), 6.0)
            if overlap == "aligned":
                np.testing.assert_array_equal(support[:, 0], support[:, 1])
                np.testing.assert_array_equal(support[:, 1], support[:, 2])
            if overlap == "disjoint":
                operator_uses_site = support.sum(axis=0) > 0
                self.assertLessEqual(int(operator_uses_site.sum(axis=0).max()), 1)

    def test_edfs_style_response_resolves_conditions_and_rejects_public_internet(self) -> None:
        response = {
            "areaResults": [{
                "targetArea": {"area": {"areaType": "CIRCLE"}},
                "edgeCloudZoneViabilities": [
                    {"edgeCloudZoneId": "S0", "viability": "VIABLE"},
                    {"edgeCloudZoneId": "S1", "viability": "CONDITIONALLY_VIABLE",
                     "conditionInfo": {"conditionType": "SERVICE_ACTIVATION_REQUIRED"}},
                    {"edgeCloudZoneId": "S2", "viability": "UNKNOWN"},
                ],
            }]
        }
        support = response_to_support(
            response, ["A0"], ["S0", "S1", "S2"], {"S1": True},
            {"S0": "PUBLIC_INTERNET", "S1": "OPERATOR_MANAGED"},
        )
        self.assertEqual(support.tolist(), [[0.0, 1.0, 0.0]])

    def test_controlled_overlap_preserves_tier_and_resource_composition(self) -> None:
        reference = None
        for overlap in ("aligned", "intermediate", "disjoint"):
            environment = build_composition_controlled_overlap_environment(
                31_000, overlap, "clustered"
            )
            support = environment.resolved_support
            tiers = np.asarray(environment.instance.site_tiers)
            self.assertTrue(environment.metadata["composition_controlled"])
            np.testing.assert_allclose(support.sum(axis=1), 6.0)
            np.testing.assert_allclose(support[:, tiers == "LOCAL"].sum(axis=1), 4.0)
            np.testing.assert_allclose(support[:, tiers == "REGIONAL"].sum(axis=1), 1.0)
            np.testing.assert_allclose(support[:, tiers == "CENTRAL"].sum(axis=1), 1.0)
            if reference is None:
                reference = environment.instance
            else:
                np.testing.assert_allclose(environment.instance.fixed_cost, reference.fixed_cost)
                np.testing.assert_allclose(
                    environment.instance.compute_capacity, reference.compute_capacity
                )
                np.testing.assert_allclose(
                    environment.instance.interface_capacity, reference.interface_capacity
                )


if __name__ == "__main__":
    unittest.main()
