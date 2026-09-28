"""Profit-maximizing deployment and assignment models.

The ASP chooses sites before execution.  For a fixed footprint, all methods
receive the same competent runtime assignment under realized binary support.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeWarning, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, diags, hstack, vstack
import warnings


TOL = 1e-7
warnings.filterwarnings("ignore", message=".*threads.*", category=OptimizeWarning)
warnings.filterwarnings("ignore", message=".*threads.*", category=RuntimeWarning)


def _quiet_solver(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(*args, **kwargs)


def _marginal_score(gain: float, additional_fixed_cost: float) -> float:
    """Rank a profitable zero-incremental-cost addition ahead of paid additions."""
    return float("inf") if additional_fixed_cost <= TOL else gain / additional_fixed_cost


@dataclass(frozen=True)
class EconomicInstance:
    site_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    group_mno: np.ndarray
    demand: np.ndarray
    revenue: np.ndarray
    compute_per_unit: np.ndarray
    throughput_per_unit: np.ndarray
    fixed_cost: np.ndarray
    variable_cost: np.ndarray
    compute_capacity: np.ndarray
    interface_capacity: np.ndarray
    contracted_quota: np.ndarray
    support: np.ndarray
    budget: float
    site_tiers: tuple[str, ...]

    def __post_init__(self) -> None:
        g, s = len(self.group_ids), len(self.site_ids)
        if not g or not s or len(set(self.site_ids)) != s or len(set(self.group_ids)) != g:
            raise ValueError("Instance dimensions must be nonempty with unique IDs")
        arrays = {
            "group_mno": ((g,), int),
            "demand": ((g,), float),
            "revenue": ((g,), float),
            "compute_per_unit": ((g,), float),
            "throughput_per_unit": ((g,), float),
            "fixed_cost": ((s,), float),
            "variable_cost": ((g, s), float),
            "compute_capacity": ((s,), float),
            "interface_capacity": ((s,), float),
            "contracted_quota": ((int(np.max(self.group_mno)) + 1, s), float),
            "support": ((g, s), float),
        }
        for name, (shape, dtype) in arrays.items():
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected {shape}, got {value.shape}")
            object.__setattr__(self, name, value)
        nonnegative = ("demand", "revenue", "compute_per_unit", "throughput_per_unit",
                       "fixed_cost", "variable_cost", "compute_capacity",
                       "interface_capacity", "contracted_quota")
        if any(np.any(getattr(self, name) < 0) for name in nonnegative):
            raise ValueError("Economic and resource inputs must be nonnegative")
        if not np.isin(self.support, [0.0, 1.0]).all():
            raise ValueError("Resolved feasibility must be binary")
        if np.any(self.group_mno < 0) or not np.isfinite(self.budget) or self.budget < 0:
            raise ValueError("Invalid MNO index or budget")
        if len(self.site_tiers) != s:
            raise ValueError("Missing site tiers")

    @property
    def max_revenue(self) -> float:
        return float(self.demand @ self.revenue)

    def with_support(self, support: np.ndarray) -> "EconomicInstance":
        return replace(self, support=np.asarray(support, dtype=float))

    def with_budget(self, budget: float) -> "EconomicInstance":
        return replace(self, budget=float(budget))


@dataclass
class Solution:
    selected: np.ndarray
    assignment: np.ndarray
    metrics: dict[str, float]
    status: str
    mip_gap: float | None = None
    profit_upper_bound: float | None = None
    seconds: float = 0.0


class ProfitModel:
    """Sparse exact LP/MILP for one binary support matrix."""

    def __init__(self, instance: EconomicInstance, cache_limit: int = 32):
        self.data = instance
        self.g, self.s = instance.support.shape
        self.gg, self.ss = np.nonzero((instance.support > 0.5) &
                                      (instance.demand[:, None] > 0))
        self.e = len(self.gg)
        self.edge_margin = (instance.revenue[self.gg] -
                            instance.variable_cost[self.gg, self.ss])
        self.profit_scale = max(instance.max_revenue, 1.0)
        self.compute_scale = max(float(instance.demand @ instance.compute_per_unit), 1.0)
        self.throughput_scale = max(float(instance.demand @ instance.throughput_per_unit), 1.0)
        self._fixed_cache: dict[tuple[bool, ...], Solution] = {}
        self.cache_limit = max(1, int(cache_limit))
        self._build_assignment_constraints()

    def _build_assignment_constraints(self) -> None:
        edge = np.arange(self.e)
        demand_rows = coo_matrix((np.ones(self.e), (self.gg, edge)),
                                 shape=(self.g, self.e)).tocsr()
        compute_rows = coo_matrix((self.data.compute_per_unit[self.gg] / self.compute_scale,
                                   (self.ss, edge)), shape=(self.s, self.e)).tocsr()
        interface_rows = coo_matrix((self.data.throughput_per_unit[self.gg] /
                                     self.throughput_scale, (self.ss, edge)),
                                    shape=(self.s, self.e)).tocsr()
        quota_row = self.data.group_mno[self.gg] * self.s + self.ss
        quota_rows = coo_matrix((self.data.throughput_per_unit[self.gg] /
                                 self.throughput_scale, (quota_row, edge)),
                                shape=(self.data.contracted_quota.size, self.e)).tocsr()
        budget_row = csr_matrix((self.data.variable_cost[self.gg, self.ss] /
                                 self.profit_scale)[None, :])
        self.assignment_A = vstack([demand_rows, compute_rows, interface_rows,
                                    quota_rows, budget_row], format="csr")

    def _mask(self, selected: Iterable[bool]) -> np.ndarray:
        selected = np.asarray(selected)
        if selected.shape != (self.s,) or not np.isin(selected, [0, 1]).all():
            raise ValueError("Footprint must be a binary vector over sites")
        selected = selected.astype(bool)
        if self.data.fixed_cost[selected].sum() > self.data.budget + TOL:
            raise ValueError("Fixed commitment alone exceeds budget")
        return selected

    def _assignment_rhs(self, selected: np.ndarray) -> np.ndarray:
        d = self.data
        remaining = d.budget - float(d.fixed_cost[selected].sum())
        return np.r_[d.demand,
                     d.compute_capacity * selected / self.compute_scale,
                     d.interface_capacity * selected / self.throughput_scale,
                     (d.contracted_quota * selected).ravel() / self.throughput_scale,
                     remaining / self.profit_scale]

    def solve_fixed(self, selected: Iterable[bool]) -> Solution:
        selected = self._mask(selected)
        key = tuple(selected.tolist())
        if key in self._fixed_cache:
            solution = self._fixed_cache.pop(key)
            self._fixed_cache[key] = solution
            return solution
        started = time.perf_counter()
        assignment = np.zeros((self.g, self.s), dtype=float)
        if self.e and selected.any():
            upper = self.data.demand[self.gg] * selected[self.ss]
            arguments = {
                "A_ub": self.assignment_A,
                "b_ub": self._assignment_rhs(selected),
                "bounds": np.column_stack([np.zeros(self.e), upper]),
                "options": {"primal_feasibility_tolerance": 1e-9,
                            "dual_feasibility_tolerance": 1e-9,
                            "threads": 1},
            }
            result = _quiet_solver(
                linprog, -self.edge_margin / self.profit_scale,
                method="highs", **arguments,
            )
            if not result.success:
                result = _quiet_solver(
                    linprog, -self.edge_margin / self.profit_scale,
                    method="highs-ipm", **arguments,
                )
            if not result.success:
                raise RuntimeError(f"Fixed-footprint assignment failed: {result.message}")
            assignment[self.gg, self.ss] = result.x
        solution = Solution(selected=selected.copy(), assignment=assignment,
                            metrics=self.metrics(selected, assignment), status="optimal",
                            seconds=time.perf_counter() - started)
        self._fixed_cache[key] = solution
        while len(self._fixed_cache) > self.cache_limit:
            self._fixed_cache.pop(next(iter(self._fixed_cache)))
        return solution

    def solve_exact(self, time_limit: float = 120.0, mip_gap: float = 1e-8) -> Solution:
        """Joint site selection and assignment with a certified HiGHS result."""
        d = self.data
        started = time.perf_counter()
        zero_gs = csr_matrix((self.g, self.s))
        zero_ss = csr_matrix((self.s, self.s))
        zero_qs = csr_matrix((d.contracted_quota.size, self.s))
        demand_x = zero_gs
        compute_x = -diags(d.compute_capacity / self.compute_scale)
        interface_x = -diags(d.interface_capacity / self.throughput_scale)
        quota_values = -(d.contracted_quota / self.throughput_scale).ravel()
        quota_rows = np.arange(d.contracted_quota.size)
        quota_cols = np.tile(np.arange(self.s), d.contracted_quota.shape[0])
        quota_x = coo_matrix((quota_values, (quota_rows, quota_cols)),
                             shape=(d.contracted_quota.size, self.s)).tocsr()
        budget_x = csr_matrix((d.fixed_cost / self.profit_scale)[None, :])
        base_x = vstack([demand_x, compute_x, interface_x, quota_x, budget_x], format="csr")
        rows = hstack([base_x, self.assignment_A], format="csr")
        gate = coo_matrix((-d.demand[self.gg], (np.arange(self.e), self.ss)),
                          shape=(self.e, self.s)).tocsr()
        rows = vstack([rows, hstack([gate, diags(np.ones(self.e))])], format="csr")
        rhs = np.r_[d.demand,
                    np.zeros(self.s + self.s + d.contracted_quota.size),
                    d.budget / self.profit_scale,
                    np.zeros(self.e)]
        objective = np.r_[d.fixed_cost / self.profit_scale,
                          -self.edge_margin / self.profit_scale]
        result = _quiet_solver(
            milp, objective,
            integrality=np.r_[np.ones(self.s), np.zeros(self.e)],
            bounds=Bounds(np.zeros(self.s + self.e),
                          np.r_[np.ones(self.s), d.demand[self.gg]]),
            constraints=LinearConstraint(rows, -np.inf, rhs),
            options={"time_limit": time_limit, "mip_rel_gap": mip_gap,
                     "mip_feasibility_tolerance": 1e-9, "threads": 1},
        )
        if result.x is None:
            raise RuntimeError(f"MILP produced no feasible incumbent: {result.message}")
        solved = result.status == 0 and result.success
        selected = result.x[:self.s] > 0.5
        # Re-optimize continuous recourse with the tighter LP tolerance. The
        # MILP establishes the footprint; this pass prevents scaled MIP
        # feasibility tolerances from leaking into reported assignments.
        fixed_solution = self.solve_fixed(selected)
        assignment = fixed_solution.assignment.copy()
        metrics = dict(fixed_solution.metrics)
        mip_profit = float(-result.fun * self.profit_scale)
        metrics["milp_recourse_adjustment"] = metrics["profit"] - mip_profit
        upper = None
        if getattr(result, "mip_dual_bound", None) is not None:
            upper = float(-result.mip_dual_bound * self.profit_scale)
        solution = Solution(selected=selected, assignment=assignment, metrics=metrics,
                            status="optimal" if solved else "time_limit",
                            mip_gap=float(result.mip_gap) if result.mip_gap is not None else None,
                            profit_upper_bound=upper,
                            seconds=time.perf_counter() - started)
        reconciliation_tolerance = max(0.02, 1e-4 * d.max_revenue)
        if solved and abs(metrics["milp_recourse_adjustment"]) > reconciliation_tolerance:
            raise AssertionError(
                "MILP objective and accounting disagree: "
                f"LP={metrics['profit']:.9f}, MIP={mip_profit:.9f}, "
                f"difference={metrics['milp_recourse_adjustment']:.9f}, "
                f"gap={result.mip_gap}"
            )
        return solution

    def metrics(self, selected: Iterable[bool], assignment: np.ndarray) -> dict[str, float]:
        selected = self._mask(selected)
        d = self.data
        assignment = np.asarray(assignment, dtype=float)
        if assignment.shape != (self.g, self.s) or not np.isfinite(assignment).all():
            raise AssertionError("Invalid assignment matrix")
        if np.any(assignment < -TOL) or np.any(assignment > d.demand[:, None] + TOL):
            raise AssertionError("Assignment outside demand bounds")
        if np.any(assignment.sum(axis=1) > d.demand + TOL):
            raise AssertionError("Demand served more than once")
        if np.any(assignment > d.demand[:, None] * d.support * selected + TOL):
            raise AssertionError("Unsupported or inactive assignment")
        compute = (assignment * d.compute_per_unit[:, None]).sum(axis=0)
        throughput = (assignment * d.throughput_per_unit[:, None]).sum(axis=0)
        quota = np.zeros_like(d.contracted_quota)
        for m in range(quota.shape[0]):
            quota[m] = (assignment[d.group_mno == m] *
                        d.throughput_per_unit[d.group_mno == m, None]).sum(axis=0)
        resource_tolerance = 1e-6
        if np.any(compute > d.compute_capacity * selected +
                  resource_tolerance * np.maximum(1.0, d.compute_capacity)):
            raise AssertionError("Shared compute capacity exceeded")
        if np.any(throughput > d.interface_capacity * selected +
                  resource_tolerance * np.maximum(1.0, d.interface_capacity)):
            raise AssertionError("Site interface capacity exceeded")
        if np.any(quota > d.contracted_quota * selected +
                  resource_tolerance * np.maximum(1.0, d.contracted_quota)):
            raise AssertionError("Contracted MNO/site quota exceeded")
        served = assignment.sum(axis=1)
        revenue = float((assignment * d.revenue[:, None]).sum())
        variable = float((assignment * d.variable_cost).sum())
        fixed = float(d.fixed_cost[selected].sum())
        expenditure = fixed + variable
        if expenditure > d.budget + resource_tolerance * max(1.0, d.budget):
            raise AssertionError("Total spending budget exceeded")
        site_used = assignment.sum(axis=0) > TOL
        result = {
            "profit": revenue - expenditure,
            "revenue": revenue,
            "fixed_cost": fixed,
            "variable_cost": variable,
            "total_expenditure": expenditure,
            "qualifying_demand": float(served.sum()),
            "qualifying_coverage": float(served.sum() / d.demand.sum()) if d.demand.sum() else 0.0,
            "unserved_demand": float((d.demand - served).sum()),
            "sites": int(selected.sum()),
            "stranded_expenditure": float(d.fixed_cost[selected & ~site_used].sum()),
            "compute_used": float(compute.sum()),
            "throughput_used": float(throughput.sum()),
            "max_revenue": d.max_revenue,
        }
        for m in range(d.contracted_quota.shape[0]):
            mask = d.group_mno == m
            total = float(d.demand[mask].sum())
            result[f"operator_{m}_coverage"] = float(served[mask].sum() / total) if total else 0.0
        for tier in sorted(set(d.site_tiers)):
            tier_mask = np.array([value == tier for value in d.site_tiers])
            result[f"{tier.lower()}_sites"] = int((selected & tier_mask).sum())
        return result


def greedy_local_search(model: ProfitModel, time_limit_seconds: float | None = None) -> Solution:
    """Marginal-profit additions and local improvement, optionally time-bounded."""
    started = time.perf_counter()

    def expired() -> bool:
        return (time_limit_seconds is not None
                and time.perf_counter() - started >= time_limit_seconds)

    def finish(mask: np.ndarray, status: str) -> Solution:
        solution = model.solve_fixed(mask)
        return replace(solution, status=status, seconds=time.perf_counter() - started)

    selected = np.zeros(model.s, dtype=bool)
    while True:
        current = model.solve_fixed(selected).metrics["profit"]
        best_mask = None
        best_score = 0.0
        for add in np.flatnonzero(~selected):
            if expired():
                return finish(selected, "time_limit")
            candidate = selected.copy()
            candidate[add] = True
            if model.data.fixed_cost[candidate].sum() > model.data.budget + TOL:
                continue
            gain = model.solve_fixed(candidate).metrics["profit"] - current
            if gain <= 1e-6:
                continue
            score = _marginal_score(gain, model.data.fixed_cost[add])
            if best_mask is None or score > best_score + 1e-12:
                best_mask, best_score = candidate, score
        if best_mask is None:
            break
        selected = best_mask

    while True:
        current = model.solve_fixed(selected).metrics["profit"]
        best_mask = None
        best_profit = current
        active = list(np.flatnonzero(selected))
        inactive = list(np.flatnonzero(~selected))
        candidates = []
        for remove in active:
            candidate = selected.copy()
            candidate[remove] = False
            candidates.append(candidate)
        for remove in active:
            for add in inactive:
                candidate = selected.copy()
                candidate[remove] = False
                candidate[add] = True
                if model.data.fixed_cost[candidate].sum() <= model.data.budget + TOL:
                    candidates.append(candidate)
        for candidate in candidates:
            if expired():
                return finish(selected, "time_limit")
            profit = model.solve_fixed(candidate).metrics["profit"]
            if profit > best_profit + 1e-6:
                best_mask, best_profit = candidate, profit
        if best_mask is None:
            break
        selected = best_mask
    return finish(selected, "local_optimum")


class ScenarioAverageModel:
    """Fixed-footprint expected profit across sampled support realizations."""

    def __init__(self, instance: EconomicInstance, supports: np.ndarray):
        supports = np.asarray(supports, dtype=float)
        if supports.ndim != 3 or supports.shape[1:] != instance.support.shape:
            raise ValueError("Scenario support tensor has wrong shape")
        if not np.isin(supports, [0.0, 1.0]).all():
            raise ValueError("Scenario supports must be binary")
        self.models = [ProfitModel(instance.with_support(support)) for support in supports]
        self.data = instance
        self.s = len(instance.site_ids)
        self.cache: dict[tuple[bool, ...], float] = {}

    def objective(self, selected: Iterable[bool]) -> float:
        selected = np.asarray(selected, dtype=bool)
        key = tuple(selected.tolist())
        if key not in self.cache:
            self.cache[key] = float(np.mean([
                model.solve_fixed(selected).metrics["profit"] for model in self.models
            ]))
        return self.cache[key]

    def greedy(self) -> np.ndarray:
        selected = np.zeros(self.s, dtype=bool)
        while True:
            current = self.objective(selected)
            best, best_score = None, 0.0
            for add in np.flatnonzero(~selected):
                candidate = selected.copy()
                candidate[add] = True
                if self.data.fixed_cost[candidate].sum() > self.data.budget + TOL:
                    continue
                gain = self.objective(candidate) - current
                if gain > 1e-6:
                    score = _marginal_score(gain, self.data.fixed_cost[add])
                    if best is None or score > best_score + 1e-12:
                        best, best_score = candidate, score
            if best is None:
                break
            selected = best
        while True:
            current = self.objective(selected)
            best, best_value = None, current
            active, inactive = list(np.flatnonzero(selected)), list(np.flatnonzero(~selected))
            for remove in active:
                candidates = [selected.copy()]
                candidates[0][remove] = False
                for add in inactive:
                    candidate = selected.copy()
                    candidate[remove] = False
                    candidate[add] = True
                    candidates.append(candidate)
                for candidate in candidates:
                    if self.data.fixed_cost[candidate].sum() > self.data.budget + TOL:
                        continue
                    value = self.objective(candidate)
                    if value > best_value + 1e-6:
                        best, best_value = candidate, value
            if best is None:
                return selected
            selected = best

    def solve_exact(self, time_limit: float = 120.0,
                    mip_gap: float = 1e-8) -> tuple[np.ndarray, float, str, float | None, float]:
        """Solve sampled support as a two-stage MILP with one shared footprint."""
        started = time.perf_counter()
        d = self.data
        scenario_count = len(self.models)
        offsets, variable_count = [], self.s
        for model in self.models:
            offsets.append(variable_count)
            variable_count += model.e
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        right_hand_side: list[float] = []
        row = 0

        def add(entries: Iterable[tuple[int, float]], bound: float) -> None:
            nonlocal row
            for column, value in entries:
                if value:
                    row_indices.append(row)
                    column_indices.append(int(column))
                    values.append(float(value))
            right_hand_side.append(float(bound))
            row += 1

        for model, offset in zip(self.models, offsets):
            edge_columns = offset + np.arange(model.e)
            for group in range(model.g):
                edges = np.flatnonzero(model.gg == group)
                add(((edge_columns[e], 1.0) for e in edges), d.demand[group])
            for site in range(model.s):
                edges = np.flatnonzero(model.ss == site)
                add([(site, -d.compute_capacity[site] / model.compute_scale)] +
                    [(edge_columns[e], d.compute_per_unit[model.gg[e]] / model.compute_scale)
                     for e in edges], 0.0)
            for site in range(model.s):
                edges = np.flatnonzero(model.ss == site)
                add([(site, -d.interface_capacity[site] / model.throughput_scale)] +
                    [(edge_columns[e], d.throughput_per_unit[model.gg[e]] /
                      model.throughput_scale) for e in edges], 0.0)
            for mno in range(d.contracted_quota.shape[0]):
                for site in range(model.s):
                    edges = np.flatnonzero(
                        (model.ss == site) & (d.group_mno[model.gg] == mno)
                    )
                    add([(site, -d.contracted_quota[mno, site] / model.throughput_scale)] +
                        [(edge_columns[e], d.throughput_per_unit[model.gg[e]] /
                          model.throughput_scale) for e in edges], 0.0)
            add([(site, d.fixed_cost[site] / model.profit_scale)
                 for site in range(model.s)] +
                [(edge_columns[e], d.variable_cost[model.gg[e], model.ss[e]] /
                  model.profit_scale) for e in range(model.e)],
                d.budget / model.profit_scale)
            for edge in range(model.e):
                add(((model.ss[edge], -d.demand[model.gg[edge]]),
                     (edge_columns[edge], 1.0)), 0.0)

        constraints = coo_matrix(
            (values, (row_indices, column_indices)), shape=(row, variable_count)
        ).tocsr()
        objective = np.zeros(variable_count)
        objective[:self.s] = d.fixed_cost / max(d.max_revenue, 1.0)
        upper = np.ones(variable_count)
        for model, offset in zip(self.models, offsets):
            objective[offset:offset + model.e] = (
                -model.edge_margin / model.profit_scale / scenario_count
            )
            upper[offset:offset + model.e] = d.demand[model.gg]
        result = _quiet_solver(
            milp, objective,
            integrality=np.r_[np.ones(self.s), np.zeros(variable_count - self.s)],
            bounds=Bounds(np.zeros(variable_count), upper),
            constraints=LinearConstraint(constraints, -np.inf, np.asarray(right_hand_side)),
            options={"time_limit": time_limit, "mip_rel_gap": mip_gap,
                     "mip_feasibility_tolerance": 1e-9, "threads": 1},
        )
        if result.x is None:
            raise RuntimeError(f"Prior-informed MILP produced no incumbent: {result.message}")
        selected = result.x[:self.s] > .5
        expected_profit = self.objective(selected)
        status = "optimal" if result.status == 0 and result.success else "time_limit"
        gap = float(result.mip_gap) if result.mip_gap is not None else None
        return selected, expected_profit, status, gap, time.perf_counter() - started
