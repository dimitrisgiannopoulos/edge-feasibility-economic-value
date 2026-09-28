"""Synthetic physical environments and exposed planning views.

The generator creates sites, operator anchors, interconnection paths, delays,
and commercial conditions first.  Binary support is derived from those facts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import EconomicInstance


PROFILE = {
    "moderate": {"delay_budget_ms": 50.0, "revenue": 1.0,
                 "compute": 1.5, "throughput": 1.0},
    "strict": {"delay_budget_ms": 20.0, "revenue": 1.0,
               "compute": 1.5, "throughput": 1.0},
    "relaxed": {"delay_budget_ms": 100.0, "revenue": 1.0,
                "compute": 1.5, "throughput": 1.0},
}


@dataclass(frozen=True)
class Environment:
    seed: int
    topology: str
    areas: tuple[dict[str, Any], ...]
    sites: tuple[dict[str, Any], ...]
    groups: tuple[dict[str, Any], ...]
    operator_anchors: tuple[dict[str, Any], ...]
    paths: tuple[dict[str, Any], ...]
    raw_status: np.ndarray
    resolved_support: np.ndarray
    delay_ms: np.ndarray
    instance: EconomicInstance
    commercial_package: dict[str, bool]
    metadata: dict[str, Any]


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))


def _area_coordinates(n_areas: int, rng: np.random.Generator, topology: str) -> np.ndarray:
    columns = int(np.ceil(np.sqrt(n_areas * 1.5)))
    rows = int(np.ceil(n_areas / columns))
    grid = np.array([(120 * c / max(columns - 1, 1), 80 * r / max(rows - 1, 1))
                     for r in range(rows) for c in range(columns)])[:n_areas]
    jitter = rng.normal(0, 2.5, grid.shape)
    if topology == "multi_hotspot":
        centers = np.array([[22., 20.], [92., 24.], [62., 66.]])
        assignment = np.arange(n_areas) % len(centers)
        grid = centers[assignment] + rng.normal(0, 10, grid.shape)
    return np.clip(grid + jitter, [0, 0], [120, 80])


def _demand_weights(coords: np.ndarray, rng: np.random.Generator, pattern: str) -> np.ndarray:
    n = len(coords)
    if pattern == "uniform":
        raw = rng.uniform(.85, 1.15, n)
    elif pattern == "concentrated":
        center = coords[rng.integers(n)]
        raw = .08 + np.exp(-np.linalg.norm(coords - center, axis=1) / 15)
    else:
        centers = coords[rng.choice(n, size=min(3, n), replace=False)]
        raw = .12 + sum(np.exp(-np.linalg.norm(coords - center, axis=1) / 12)
                        for center in centers)
    return raw / raw.sum()


def _make_sites(coords: np.ndarray, rng: np.random.Generator,
                n_sites: int = 48) -> tuple[dict[str, Any], ...]:
    if n_sites < 8:
        raise ValueError("At least eight sites are needed")
    central_n = max(1, round(n_sites / 16))
    regional_n = max(2, round(3 * n_sites / 16))
    local_n = n_sites - regional_n - central_n
    sites: list[dict[str, Any]] = []
    for index in range(local_n):
        area = index % len(coords)
        point = np.clip(coords[area] + rng.normal(0, 5, 2), [0, 0], [120, 80])
        sites.append({"id": f"L{index:03d}", "tier": "LOCAL", "ecsp": index % 3,
                      "x": float(point[0]), "y": float(point[1]),
                      "fixed_cost": float(rng.uniform(135, 185)),
                      "variable_cost": float(rng.uniform(.16, .22))})
    if n_sites == 48:
        # Preserve the frozen pilot geometry exactly.
        regional_centers = np.array([
            (20 + 40 * (i % 3), 15 + 45 * (i // 3)) for i in range(regional_n)
        ])
    else:
        regional_columns = max(1, int(np.ceil(np.sqrt(1.5 * regional_n))))
        regional_rows = int(np.ceil(regional_n / regional_columns))
        regional_centers = np.array([
            (x, y)
            for y in np.linspace(12.0, 68.0, regional_rows)
            for x in np.linspace(12.0, 108.0, regional_columns)
        ])[:regional_n]
    for index, point in enumerate(regional_centers):
        point = np.clip(point + rng.normal(0, 3, 2), [0, 0], [120, 80])
        sites.append({"id": f"R{index:03d}", "tier": "REGIONAL", "ecsp": index % 3,
                      "x": float(point[0]), "y": float(point[1]),
                      "fixed_cost": float(rng.uniform(265, 345)),
                      "variable_cost": float(rng.uniform(.11, .16))})
    central_points = np.array([[10., 72.], [60., 40.], [112., 8.]])
    for index in range(central_n):
        point = central_points[index % len(central_points)]
        if n_sites != 48:
            point = np.clip(point + rng.normal(0, 2.0, 2), [0, 0], [120, 80])
        sites.append({"id": f"C{index:03d}", "tier": "CENTRAL", "ecsp": index % 3,
                      "x": float(point[0]), "y": float(point[1]),
                      "fixed_cost": float(rng.uniform(480, 620)),
                      "variable_cost": float(rng.uniform(.07, .11))})
    return tuple(sites)


def _make_anchors(coords: np.ndarray, n_mnos: int,
                  rng: np.random.Generator) -> tuple[dict[str, Any], ...]:
    anchors: list[dict[str, Any]] = []
    region_count = 6
    order = np.argsort(coords[:, 0] + 1.5 * coords[:, 1])
    regions = np.array_split(order, region_count)
    for mno in range(n_mnos):
        for region, members in enumerate(regions):
            center = coords[members].mean(axis=0) + rng.normal(0, 3.5, 2)
            anchors.append({"id": f"M{mno}_A{region}", "mno": mno, "kind": "REGIONAL",
                            "region": region, "x": float(center[0]), "y": float(center[1])})
        core = np.array([60., 40.]) + rng.normal(0, 15, 2)
        anchors.append({"id": f"M{mno}_CORE", "mno": mno, "kind": "CORE",
                        "region": -1, "x": float(core[0]), "y": float(core[1])})
    return tuple(anchors)


def _area_regions(coords: np.ndarray) -> np.ndarray:
    order = np.argsort(coords[:, 0] + 1.5 * coords[:, 1])
    regions = np.empty(len(coords), dtype=int)
    for region, members in enumerate(np.array_split(order, 6)):
        regions[members] = region
    return regions


def _partnership_probability(topology: str, mno: int, ecsp: int, tier: str) -> float:
    tier_base = {"LOCAL": .48, "REGIONAL": .68, "CENTRAL": .95}[tier]
    if topology == "asymmetric":
        preferred = (mno + 1) % 3
        return np.clip(tier_base + (.23 if ecsp == preferred else -.30), .08, .99)
    if topology == "multi_hotspot":
        return np.clip(tier_base + (.08 if mno == ecsp else -.10), .1, .99)
    return tier_base


def _generate_paths(areas: tuple[dict[str, Any], ...], sites: tuple[dict[str, Any], ...],
                    anchors: tuple[dict[str, Any], ...], n_mnos: int,
                    topology: str, seed: int,
                    premium: dict[str, bool]) -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    rng = np.random.default_rng(seed)
    coords = np.array([(area["x"], area["y"]) for area in areas])
    regions = _area_regions(coords)
    anchor_lookup = {(a["mno"], a["kind"], a["region"]): a for a in anchors}
    g = len(areas) * n_mnos
    delays = np.full((g, len(sites)), np.inf)
    paths: list[dict[str, Any]] = []
    site_policy: dict[tuple[int, int], dict[str, Any]] = {}
    for mno in range(n_mnos):
        regional_anchors = [anchor_lookup[(mno, "REGIONAL", region)] for region in range(6)]
        for site_index, site in enumerate(sites):
            site_point = (site["x"], site["y"])
            ordered_regions = sorted(
                range(6),
                key=lambda region: _distance(
                    site_point,
                    (regional_anchors[region]["x"], regional_anchors[region]["y"]),
                ),
            )
            interconnect = rng.random() < _partnership_probability(
                topology, mno, int(site["ecsp"]), str(site["tier"])
            )
            policy_denied = rng.random() < (.05 if topology == "regular" else .10)
            condition = None
            if interconnect and site["tier"] != "CENTRAL" and rng.random() < .20:
                condition = "premium_breakout"
            exposed_regions = set()
            if interconnect and site["tier"] == "LOCAL":
                exposed_regions.add(ordered_regions[0])
            elif interconnect and site["tier"] == "REGIONAL":
                exposed_regions.update(ordered_regions[:2])
            site_policy[(mno, site_index)] = {
                "interconnect": interconnect,
                "policy_denied": policy_denied,
                "condition": condition,
                "exposed_regions": exposed_regions,
            }
    for area_index, area in enumerate(areas):
        area_point = (area["x"], area["y"])
        for mno in range(n_mnos):
            group = area_index * n_mnos + mno
            access = 5.0 + .7 * mno + rng.uniform(0, 2)
            regional = anchor_lookup[(mno, "REGIONAL", int(regions[area_index]))]
            core = anchor_lookup[(mno, "CORE", -1)]
            for site_index, site in enumerate(sites):
                site_point = (site["x"], site["y"])
                policy = site_policy[(mno, site_index)]
                direct = int(regions[area_index]) in policy["exposed_regions"]
                policy_denied = policy["policy_denied"]
                condition = policy["condition"] if direct else None
                if policy_denied:
                    kind, allowed, delay, anchor_id = "DENIED", False, np.inf, None
                elif direct:
                    delay = (access + 3.5 + .075 * _distance(area_point, (regional["x"], regional["y"]))
                             + .075 * _distance((regional["x"], regional["y"]), site_point)
                             + rng.uniform(0, 2.5))
                    kind, allowed, anchor_id = "LOCAL_BREAKOUT", True, regional["id"]
                else:
                    delay = (access + 30.0 + .15 * _distance(area_point, (core["x"], core["y"]))
                             + .15 * _distance((core["x"], core["y"]), site_point)
                             + rng.uniform(1, 5))
                    kind, allowed, anchor_id = "CORE_HAIRPIN", True, core["id"]
                    condition = None
                delays[group, site_index] = delay
                paths.append({"group": group, "area": area_index, "mno": mno,
                              "site": site_index, "kind": kind, "anchor": anchor_id,
                              "allowed": allowed, "delay_ms": None if not np.isfinite(delay) else float(delay),
                              "condition": condition,
                              "condition_satisfied": condition is None or premium[f"MNO_{mno}"],
                              "site_interconnected": bool(policy["interconnect"]),
                              "breakout_regions": sorted(policy["exposed_regions"])})
    return tuple(paths), delays


def _resolve(paths: tuple[dict[str, Any], ...], delays: np.ndarray,
             delay_budget: float, groups: int, sites: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.empty((groups, sites), dtype=object)
    support = np.zeros((groups, sites), dtype=float)
    for path in paths:
        g, s = int(path["group"]), int(path["site"])
        if not path["allowed"]:
            raw[g, s] = "UNKNOWN" if path["kind"] == "DENIED" else "UNVIABLE"
        elif delays[g, s] > delay_budget:
            raw[g, s] = "UNVIABLE"
        elif path["condition"] is not None:
            raw[g, s] = "PARTIAL"
            support[g, s] = float(path["condition_satisfied"])
        else:
            raw[g, s] = "VIABLE"
            support[g, s] = 1.0
    return raw, support


def _capacities(sites: tuple[dict[str, Any], ...], total_compute: float,
                total_throughput: float, compute_ratio: float,
                throughput_ratio: float, n_mnos: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array([{"LOCAL": 1.0, "REGIONAL": 2.4, "CENTRAL": 5.5}[s["tier"]]
                        * rng.uniform(.85, 1.15) for s in sites])
    compute = compute_ratio * total_compute * weights / weights.sum()
    throughput = throughput_ratio * total_throughput * weights / weights.sum()
    shares = rng.dirichlet(np.full(n_mnos, 5.0), size=len(sites)).T
    # Contracted quotas sum to more than the interface capacity so they shape
    # operator use without manufacturing additional aggregate throughput.
    quotas = shares * throughput[None, :] * 1.35
    return compute, throughput, quotas


def _budget_levels(sites: tuple[dict[str, Any], ...], max_variable: float) -> tuple[float, ...]:
    all_fixed = sum(float(site["fixed_cost"]) for site in sites)
    ceiling = all_fixed + max_variable
    return tuple(float(fraction * ceiling) for fraction in (.08, .15, .25, .40, 1.05))


def build_environment(seed: int, topology: str = "regular", n_areas: int = 24,
                      n_mnos: int = 3, n_sites: int = 48,
                      profile: str = "moderate", demand_pattern: str = "hotspots",
                      compute_ratio: float = 3.0, throughput_ratio: float = 3.0,
                      operator_shares: np.ndarray | None = None,
                      total_demand: float = 10_000.0) -> Environment:
    if topology not in {"regular", "multi_hotspot", "asymmetric"}:
        raise ValueError("Unknown topology family")
    if profile not in PROFILE:
        raise ValueError("Unknown application profile")
    rng = np.random.default_rng(seed)
    coords = _area_coordinates(n_areas, rng, topology)
    weights = _demand_weights(coords, rng, demand_pattern)
    areas = tuple({"id": f"A{i:03d}", "x": float(point[0]), "y": float(point[1]),
                   "demand_weight": float(weights[i])} for i, point in enumerate(coords))
    sites = _make_sites(coords, rng, n_sites)
    anchors = _make_anchors(coords, n_mnos, rng)
    if operator_shares is None:
        operator_shares = np.array([.45, .35, .20]) if n_mnos == 3 else np.ones(n_mnos) / n_mnos
    operator_shares = np.asarray(operator_shares, dtype=float)
    operator_shares /= operator_shares.sum()
    if total_demand <= 0:
        raise ValueError("total_demand must be positive")
    spec = PROFILE[profile]
    groups: list[dict[str, Any]] = []
    for area_index, area in enumerate(areas):
        for mno in range(n_mnos):
            groups.append({"id": f"{area['id']}:MNO_{mno}:{profile}", "area": area_index,
                           "mno": mno, "profile": profile,
                           "demand": total_demand * area["demand_weight"] * operator_shares[mno],
                           "revenue": spec["revenue"], "compute": spec["compute"],
                           "throughput": spec["throughput"]})
    premium = {f"MNO_{mno}": mno % 2 == 0 for mno in range(n_mnos)}
    paths, delays = _generate_paths(areas, sites, anchors, n_mnos, topology,
                                    seed + 1_000_003, premium)
    raw, support = _resolve(paths, delays, spec["delay_budget_ms"], len(groups), len(sites))
    demand = np.array([group["demand"] for group in groups])
    compute_unit = np.array([group["compute"] for group in groups])
    throughput_unit = np.array([group["throughput"] for group in groups])
    compute, interface, quota = _capacities(
        sites, float(demand @ compute_unit), float(demand @ throughput_unit),
        compute_ratio, throughput_ratio, n_mnos, rng)
    fixed = np.array([site["fixed_cost"] for site in sites])
    site_variable = np.array([site["variable_cost"] for site in sites])
    # Serving cost depends on the purchased site but service revenue does not.
    distance_cost = np.array([[
        .00025 * _distance((areas[group["area"]]["x"], areas[group["area"]]["y"]),
                           (site["x"], site["y"])) for site in sites] for group in groups])
    variable = site_variable[None, :] + distance_cost
    max_variable = float(np.max(variable) * demand.sum())
    budgets = _budget_levels(sites, max_variable)
    instance = EconomicInstance(
        site_ids=tuple(site["id"] for site in sites),
        group_ids=tuple(group["id"] for group in groups),
        group_mno=np.array([group["mno"] for group in groups], dtype=int),
        demand=demand, revenue=np.array([group["revenue"] for group in groups]),
        compute_per_unit=compute_unit, throughput_per_unit=throughput_unit,
        fixed_cost=fixed, variable_cost=variable, compute_capacity=compute,
        interface_capacity=interface, contracted_quota=quota,
        support=support, budget=budgets[2], site_tiers=tuple(site["tier"] for site in sites))
    metadata = {"profile": profile, "delay_budget_ms": spec["delay_budget_ms"],
                "demand_pattern": demand_pattern, "compute_ratio": compute_ratio,
                "throughput_ratio": throughput_ratio, "budget_levels": budgets,
                "total_demand": total_demand, "n_mnos": n_mnos,
                "conditions_resolved_before_planning": True,
                "truthful_authoritative_declarations": True}
    return Environment(seed, topology, areas, sites, tuple(groups), anchors, paths,
                       raw, support, delays, instance, premium, metadata)


def resample_support(environment: Environment, seed: int) -> np.ndarray:
    """Sample another operator path realization over the same public catalog."""
    paths, delays = _generate_paths(environment.areas, environment.sites,
                                    environment.operator_anchors,
                                    environment.metadata["n_mnos"], environment.topology,
                                    seed, environment.commercial_package)
    _, support = _resolve(paths, delays, environment.metadata["delay_budget_ms"],
                          len(environment.groups), len(environment.sites))
    return support


def catalog_support(environment: Environment, calibration: dict[str, float]) -> np.ndarray:
    """Geographic/tier estimate with no current operator path information."""
    result = np.zeros_like(environment.resolved_support)
    for g, group in enumerate(environment.groups):
        area = environment.areas[group["area"]]
        for s, site in enumerate(environment.sites):
            distance = _distance((area["x"], area["y"]), (site["x"], site["y"]))
            predicted = (calibration["intercept"] + calibration["distance"] * distance
                         + calibration[f"tier_{site['tier'].lower()}"])
            result[g, s] = float(predicted <= environment.metadata["delay_budget_ms"])
    return result


def build_overlap_environment(seed: int, overlap: str, pattern: str,
                              alternatives: int = 6) -> Environment:
    """Matched policy/path environments for operator-overlap experiments."""
    if overlap not in {"aligned", "intermediate", "disjoint"}:
        raise ValueError("Unknown overlap structure")
    if pattern not in {"clustered", "dispersed"}:
        raise ValueError("Unknown support geography")
    base = build_environment(seed, topology="regular", demand_pattern="uniform")
    paths, delays, raw, support = _matched_realization(
        base, overlap, pattern, alternatives, seed + 7_000_001
    )
    metadata = dict(base.metadata)
    metadata.update(overlap=overlap, support_pattern=pattern,
                    supported_alternatives_per_group=alternatives,
                    matched_support_count=True,
                    support_realization_seed=seed + 7_000_001)
    instance = base.instance.with_support(support)
    return replace(base, topology=f"matched_{overlap}_{pattern}", paths=paths,
                   delay_ms=delays, raw_status=raw, resolved_support=support,
                   instance=instance, metadata=metadata)


def _matched_realization(base: Environment, overlap: str, pattern: str,
                         alternatives: int, policy_seed: int
                         ) -> tuple[tuple[dict[str, Any], ...], np.ndarray, np.ndarray, np.ndarray]:
    """Derive one support map from a matched operator-policy realization."""
    rng = np.random.default_rng(policy_seed)
    n_mnos = base.metadata["n_mnos"]
    paths = [dict(path) for path in base.paths]
    delays = np.full_like(base.delay_ms, np.inf)
    by_key = {(path["group"], path["site"]): path for path in paths}
    all_sites = np.arange(len(base.sites))
    shuffled = rng.permutation(all_sites)
    if overlap == "intermediate":
        common_pool = shuffled[:12]
        operator_pools = np.array_split(shuffled[12:], n_mnos)
    elif overlap == "disjoint":
        common_pool = np.array([], dtype=int)
        operator_pools = np.array_split(shuffled, n_mnos)
    else:
        common_pool = all_sites
        operator_pools = [all_sites] * n_mnos

    def choose(candidates: np.ndarray, count: int, distances: np.ndarray) -> np.ndarray:
        eligible = np.asarray([
            site for site in candidates
            if 8.0 + .12 * distances[site] <= base.metadata["delay_budget_ms"]
        ], dtype=int)
        if len(eligible) < count:
            raise RuntimeError("Insufficient delay-feasible sites in an operator policy pool")
        ordered = eligible[np.argsort(distances[eligible] + rng.normal(0, 3.0, len(eligible)))]
        if pattern == "clustered" or count == len(ordered):
            return ordered[:count]
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        return ordered[positions]

    for area_index, area in enumerate(base.areas):
        distances = np.array([_distance((area["x"], area["y"]), (site["x"], site["y"]))
                              for site in base.sites])
        shared_count = (alternatives // 2 if overlap == "intermediate"
                        else alternatives if overlap == "aligned" else 0)
        shared_sites = (choose(common_pool, shared_count, distances)
                        if shared_count else np.array([], dtype=int))
        for mno in range(n_mnos):
            if overlap == "aligned":
                chosen = shared_sites
            elif overlap == "disjoint":
                chosen = choose(operator_pools[mno], alternatives, distances)
            else:
                unique_sites = choose(operator_pools[mno], alternatives - shared_count, distances)
                chosen = np.r_[shared_sites, unique_sites]
            group = area_index * n_mnos + mno
            for site_index in range(len(base.sites)):
                path = by_key[(group, site_index)]
                if site_index in chosen:
                    delay = 8.0 + .12 * distances[site_index]
                    path.update(kind="DECLARED_DIRECT", anchor=f"M{mno}_POLICY_A{area_index}",
                                allowed=True, delay_ms=float(delay), condition=None,
                                condition_satisfied=True)
                    delays[group, site_index] = delay
                else:
                    path.update(kind="POLICY_UNSUPPORTED", anchor=None, allowed=False,
                                delay_ms=None, condition=None, condition_satisfied=True)
    raw, support = _resolve(tuple(paths), delays, base.metadata["delay_budget_ms"],
                            len(base.groups), len(base.sites))
    if not np.all(support.sum(axis=1) == alternatives):
        raise AssertionError("Matched support count was not preserved")
    return tuple(paths), delays, raw, support


def resample_overlap_support(environment: Environment, seed: int) -> np.ndarray:
    """Sample the known overlap family without revealing its current support map."""
    _, _, _, support = _matched_realization(
        environment, str(environment.metadata["overlap"]),
        str(environment.metadata["support_pattern"]),
        int(environment.metadata["supported_alternatives_per_group"]), seed,
    )
    return support


def build_composition_controlled_overlap_environment(
    seed: int, overlap: str, pattern: str,
) -> Environment:
    """Matched overlap environment with identical tier/resource composition.

    Every area-MNO group receives four local, one regional, and one central
    supported alternative. Costs and capacities are normalized within tier so
    the overlap comparison is not driven by a different usable catalog mix.
    """
    if overlap not in {"aligned", "intermediate", "disjoint"}:
        raise ValueError("Unknown overlap structure")
    if pattern not in {"clustered", "dispersed"}:
        raise ValueError("Unknown support geography")
    base = build_environment(seed, topology="regular", demand_pattern="uniform")
    rng = np.random.default_rng(seed + 11_000_003)
    tiers = np.asarray(base.instance.site_tiers)
    tier_indices = {tier: np.flatnonzero(tiers == tier) for tier in ("LOCAL", "REGIONAL", "CENTRAL")}
    quotas = {"LOCAL": 4, "REGIONAL": 1, "CENTRAL": 1}
    n_mnos = int(base.metadata["n_mnos"])

    pools: dict[str, dict[str, Any]] = {}
    for tier, indices in tier_indices.items():
        shuffled = rng.permutation(indices)
        if overlap == "aligned":
            pools[tier] = {"shared": shuffled, "operators": [shuffled] * n_mnos}
        elif overlap == "disjoint":
            pools[tier] = {"shared": np.array([], dtype=int),
                           "operators": list(np.array_split(shuffled, n_mnos))}
        else:
            shared_count = 6 if tier == "LOCAL" else 0
            pools[tier] = {"shared": shuffled[:shared_count],
                           "operators": list(np.array_split(shuffled[shared_count:], n_mnos))}

    support = np.zeros_like(base.resolved_support)
    paths = [dict(path) for path in base.paths]
    path_lookup = {(int(path["group"]), int(path["site"])): path for path in paths}
    delays = np.full_like(base.delay_ms, np.inf)

    def choose(candidates: np.ndarray, count: int, distance: np.ndarray) -> np.ndarray:
        if len(candidates) < count:
            raise RuntimeError("Insufficient candidates for composition-controlled support")
        ordered = candidates[np.argsort(distance[candidates])]
        if pattern == "clustered" or count == 1:
            return ordered[:count]
        positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
        return ordered[positions]

    for area_index, area in enumerate(base.areas):
        distance = np.array([
            _distance((area["x"], area["y"]), (site["x"], site["y"]))
            for site in base.sites
        ])
        aligned_choice = {
            tier: choose(tier_indices[tier], count, distance)
            for tier, count in quotas.items()
        }
        shared_local = (
            choose(pools["LOCAL"]["shared"], 2, distance)
            if overlap == "intermediate" else np.array([], dtype=int)
        )
        for mno in range(n_mnos):
            if overlap == "aligned":
                chosen = np.concatenate(list(aligned_choice.values()))
            elif overlap == "disjoint":
                chosen = np.concatenate([
                    choose(pools[tier]["operators"][mno], count, distance)
                    for tier, count in quotas.items()
                ])
            else:
                chosen = np.concatenate([
                    shared_local,
                    choose(pools["LOCAL"]["operators"][mno], 2, distance),
                    choose(pools["REGIONAL"]["operators"][mno], 1, distance),
                    choose(pools["CENTRAL"]["operators"][mno], 1, distance),
                ])
            group = area_index * n_mnos + mno
            support[group, chosen] = 1.0
            for site_index in range(len(base.sites)):
                path = path_lookup[(group, site_index)]
                if support[group, site_index]:
                    delay = 8.0 + .12 * distance[site_index]
                    path.update(kind="DECLARED_DIRECT", allowed=True, delay_ms=float(delay),
                                condition=None, condition_satisfied=True,
                                anchor=f"M{mno}_CONTROL_A{area_index}")
                    delays[group, site_index] = delay
                else:
                    path.update(kind="POLICY_UNSUPPORTED", allowed=False, delay_ms=None,
                                condition=None, condition_satisfied=True, anchor=None)
    raw, resolved = _resolve(tuple(paths), delays, base.metadata["delay_budget_ms"],
                             len(base.groups), len(base.sites))
    np.testing.assert_array_equal(resolved, support)
    np.testing.assert_allclose(support.sum(axis=1), 6.0)
    for tier, count in quotas.items():
        np.testing.assert_allclose(support[:, tiers == tier].sum(axis=1), count)

    fixed = base.instance.fixed_cost.copy()
    compute = base.instance.compute_capacity.copy()
    interface = base.instance.interface_capacity.copy()
    variable = base.instance.variable_cost.copy()
    site_variable = np.asarray([site["variable_cost"] for site in base.sites])
    for tier, indices in tier_indices.items():
        fixed[indices] = fixed[indices].mean()
        compute[indices] = compute[indices].mean()
        interface[indices] = interface[indices].mean()
        distance_component = variable[:, indices] - site_variable[indices][None, :]
        variable[:, indices] = distance_component + site_variable[indices].mean()
    contracted = np.repeat((interface * 1.35 / n_mnos)[None, :], n_mnos, axis=0)
    # Budget levels must reflect normalized fixed costs, not the original site dictionaries.
    ceiling = float(fixed.sum() + np.max(variable) * base.instance.demand.sum())
    budgets = tuple(float(fraction * ceiling) for fraction in (.08, .15, .25, .40, 1.05))
    instance = replace(
        base.instance, fixed_cost=fixed, variable_cost=variable,
        compute_capacity=compute, interface_capacity=interface,
        contracted_quota=contracted, support=resolved, budget=budgets[2],
    )
    metadata = dict(base.metadata)
    metadata.update(
        overlap=overlap, support_pattern=pattern, supported_alternatives_per_group=6,
        matched_support_count=True, composition_controlled=True,
        supported_tier_composition={"LOCAL": 4, "REGIONAL": 1, "CENTRAL": 1},
        budget_levels=budgets,
    )
    return replace(base, topology=f"controlled_{overlap}_{pattern}", paths=tuple(paths),
                   delay_ms=delays, raw_status=raw, resolved_support=resolved,
                   instance=instance, metadata=metadata)


def environment_to_dict(environment: Environment) -> dict[str, Any]:
    instance = asdict(environment.instance)
    for key, value in list(instance.items()):
        if isinstance(value, np.ndarray):
            instance[key] = value.tolist()
        elif isinstance(value, tuple):
            instance[key] = list(value)
    return {"seed": environment.seed, "topology": environment.topology,
            "areas": list(environment.areas), "sites": list(environment.sites),
            "groups": list(environment.groups), "operator_anchors": list(environment.operator_anchors),
            "paths": list(environment.paths), "raw_status": environment.raw_status.tolist(),
            "resolved_support": environment.resolved_support.tolist(),
            "delay_ms": np.where(np.isfinite(environment.delay_ms), environment.delay_ms, -1).tolist(),
            "instance": instance, "commercial_package": environment.commercial_package,
            "metadata": environment.metadata}


def save_environment(environment: Environment, path: Path) -> None:
    path.write_text(json.dumps(environment_to_dict(environment), indent=2,
                               allow_nan=False) + "\n")
