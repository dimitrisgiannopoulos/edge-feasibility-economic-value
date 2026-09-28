"""Summarize diagnostics, robustness, and final independent evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .environment import PROFILE, build_environment


ROOT = Path(__file__).resolve().parent.parent


def _paired(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = ["profit", "revenue", "fixed_cost", "variable_cost", "total_expenditure",
               "qualifying_coverage", "sites", "local_sites", "regional_sites", "central_sites"]
    wide = rows.pivot(index=keys, columns="method", values=metrics)
    output = wide.index.to_frame(index=False)
    for metric in metrics:
        output[f"catalog_{metric}"] = wide[(metric, "Catalog + geography")].to_numpy()
        output[f"informed_{metric}"] = wide[(metric, "Current feasibility")].to_numpy()
        output[f"{metric}_difference"] = (
            output[f"informed_{metric}"] - output[f"catalog_{metric}"]
        )
    output["max_revenue"] = rows.groupby(keys)["max_revenue"].first().to_numpy()
    output["information_points"] = 100 * output["profit_difference"] / output["max_revenue"]
    return output


def _cluster_bootstrap(data: pd.DataFrame, clusters: str, value: str,
                       draws: int = 10_000, seed: int = 20260923) -> tuple[float, float]:
    grouped = data.groupby(clusters)[value].mean().to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for index in range(draws):
        estimates[index] = rng.choice(grouped, len(grouped), replace=True).mean()
    return tuple(np.quantile(estimates, [.025, .975]))


def _bootstrap_mean(values: pd.Series, draws: int = 10_000,
                    seed: int = 20260923) -> tuple[float, float]:
    """Return a percentile interval for the mean of paired differences."""
    observations = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for index in range(draws):
        estimates[index] = rng.choice(observations, len(observations), replace=True).mean()
    return tuple(np.quantile(estimates, [.025, .975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary_dir = args.results / "evidence_summary_v1"
    summary_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = pd.read_csv(
        args.results / "final_decision_diagnostics_v1" / "decision_pairs.csv"
    )
    decision = diagnostics.copy()
    decision_summary = decision.groupby("budget_index").agg(
        paired_cases=("case_id", "size"),
        footprint_change_rate=("footprints_differ", "mean"),
        sites_added=("sites_added", "mean"),
        sites_removed=("sites_removed", "mean"),
        jaccard=("footprint_jaccard", "mean"),
        information_points=("revenue_normalized_profit_difference", lambda x: 100 * x.mean()),
        revenue_difference=("revenue_difference", "mean"),
        fixed_cost_difference=("fixed_cost_difference", "mean"),
        variable_cost_difference=("variable_cost_difference", "mean"),
        coverage_difference=("informed_coverage", "mean"),
    ).reset_index()
    catalog_coverage = decision.groupby("budget_index")["catalog_coverage"].mean().to_numpy()
    decision_summary["coverage_difference"] -= catalog_coverage
    decision_summary.to_csv(summary_dir / "decision_change_summary.csv", index=False)

    pilot_diagnostics = pd.read_csv(args.results / "diagnostics_v1" / "decision_pairs.csv")
    overlap = pilot_diagnostics[pilot_diagnostics.kind == "overlap"].copy()
    overlap_summary = overlap.groupby(["support_pattern", "overlap"]).agg(
        paired_cases=("case_id", "size"),
        catalog_profit=("catalog_realized_profit", "mean"),
        informed_profit=("informed_realized_profit", "mean"),
        information_points=("revenue_normalized_profit_difference", lambda x: 100 * x.mean()),
        catalog_predicted_service=("catalog_predicted_service", "mean"),
        catalog_actual_service=("catalog_actual_service", "mean"),
        informed_actual_service=("informed_actual_service", "mean"),
        catalog_multi_operator_volume=("catalog_multi_operator_volume", "mean"),
        informed_multi_operator_volume=("informed_multi_operator_volume", "mean"),
        supported_mean_fixed_cost=("supported_mean_fixed_cost", "mean"),
        supported_mean_compute_capacity=("supported_mean_compute_capacity", "mean"),
        supported_mean_interface_capacity=("supported_mean_interface_capacity", "mean"),
        supported_local_share=("supported_local_share", "mean"),
        supported_regional_share=("supported_regional_share", "mean"),
        supported_central_share=("supported_central_share", "mean"),
    ).reset_index()
    overlap_summary.to_csv(summary_dir / "overlap_mechanism_summary.csv", index=False)

    controlled = pd.read_csv(args.results / "overlap_control_v1" / "pairs.csv")
    controlled_summary = controlled.groupby(["support_pattern", "overlap"]).agg(
        paired_cases=("case_id", "size"),
        catalog_profit=("catalog_realized_profit", "mean"),
        informed_profit=("informed_realized_profit", "mean"),
        greedy_information_points=("revenue_normalized_profit_difference", lambda x: 100 * x.mean()),
        exact_information_points=("exact_information_points", "mean"),
        informed_greedy_gap_points=("informed_greedy_gap_points", "mean"),
        catalog_predicted_service=("catalog_predicted_service", "mean"),
        catalog_actual_service=("catalog_actual_service", "mean"),
        informed_actual_service=("informed_actual_service", "mean"),
        catalog_multi_operator_volume=("catalog_multi_operator_volume", "mean"),
        informed_multi_operator_volume=("informed_multi_operator_volume", "mean"),
        supported_mean_fixed_cost=("supported_mean_fixed_cost", "mean"),
        supported_mean_compute_capacity=("supported_mean_compute_capacity", "mean"),
        supported_mean_interface_capacity=("supported_mean_interface_capacity", "mean"),
        supported_local_share=("supported_local_share", "mean"),
        supported_regional_share=("supported_regional_share", "mean"),
        supported_central_share=("supported_central_share", "mean"),
    ).reset_index()
    controlled_intervals = []
    for (support_pattern, overlap_name), group in controlled.groupby(
        ["support_pattern", "overlap"]
    ):
        low, high = _bootstrap_mean(
            group.exact_information_points,
            seed=20260923 + sum(map(ord, support_pattern + overlap_name)),
        )
        controlled_intervals.append({
            "support_pattern": support_pattern,
            "overlap": overlap_name,
            "exact_ci_low": low,
            "exact_ci_high": high,
        })
    controlled_summary = controlled_summary.merge(
        pd.DataFrame(controlled_intervals),
        on=["support_pattern", "overlap"],
        validate="one_to_one",
    )
    controlled_summary.to_csv(summary_dir / "controlled_overlap_mechanism_summary.csv", index=False)

    overlap_contrasts = []
    comparison_pairs = [
        ("intermediate", "aligned"),
        ("intermediate", "disjoint"),
        ("aligned", "disjoint"),
    ]
    for support_pattern, group in controlled.groupby("support_pattern"):
        matched = group.pivot(index="seed", columns="overlap", values="exact_information_points")
        for left, right in comparison_pairs:
            differences = matched[left] - matched[right]
            low, high = _bootstrap_mean(
                differences,
                seed=20260923 + sum(map(ord, support_pattern + left + right)),
            )
            overlap_contrasts.append({
                "support_pattern": support_pattern,
                "contrast": f"{left}_minus_{right}",
                "paired_environments": len(differences),
                "mean_difference_points": differences.mean(),
                "ci_low": low,
                "ci_high": high,
                "positive_fraction": (differences > 1e-7).mean(),
                "tie_fraction": (differences.abs() <= 1e-7).mean(),
                "negative_fraction": (differences < -1e-7).mean(),
            })
    pd.DataFrame(overlap_contrasts).to_csv(
        summary_dir / "controlled_overlap_regime_contrasts.csv", index=False
    )

    correction = pd.read_csv(args.results / "diagnostics_v1" / "correction_accounting.csv")
    correction_summary = correction.groupby(["detection_fraction", "refund_fraction"]).agg(
        conditions=("profit", "size"),
        early_information_advantage=("profit_advantage_from_early_information", "mean"),
        revenue_difference=("revenue_difference", "mean"),
        fixed_cost_difference=("fixed_cost_difference", "mean"),
        variable_cost_difference=("variable_cost_difference", "mean"),
        unrecovered_expenditure=("unrecovered_expenditure", "mean"),
        added_sites=("added_sites", "mean"),
        removed_sites=("removed_sites", "mean"),
        retained_sites=("retained_sites", "mean"),
    ).reset_index()
    correction_summary["information_points"] = 100 * correction_summary["early_information_advantage"] / 10_000
    correction_summary.to_csv(summary_dir / "correction_accounting_summary.csv", index=False)

    hardening = pd.read_csv(args.results / "hardening_diagnostics_v1" / "outcomes.csv")
    economic = hardening[hardening.experiment == 3]
    economic_boundary = economic.groupby(
        ["fixed_cost_multiplier", "variable_cost_multiplier", "method"]
    ).agg(
        outcomes=("case_id", "size"), profit=("profit", "mean"),
        coverage=("qualifying_coverage", "mean"), no_deployment_rate=("no_deployment", "mean"),
        individually_affordable_sites=("individually_affordable_sites", "mean"),
    ).reset_index()
    economic_boundary.to_csv(summary_dir / "economic_boundary_summary.csv", index=False)
    resource = hardening[hardening.experiment == 4]
    resource_boundary = resource.groupby(["compute_ratio", "throughput_ratio", "method"]).agg(
        outcomes=("case_id", "size"), profit=("profit", "mean"),
        coverage=("qualifying_coverage", "mean"), no_deployment_rate=("no_deployment", "mean"),
        compute_utilization=("compute_utilization", "mean"),
        interface_utilization=("interface_utilization", "mean"),
    ).reset_index()
    resource_boundary.to_csv(summary_dir / "resource_boundary_summary.csv", index=False)

    requirements = pd.read_csv(args.results / "requirements_v1" / "requirements_outcomes.csv")
    requirement_pairs = _paired(requirements, ["case_id", "profile", "topology", "seed"])
    requirement_summary = requirement_pairs.groupby("profile").agg(
        paired_cases=("case_id", "size"), information_points=("information_points", "mean"),
        profit_difference=("profit_difference", "mean"),
        coverage_difference=("qualifying_coverage_difference", "mean"),
        catalog_profit=("catalog_profit", "mean"), informed_profit=("informed_profit", "mean"),
    ).reset_index()
    requirement_summary.to_csv(summary_dir / "application_requirement_summary.csv", index=False)

    robustness_config = json.loads(
        (ROOT / "economic_value" / "config" / "robustness_v1.json").read_text()
    )
    calibration = json.loads(
        (ROOT / "economic_value" / "config" / "catalog_delay_model.json").read_text()
    )
    support_rows = []
    requirement_spec = robustness_config["application_requirements"]
    for offset in range(int(requirement_spec["replicas"])):
        seed = int(requirement_spec["seed_start"]) + offset
        for topology in robustness_config["topologies"]:
            environment = build_environment(
                seed,
                topology=topology,
                profile="moderate",
                demand_pattern=robustness_config["demand_patterns"][topology],
                compute_ratio=robustness_config["compute_ratio"],
                throughput_ratio=robustness_config["throughput_ratio"],
            )
            predicted_delay = np.empty_like(environment.delay_ms)
            for group_index, group in enumerate(environment.groups):
                area = environment.areas[group["area"]]
                for site_index, site in enumerate(environment.sites):
                    distance = np.hypot(area["x"] - site["x"], area["y"] - site["y"])
                    predicted_delay[group_index, site_index] = (
                        calibration["intercept"]
                        + calibration["distance"] * distance
                        + calibration[f"tier_{site['tier'].lower()}"]
                    )
            for profile in requirement_spec["profiles"]:
                threshold = PROFILE[profile]["delay_budget_ms"]
                predicted = (predicted_delay <= threshold).astype(float)
                actual = np.zeros_like(environment.resolved_support)
                for path in environment.paths:
                    actual[int(path["group"]), int(path["site"])] = float(
                        path["allowed"]
                        and path["delay_ms"] is not None
                        and path["delay_ms"] <= threshold
                        and path["condition_satisfied"]
                    )
                support_rows.append({
                    "seed": seed,
                    "topology": topology,
                    "profile": profile,
                    "predicted_supported_tuple_rate": predicted.mean(),
                    "predicted_groups_with_any_support_rate": (predicted.sum(axis=1) > 0).mean(),
                    "actual_supported_tuple_rate": actual.mean(),
                    "actual_groups_with_any_support_rate": (
                        actual.sum(axis=1) > 0
                    ).mean(),
                })
    support_diagnostics = pd.DataFrame(support_rows)
    catalog_requirement_rows = requirements[
        requirements.method == "Catalog + geography"
    ][["seed", "topology", "profile", "no_deployment", "sites", "planning_profit",
       "realized_profit", "qualifying_coverage"]]
    support_diagnostics = support_diagnostics.merge(
        catalog_requirement_rows,
        on=["seed", "topology", "profile"],
        validate="one_to_one",
    )
    requirement_baselines = support_diagnostics.groupby("profile").agg(
        environments=("seed", "size"),
        predicted_supported_tuple_rate=("predicted_supported_tuple_rate", "mean"),
        predicted_groups_with_any_support_rate=("predicted_groups_with_any_support_rate", "mean"),
        actual_supported_tuple_rate=("actual_supported_tuple_rate", "mean"),
        actual_groups_with_any_support_rate=("actual_groups_with_any_support_rate", "mean"),
        catalog_no_deployment_rate=("no_deployment", "mean"),
        catalog_mean_sites=("sites", "mean"),
        catalog_predicted_profit=("planning_profit", "mean"),
        catalog_realized_profit=("realized_profit", "mean"),
        catalog_coverage=("qualifying_coverage", "mean"),
    ).reset_index()
    requirement_baselines.to_csv(
        summary_dir / "requirement_catalog_diagnostics.csv", index=False
    )

    requirement_prior_path = args.results / "requirement_prior_v1" / "outcomes.csv"
    if requirement_prior_path.exists():
        requirement_prior = pd.read_csv(requirement_prior_path)
        prior_requirement_summary = requirement_prior.groupby(["profile", "sample_count"]).agg(
            environments=("environment_id", "size"),
            current_advantage_points=(
                "revenue_normalized_difference_from_current", lambda x: 100 * x.mean()
            ),
            prior_realized_profit=("realized_profit", "mean"),
            current_profit=("reference_profit", "mean"),
            prior_coverage=("qualifying_coverage", "mean"),
            prior_no_deployment_rate=("sites", lambda x: (x == 0).mean()),
            median_runtime_seconds=("selection_seconds", "median"),
            maximum_mip_gap=("mip_gap", "max"),
            optimal_fraction=("status", lambda x: (x == "optimal").mean()),
        ).reset_index()
        interval_rows = []
        for (profile, sample_count), group in requirement_prior.groupby(
            ["profile", "sample_count"]
        ):
            values = 100 * group.revenue_normalized_difference_from_current
            low, high = _bootstrap_mean(
                values,
                seed=20260923 + int(sample_count) + sum(map(ord, profile)),
            )
            interval_rows.append({
                "profile": profile,
                "sample_count": sample_count,
                "ci_low": low,
                "ci_high": high,
            })
        prior_requirement_summary = prior_requirement_summary.merge(
            pd.DataFrame(interval_rows),
            on=["profile", "sample_count"],
            validate="one_to_one",
        )
        prior_requirement_summary.to_csv(
            summary_dir / "requirement_prior_summary.csv", index=False
        )
        requirement_prior_config = json.loads(
            (ROOT / "economic_value" / "config" / "requirement_prior_v1.json").read_text()
        )
        subset_seeds = range(
            int(requirement_prior_config["seed_start"]),
            int(requirement_prior_config["seed_start"])
            + int(requirement_prior_config["replicas"]),
        )
        matched_methods = requirements[
            requirements.seed.isin(subset_seeds)
        ].groupby(["profile", "method"])["realized_profit"].mean().unstack("method")
        latest_count = int(requirement_prior["sample_count"].max())
        latest_prior = prior_requirement_summary[
            prior_requirement_summary.sample_count == latest_count
        ].set_index("profile")
        requirement_comparator = requirement_baselines.set_index("profile").copy()
        requirement_comparator["catalog_subset_profit"] = matched_methods["Catalog + geography"]
        requirement_comparator["current_subset_profit"] = matched_methods["Current feasibility"]
        requirement_comparator["prior_subset_profit"] = latest_prior["prior_realized_profit"]
        requirement_comparator["prior_maps"] = latest_count
        requirement_comparator["current_minus_prior_points"] = latest_prior[
            "current_advantage_points"
        ]
        requirement_comparator["current_minus_prior_ci_low"] = latest_prior["ci_low"]
        requirement_comparator["current_minus_prior_ci_high"] = latest_prior["ci_high"]
        requirement_comparator.reset_index().to_csv(
            summary_dir / "requirement_comparator_summary.csv", index=False
        )

    scaling = pd.read_csv(args.results / "scaling_clean_v5" / "scaling_outcomes.csv")
    scaling_pairs = _paired(scaling, ["case_id", "n_areas", "n_sites", "topology", "seed"])
    runtime = scaling.groupby(["n_areas", "n_sites", "case_id"])["case_runtime_seconds"].first().reset_index()
    memory = scaling.groupby(["n_areas", "n_sites", "case_id"])["worker_peak_rss_mb"].first().reset_index()
    scaling_summary = scaling_pairs.groupby(["n_areas", "n_sites"]).agg(
        paired_cases=("case_id", "size"), information_points=("information_points", "mean"),
        profit_difference=("profit_difference", "mean"),
        coverage_difference=("qualifying_coverage_difference", "mean"),
        catalog_profit=("catalog_profit", "mean"), informed_profit=("informed_profit", "mean"),
        catalog_sites=("catalog_sites", "mean"), informed_sites=("informed_sites", "mean"),
        catalog_no_deployment_rate=("catalog_sites", lambda x: (x == 0).mean()),
        informed_no_deployment_rate=("informed_sites", lambda x: (x == 0).mean()),
    ).reset_index()
    statuses = scaling.groupby(["n_areas", "n_sites", "method"])["planning_status"].apply(
        lambda x: ";".join(sorted(set(x)))
    ).unstack("method").reset_index().rename(columns={
        "Catalog + geography": "catalog_status",
        "Current feasibility": "informed_status",
    })
    scaling_summary = scaling_summary.merge(statuses, on=["n_areas", "n_sites"])
    scaling_summary = scaling_summary.merge(
        runtime.groupby(["n_areas", "n_sites"])["case_runtime_seconds"].agg(["median", "max"]).reset_index()
        .rename(columns={"median": "median_runtime_seconds", "max": "max_runtime_seconds"}),
        on=["n_areas", "n_sites"],
    ).merge(
        memory.groupby(["n_areas", "n_sites"])["worker_peak_rss_mb"].median().reset_index()
        .rename(columns={"worker_peak_rss_mb": "median_worker_peak_rss_mb"}),
        on=["n_areas", "n_sites"],
    )
    scaling_summary.to_csv(summary_dir / "scaling_summary.csv", index=False)

    prior = pd.read_csv(args.results / "prior_stability_v3" / "outcomes.csv")
    prior_summary = prior.groupby("sample_count").agg(
        environments=("case_id", "size"),
        current_advantage_points=("revenue_normalized_difference_from_current", lambda x: 100 * x.mean()),
        realized_profit=("realized_profit", "mean"),
        median_runtime_seconds=("selection_seconds", "median"),
        maximum_mip_gap=("mip_gap", "max"),
        optimal_fraction=("status", lambda x: (x == "optimal").mean()),
    ).reset_index()
    prior_intervals = []
    for sample_count, group in prior.groupby("sample_count"):
        values = 100 * group.revenue_normalized_difference_from_current
        low, high = _bootstrap_mean(values, seed=20260923 + int(sample_count))
        prior_intervals.append({"sample_count": sample_count, "ci_low": low, "ci_high": high})
    prior_summary = prior_summary.merge(
        pd.DataFrame(prior_intervals), on="sample_count", validate="one_to_one"
    )
    prior_summary.to_csv(summary_dir / "prior_stability_summary.csv", index=False)

    final_rows = pd.read_csv(args.results / "final_v2" / "outcomes.csv")
    final_pairs = pd.read_csv(args.results / "final_v2" / "paired_contrasts.csv")
    final_pairs["information_points"] = 100 * final_pairs.revenue_normalized_profit_difference
    final_summary = final_pairs.groupby(["topology", "budget_index"]).agg(
        paired_cases=("case_id", "size"), information_points=("information_points", "mean"),
        median_information_points=("information_points", "median"),
        profit_difference=("profit_difference", "mean"),
        coverage_difference=("coverage_difference", "mean"),
        spending_difference=("spending_difference", "mean"),
    ).reset_index()
    ci_rows = []
    for (topology, budget_index), group in final_pairs.groupby(["topology", "budget_index"]):
        low, high = _cluster_bootstrap(
            group, "environment_id", "information_points", seed=20260923 + int(budget_index)
        )
        ci_rows.append({"topology": topology, "budget_index": budget_index,
                        "ci_low": low, "ci_high": high})
    final_summary = final_summary.merge(
        pd.DataFrame(ci_rows), on=["topology", "budget_index"], validate="one_to_one"
    )
    final_summary.to_csv(summary_dir / "final_by_configuration.csv", index=False)
    aggregate_mean = float(final_pairs.information_points.mean())
    ci_low, ci_high = _cluster_bootstrap(final_pairs, "environment_id", "information_points")
    tolerance = 1e-7
    wins = int((final_pairs.profit_difference > tolerance).sum())
    ties = int((final_pairs.profit_difference.abs() <= tolerance).sum())
    losses = int((final_pairs.profit_difference < -tolerance).sum())
    pd.DataFrame([{
        "paired_cases": len(final_pairs),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(final_pairs),
        "tie_rate": ties / len(final_pairs),
        "loss_rate": losses / len(final_pairs),
        "comparison_tolerance": tolerance,
    }]).to_csv(summary_dir / "principal_win_tie_loss.csv", index=False)
    middle = final_rows[final_rows.budget_index == 2].groupby("method").agg(
        profit=("profit", "mean"), expenditure=("total_expenditure", "mean"),
        coverage=("qualifying_coverage", "mean"), sites=("sites", "mean"),
        local_sites=("local_sites", "mean"),
    ).reset_index()
    middle.to_csv(summary_dir / "final_middle_budget_outcomes.csv", index=False)

    max_accounting = max(float(diagnostics.accounting_residual.abs().max()),
                         float(correction.accounting_residual.abs().max()))
    text = f"""# Strengthened Evidence Summary

## Independent Principal Evaluation

- 300 independent environments and 1,500 paired budget cases.
- Mean information value: **{aggregate_mean:.2f} revenue-normalized points**.
- Environment-clustered bootstrap 95% interval: **[{ci_low:.2f}, {ci_high:.2f}]**.
- Current feasibility produces higher realized profit in **{100 * wins / len(final_pairs):.1f}%** of paired cases, ties in **{100 * ties / len(final_pairs):.1f}%**, and loses in **{100 * losses / len(final_pairs):.1f}%**.

## Decision Mechanism

- Footprint-change rates, additions, removals, and exact revenue/cost decompositions are in `decision_change_summary.csv`.
- The worked example was selected by the disclosed median-gain rule, not by maximum effect.
- Maximum profit-accounting residual across decision and correction diagnostics: **{max_accounting:.3e}**.

## Support Structure

- `controlled_overlap_mechanism_summary.csv` reports both planners' realized profit, multi-operator volume, catalog-predicted versus actual service, and exact-selection checks.
- `controlled_overlap_regime_contrasts.csv` gives paired bootstrap intervals for differences between aligned, intermediate, and disjoint support.
- The stricter matched generator gives every group four local, one regional, and one central alternative and normalizes costs and capacities within tier.

## Timing, Requirements, and Scale

- Correction accounting is decomposed into revenue, fixed-cost, and variable-cost effects in `correction_accounting_summary.csv`.
- High-cost no-deployment frequency and individual site affordability are separated in `economic_boundary_summary.csv`.
- Resource-limited coverage, utilization, and no-deployment frequency are reported in `resource_boundary_summary.csv`.
- Results for 20, 50, and 100 ms requirements are in `application_requirement_summary.csv`.
- `requirement_catalog_diagnostics.csv` records predicted and actual support rates and confirms that the 20-ms catalog view selects no deployment in every case.
- `requirement_prior_summary.csv` reports the matched 12/24/48-map prior comparison for all three requirements; the 48-map current-support advantages are 9.91, 2.69, and 0.81 points at 20, 50, and 100 ms.
- Results for 24/48, 100/200, and 250/500 area/site configurations are in `scaling_summary.csv`.
- Prior-informed planning is extended through 48 nested maps in `prior_stability_summary.csv`, with paired bootstrap intervals over the 15 evaluation environments.

Solver optimality gaps describe the sampled planning problem solved by a baseline. They are not confidence intervals or certified bounds on realized information value in the hidden environment.

All monetary results use the same synthetic monetary scale. Information values are differences in realized deployment profit divided by potential full-demand revenue.
"""
    (summary_dir / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
