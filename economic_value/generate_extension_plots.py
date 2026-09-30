"""Generate publication-style figures from strengthened evidence CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
COLORS = {"catalog": "#6b7280", "informed": "#007f5f", "accent": "#d97706"}


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "evidence_v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.results / "evidence_summary_v1"
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False})

    decision = pd.read_csv(summary / "decision_change_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    x = decision.budget_index.to_numpy()
    axes[0].plot(x, 100 * decision.footprint_change_rate, marker="o", color=COLORS["informed"])
    axes[0].set(xlabel="Budget level", ylabel="Pairs with changed footprint (%)", ylim=(0, 105))
    axes[0].set_xticks(x, ["Very low", "Low", "Middle", "High", "Nonbinding"])
    axes[0].tick_params(axis="x", rotation=22)
    axes[1].plot(x, decision.sites_added, marker="o", label="Added", color=COLORS["informed"])
    axes[1].plot(x, decision.sites_removed, marker="s", label="Removed", color=COLORS["accent"])
    axes[1].set(xlabel="Budget level", ylabel="Mean site changes per pair")
    axes[1].set_xticks(x, ["Very low", "Low", "Middle", "High", "Nonbinding"])
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, args.output / "decision_changes_by_budget.png")

    overlap = pd.read_csv(summary / "controlled_overlap_mechanism_summary.csv")
    order = ["aligned", "intermediate", "disjoint"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    for pattern, marker in (("clustered", "o"), ("dispersed", "s")):
        data = overlap[overlap.support_pattern == pattern].set_index("overlap").loc[order]
        axes[0].plot(order, data.catalog_profit, marker=marker, linestyle="--",
                     color=COLORS["catalog"], alpha=.8, label=f"Catalog, {pattern}")
        axes[0].plot(order, data.informed_profit, marker=marker,
                     color=COLORS["informed"], label=f"Informed, {pattern}")
        axes[1].plot(order, data.catalog_support_overestimate if "catalog_support_overestimate" in data else
                     data.catalog_predicted_service - data.catalog_actual_service,
                     marker=marker, label=pattern)
        axes[2].plot(order, data.catalog_multi_operator_volume, marker=marker, linestyle="--",
                     color=COLORS["catalog"], alpha=.8, label=f"Catalog, {pattern}")
        axes[2].plot(order, data.informed_multi_operator_volume, marker=marker,
                     color=COLORS["informed"], label=f"Informed, {pattern}")
    axes[0].set(ylabel="Realized profit", xlabel="Cross-operator support")
    axes[1].set(ylabel="Catalog service error", xlabel="Cross-operator support")
    axes[2].set(ylabel="Multi-MNO service", xlabel="Cross-operator support")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False, fontsize=6.8)
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    _save(fig, args.output / "overlap_mechanism.png")

    correction = pd.read_csv(summary / "correction_accounting_summary.csv")
    matrix = correction.pivot(index="refund_fraction", columns="detection_fraction",
                              values="information_points").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    image = ax.imshow(matrix.to_numpy(), cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [f"{100*x:g}%" for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [f"{100*x:g}%" for x in matrix.index])
    ax.set(xlabel="Detection delay (% of horizon)", ylabel="Refunded fixed commitment",
           title="Value of knowing before commitment")
    for row in range(len(matrix.index)):
        for col in range(len(matrix.columns)):
            ax.text(col, row, f"{matrix.iloc[row, col]:.1f}", ha="center", va="center",
                    color="white" if matrix.iloc[row, col] > matrix.to_numpy().max() * .55 else "black")
    fig.colorbar(image, ax=ax, label="Revenue-normalized profit points")
    fig.tight_layout()
    _save(fig, args.output / "correction_accounting.png")

    requirements = pd.read_csv(summary / "application_requirement_summary.csv")
    profile_order = ["strict", "moderate", "relaxed"]
    requirements = requirements.set_index("profile").loc[profile_order].reset_index()
    scaling = pd.read_csv(summary / "scaling_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axes[0].bar(["20 ms", "50 ms", "100 ms"], requirements.information_points,
                color=["#b91c1c", COLORS["accent"], COLORS["informed"]])
    axes[0].set(ylabel="Information value (revenue-normalized points)", xlabel="Requirement")
    axes[1].plot(scaling.n_sites, scaling.median_runtime_seconds, marker="o", color="#2563eb")
    axes[1].set(xlabel="Candidate sites", ylabel="Median pair runtime (s)")
    axes[1].set_xticks(scaling.n_sites,
                       [f"{a}/{s}" for a, s in zip(scaling.n_areas, scaling.n_sites)])
    axes[1].set_title("Areas/sites")
    fig.tight_layout()
    _save(fig, args.output / "requirements_and_scaling.png")

    requirement_comparator_path = summary / "requirement_comparator_summary.csv"
    if requirement_comparator_path.exists():
        comparator = pd.read_csv(requirement_comparator_path).set_index("profile").loc[
            profile_order
        ]
        labels = ["20 ms", "50 ms", "100 ms"]
        positions = np.arange(len(labels))
        width = .25
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
        effects = comparator.current_minus_prior_points.to_numpy()
        lower = comparator.current_minus_prior_ci_low.to_numpy()
        upper = comparator.current_minus_prior_ci_high.to_numpy()
        axes[0].axhline(0, color=COLORS["catalog"], linewidth=.8)
        axes[0].errorbar(
            positions, effects, yerr=np.vstack((effects - lower, upper - effects)),
            fmt="o", markersize=7, capsize=4, linewidth=1.6,
            color=COLORS["informed"],
        )
        for x, value, high in zip(positions, effects, upper):
            axes[0].annotate(f"{value:.2f}", (x, high), xytext=(0, 5),
                             textcoords="offset points", ha="center", fontsize=8)
        axes[0].set_xticks(positions, labels)
        axes[0].set(ylabel="Gain vs 48-map prior\n(profit points)",
                    xlabel="Delay requirement", ylim=(-.8, max(upper) + 2.0))

        axes[1].bar(
            positions - width,
            100 * comparator.predicted_supported_tuple_rate,
            width,
            label="Catalog-predicted tuples",
            color="#2563eb",
        )
        axes[1].bar(
            positions,
            100 * comparator.actual_supported_tuple_rate,
            width,
            label="Actually supported tuples",
            color=COLORS["informed"],
        )
        axes[1].bar(
            positions + width,
            100 * comparator.catalog_no_deployment_rate,
            width,
            label="Catalog no deployment",
            color="#b91c1c",
        )
        axes[1].set_xticks(positions, labels)
        axes[1].set(
            ylabel="Share of tuples or cases (%)",
            xlabel="Delay requirement",
            ylim=(0, 105),
        )
        axes[1].legend(frameon=False, fontsize=6.8, loc="upper center",
                       bbox_to_anchor=(.5, -.27), ncol=3)
        fig.tight_layout(rect=(0, .13, 1, 1))
        _save(fig, args.output / "requirement_baseline_and_prior.png")

    prior = pd.read_csv(summary / "prior_stability_summary.csv")
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.errorbar(
        prior.sample_count,
        prior.current_advantage_points,
        yerr=[
            prior.current_advantage_points - prior.ci_low,
            prior.ci_high - prior.current_advantage_points,
        ],
        marker="o",
        capsize=3,
        color=COLORS["informed"],
    )
    ax.set(xlabel="Support maps sampled by prior-informed benchmark",
           ylabel="Current-map advantage\n(revenue-normalized points)")
    ax.set_xticks(prior.sample_count)
    fig.tight_layout()
    _save(fig, args.output / "prior_stability_48_maps.png")

    final = pd.read_csv(summary / "final_by_configuration.csv")
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for topology, marker in (("regular", "o"), ("multi_hotspot", "s"), ("asymmetric", "^")):
        data = final[final.topology == topology]
        ax.errorbar(
            data.budget_index, data.information_points,
            yerr=[data.information_points - data.ci_low,
                  data.ci_high - data.information_points],
            marker=marker, capsize=2.5, label=topology.replace("_", " ").title()
        )
    ax.set(xlabel="Budget level", ylabel="Profit gain (points)")
    ax.set_xticks(range(5), ["Very low", "Low", "Middle", "High", "Nonbinding"])
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    _save(fig, args.output / "final_information_value.png")

    worked = args.results / "final_decision_diagnostics_v1" / "worked_example"
    environment = json.loads((worked / "environment.json").read_text())
    sites = pd.read_csv(worked / "sites.csv")
    summary_json = json.loads((worked / "summary.json").read_text())
    areas = pd.DataFrame(environment["areas"])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.35), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].scatter(areas.x, areas.y, s=20 + 240 * areas.demand_weight, c="#d1d5db",
                    edgecolor="#6b7280", linewidth=.4, label="Demand area")
    categories = {
        "Catalog only": (sites.catalog_selected.eq(1) & sites.informed_selected.eq(0), "X", COLORS["catalog"]),
        "Both": (sites.catalog_selected.eq(1) & sites.informed_selected.eq(1), "o", "#2563eb"),
        "Informed only": (sites.catalog_selected.eq(0) & sites.informed_selected.eq(1), "P", COLORS["informed"]),
    }
    for label, (mask, marker, color) in categories.items():
        axes[0].scatter(sites.loc[mask, "x"], sites.loc[mask, "y"], marker=marker,
                        s=95, c=color, edgecolor="white", linewidth=.7, label=label)
    axes[0].set(xlabel="Synthetic east-west coordinate", ylabel="Synthetic north-south coordinate")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    components = [summary_json["revenue_difference"], -summary_json["fixed_cost_difference"],
                  -summary_json["variable_cost_difference"]]
    axes[1].bar(["Revenue", "Fixed cost", "Variable cost"], components,
                color=[COLORS["informed"], COLORS["accent"], "#b91c1c"])
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(ylabel="Contribution to informed-minus-catalog profit")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].text(.98, .96, f"Net gain = {summary_json['profit_difference']:.0f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontweight="bold")
    fig.tight_layout()
    _save(fig, args.output / "worked_example_footprint_and_accounting.png")

    print(f"wrote extension figures to {args.output}")


if __name__ == "__main__":
    main()
