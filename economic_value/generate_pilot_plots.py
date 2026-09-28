"""Generate pilot figures from saved CSV outputs only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT.parent / "results" / "pilot_v1"
DEFAULT_FIGURES = ROOT.parent / "figures" / "pilot_v1"
COLORS = {"information_greedy": "#0072B2", "information_milp": "#009E73",
          "optimization_with_information": "#D55E00", "instance_information_vs_prior": "#CC79A7"}
LABELS = {"information_greedy": "Information effect, greedy",
          "information_milp": "Information effect, MILP",
          "optimization_with_information": "Optimization effect, informed",
          "instance_information_vs_prior": "Instance map vs prior"}
METHOD_COLORS = {"G-Catalog": "#9E9E9E", "M-Catalog": "#595959", "M-Prior": "#E69F00",
                 "G-EDFS": "#56B4E9", "M-EDFS": "#0072B2"}


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return mean, half


def experiment1(contrasts: list[dict], outcomes: list[dict], output: Path) -> None:
    topologies = ["regular", "multi_hotspot", "asymmetric"]
    contrasts_to_plot = ["information_greedy", "information_milp", "optimization_with_information"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=True)
    for axis, topology in zip(axes, topologies):
        for contrast in contrasts_to_plot:
            means, halves = [], []
            for budget in range(5):
                values = [float(row["revenue_normalized_profit_difference"]) * 100
                          for row in contrasts if row["experiment"] == "1"
                          and row["topology"] == topology and int(row["budget_index"]) == budget
                          and row["contrast"] == contrast]
                mean, half = _mean_ci(values)
                means.append(mean)
                halves.append(half)
            axis.errorbar(range(5), means, yerr=halves, marker="o", linewidth=1.8,
                          capsize=2.5, color=COLORS[contrast], label=LABELS[contrast])
        axis.axhline(0, color="#777777", linewidth=.7)
        axis.set_title(topology.replace("_", " ").title())
        axis.set_xticks(range(5), ["B1", "B2", "B3", "B4", "B5"])
        axis.set_xlabel("Budget level")
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0].set_ylabel("Revenue-normalized profit difference (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / "exp1_information_vs_optimization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    for axis, topology in zip(axes, topologies):
        methods = list(METHOD_COLORS)
        values = [[float(row["realized_profit"]) for row in outcomes
                   if row["experiment"] == "1" and row["topology"] == topology
                   and row["budget_index"] == "2" and row["method"] == method]
                  for method in methods]
        box = axis.boxplot(values, patch_artist=True, showfliers=False, widths=.68)
        for patch, method in zip(box["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(.85)
        axis.set_xticks(range(1, 6), ["G-Cat", "M-Cat", "M-Prior", "G-Info", "M-Info"],
                        rotation=30, ha="right")
        axis.set_title(topology.replace("_", " ").title())
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0].set_ylabel("Realized profit (normalized units)")
    fig.tight_layout()
    fig.savefig(output / "exp1_realized_profit_mid_budget.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def experiment2(contrasts: list[dict], outcomes: list[dict], output: Path) -> None:
    overlap_order = ["aligned", "intermediate", "disjoint"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for axis, pattern in zip(axes, ("clustered", "dispersed")):
        for contrast in ("information_greedy", "information_milp", "optimization_with_information"):
            means, halves = [], []
            for overlap in overlap_order:
                values = [float(row["revenue_normalized_profit_difference"]) * 100
                          for row in contrasts if row["experiment"] == "2" and row["study"] == "overlap"
                          and row["support_pattern"] == pattern and row["overlap"] == overlap
                          and row["contrast"] == contrast]
                mean, half = _mean_ci(values)
                means.append(mean)
                halves.append(half)
            axis.errorbar(range(3), means, yerr=halves, marker="o", capsize=2.5,
                          linewidth=1.8, color=COLORS[contrast], label=LABELS[contrast])
        axis.axhline(0, color="#777777", linewidth=.7)
        axis.set_xticks(range(3), ["Aligned", "Intermediate", "Disjoint"])
        axis.set_title(pattern.title() + " support")
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0].set_ylabel("Revenue-normalized profit difference (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(output / "exp2_information_value_by_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    for axis, pattern in zip(axes, ("clustered", "dispersed")):
        for contrast in ("information_greedy", "information_milp"):
            means, halves = [], []
            for alternatives in (2, 6, 12):
                study = "overlap" if alternatives == 6 else "scarcity"
                values = [float(row["revenue_normalized_profit_difference"]) * 100
                          for row in contrasts if row["experiment"] == "2" and row["study"] == study
                          and row["support_pattern"] == pattern and row["overlap"] == "intermediate"
                          and int(row["alternatives"]) == alternatives and row["contrast"] == contrast]
                mean, half = _mean_ci(values)
                means.append(mean)
                halves.append(half)
            axis.errorbar((2, 6, 12), means, yerr=halves, marker="o", capsize=2.5,
                          linewidth=1.8, color=COLORS[contrast], label=LABELS[contrast])
        axis.set_title(pattern.title() + " support")
        axis.set_xlabel("Supported alternatives per demand group")
        axis.set_xticks((2, 6, 12))
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
    axes[0].set_ylabel("Revenue-normalized profit difference (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(output / "exp2_information_value_by_scarcity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    for axis, metric, ylabel in zip(
        axes, ("multi_operator_sites", "stranded_expenditure"),
        ("Sites serving multiple MNOs", "Stranded expenditure (normalized units)"),
    ):
        width = .15
        for index, method in enumerate(METHOD_COLORS):
            means = []
            for overlap in overlap_order:
                values = [float(row[metric]) for row in outcomes if row["experiment"] == "2"
                          and row["study"] == "overlap" and row["support_pattern"] == "clustered"
                          and row["overlap"] == overlap and row["method"] == method]
                means.append(float(np.mean(values)))
            axis.bar(np.arange(3) + (index - 2) * width, means, width,
                     color=METHOD_COLORS[method], label=method)
        axis.set_xticks(range(3), ["Aligned", "Intermediate", "Disjoint"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(output / "exp2_footprint_mechanisms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contrasts = _rows(args.results / "paired_contrasts.csv")
    outcomes = _rows(args.results / "outcomes.csv")
    experiment1(contrasts, outcomes, args.output)
    experiment2(contrasts, outcomes, args.output)
    print(f"wrote pilot figures to {args.output}")


if __name__ == "__main__":
    main()
