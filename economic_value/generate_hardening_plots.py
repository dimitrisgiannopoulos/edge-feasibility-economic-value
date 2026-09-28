"""Generate hardening figures from saved CSV results only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "hardening_v1"
DEFAULT_FIGURES = ROOT / "figures" / "hardening_v1"
DEFAULT_PRIOR = ROOT / "results" / "prior_stability_v1"

NAVY = "#103b4d"
TEAL = "#08756c"
ORANGE = "#d87936"
VIOLET = "#765a9b"
GRID = "#d7dde2"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#72808b",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _heatmap(
    axis: plt.Axes,
    table: pd.DataFrame,
    title: str,
    color_map: str,
    color_label: str,
    decimals: int = 1,
) -> None:
    values = table.to_numpy(dtype=float)
    image = axis.imshow(values, cmap=color_map, aspect="auto")
    axis.set_title(title, color=NAVY)
    axis.set_xticks(range(len(table.columns)), [f"{value:g}x" for value in table.columns])
    axis.set_yticks(range(len(table.index)), [f"{value:g}x" for value in table.index])
    threshold = (np.nanmin(values) + np.nanmax(values)) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                color="white" if value > threshold else NAVY,
                fontweight="bold",
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(color_label)


def cost_sensitivity(outcomes: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    informed = outcomes[
        (outcomes["experiment"] == 3) & (outcomes["method"] == "Current feasibility")
    ]
    profit = informed.pivot_table(
        index="fixed_cost_multiplier",
        columns="variable_cost_multiplier",
        values="realized_profit",
        aggfunc="mean",
    )
    exp = contrasts[contrasts["experiment"] == 3].copy()
    exp["points"] = 100 * exp["revenue_normalized_profit_difference"]
    gain = exp.pivot_table(
        index="fixed_cost_multiplier",
        columns="variable_cost_multiplier",
        values="points",
        aggfunc="mean",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 4.35))
    _heatmap(
        axes[0], profit, "Profit with current feasibility", "YlGnBu",
        "Synthetic monetary units", decimals=0,
    )
    _heatmap(
        axes[1], gain, "Information value", "YlOrBr",
        "Revenue-normalized points", decimals=1,
    )
    axes[0].set_ylabel("Fixed-cost multiplier")
    axes[1].set_ylabel("Fixed-cost multiplier")
    figure.supxlabel("Variable-cost multiplier", y=0.01)
    figure.tight_layout(rect=(0, 0.04, 1, 1), pad=1.0)
    figure.savefig(output / "exp3_cost_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def resource_sensitivity(outcomes: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    informed = outcomes[
        (outcomes["experiment"] == 4) & (outcomes["method"] == "Current feasibility")
    ]
    profit = informed.pivot_table(
        index="compute_ratio",
        columns="throughput_ratio",
        values="realized_profit",
        aggfunc="mean",
    )
    exp = contrasts[contrasts["experiment"] == 4].copy()
    exp["points"] = 100 * exp["revenue_normalized_profit_difference"]
    gain = exp.pivot_table(
        index="compute_ratio",
        columns="throughput_ratio",
        values="points",
        aggfunc="mean",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 4.35))
    _heatmap(
        axes[0], profit, "Profit with current feasibility", "YlGnBu",
        "Synthetic monetary units", decimals=0,
    )
    _heatmap(
        axes[1], gain, "Information value", "YlOrBr",
        "Revenue-normalized points", decimals=1,
    )
    axes[0].set_ylabel("Compute supply / demand")
    axes[1].set_ylabel("Compute supply / demand")
    figure.supxlabel("Throughput supply / demand", y=0.01)
    figure.tight_layout(rect=(0, 0.04, 1, 1), pad=1.0)
    figure.savefig(output / "exp4_resource_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def correction(correction_rows: pd.DataFrame, output: Path) -> None:
    frame = correction_rows.copy()
    frame["points"] = 100 * frame["revenue_normalized_profit_difference"]
    figure, axis = plt.subplots(figsize=(9.8, 4.5))
    colors = {0.0: VIOLET, 0.5: ORANGE, 1.0: TEAL}
    labels = {0.0: "No refund", 0.5: "50% refund", 1.0: "Full refund"}
    for refund in sorted(frame["refund_fraction"].unique()):
        means, errors, delays = [], [], []
        for delay, group in frame[frame["refund_fraction"] == refund].groupby(
            "detection_fraction", sort=True
        ):
            values = group["points"].to_numpy(dtype=float)
            means.append(float(values.mean()))
            errors.append(
                float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
                if len(values) > 1 else 0.0
            )
            delays.append(100 * float(delay))
        axis.errorbar(
            delays,
            means,
            yerr=errors,
            marker="o",
            linewidth=2.3,
            capsize=4,
            color=colors[float(refund)],
            label=labels[float(refund)],
        )
    axis.axhline(0, color="#87939c", linewidth=1)
    axis.set_xlabel("Detection delay (% of planning horizon)")
    axis.set_ylabel("Profit retained by knowing before commitment\n(per 100 potential-revenue units)")
    axis.set_xticks([0, 1, 10, 25])
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, title="Refund on initial commitments")
    figure.tight_layout(pad=1.0)
    figure.savefig(output / "exp5_correction_value.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def prior_stability(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.copy()
    frame["points"] = 100 * frame["revenue_normalized_difference_from_current"]
    counts, means, errors, runtimes = [], [], [], []
    for count, group in frame.groupby("sample_count", sort=True):
        values = group["points"].to_numpy(dtype=float)
        counts.append(int(count))
        means.append(float(values.mean()))
        errors.append(
            float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
            if len(values) > 1 else 0.0
        )
        runtimes.append(float(group["selection_seconds"].median()))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.25))
    axes[0].errorbar(
        counts, means, yerr=errors, marker="o", linewidth=2.3, capsize=4, color=TEAL,
    )
    axes[0].set_title("Current map advantage over sampled priors", color=NAVY)
    axes[0].set_ylabel("Profit units per 100 potential-revenue units")
    axes[1].plot(counts, runtimes, marker="s", linewidth=2.3, color=VIOLET)
    axes[1].set_title("Median benchmark solve time", color=NAVY)
    axes[1].set_ylabel("Seconds")
    for axis in axes:
        axis.set_xlabel("Sampled operator maps")
        axis.set_xticks(counts)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=1.0)
    figure.savefig(output / "prior_sample_stability.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()
    outcomes = pd.read_csv(args.results / "outcomes.csv")
    contrasts = pd.read_csv(args.results / "paired_contrasts.csv")
    correction_rows = pd.read_csv(args.results / "experiment5_outcomes.csv")
    prior_rows = pd.read_csv(args.prior / "outcomes.csv")
    cost_sensitivity(outcomes, contrasts, args.output)
    resource_sensitivity(outcomes, contrasts, args.output)
    correction(correction_rows, args.output)
    prior_stability(prior_rows, args.output)
    print(f"wrote hardening figures to {args.output}")


if __name__ == "__main__":
    main()
