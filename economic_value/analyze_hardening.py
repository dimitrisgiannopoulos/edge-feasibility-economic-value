"""Create reproducible numerical summaries from saved pilot outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PILOT = ROOT / "results" / "pilot_v1"
DEFAULT_HARDENING = ROOT / "results" / "hardening_v1"
DEFAULT_PRIOR = ROOT / "results" / "prior_stability_v1"


def _cluster_summary(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "environment_id",
    seed: int = 2027,
    samples: int = 20_000,
) -> dict[str, float]:
    clustered = frame.groupby(cluster, sort=True)[value].mean().to_numpy(dtype=float)
    if not len(clustered):
        raise ValueError("Cannot summarize an empty frame")
    rng = np.random.default_rng(seed)
    draws = rng.choice(clustered, size=(samples, len(clustered)), replace=True).mean(axis=1)
    return {
        "mean": float(clustered.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "median": float(np.median(clustered)),
        "q25": float(np.quantile(clustered, 0.25)),
        "q75": float(np.quantile(clustered, 0.75)),
        "clusters": int(len(clustered)),
    }


def _pilot_summary(path: Path) -> dict:
    outcomes = pd.read_csv(path / "experiment1_outcomes.csv")
    pivot = outcomes.pivot(index="case_id", columns="method", values="revenue_normalized_profit")
    metadata = outcomes.drop_duplicates("case_id").set_index("case_id")[["environment_id"]]
    greedy = (100 * (pivot["G-EDFS"] - pivot["G-Catalog"])).rename("difference").to_frame().join(metadata)
    prior = (100 * (pivot["G-EDFS"] - pivot["M-Prior"])).rename("difference").to_frame().join(metadata)
    middle = outcomes[
        (outcomes["budget_index"] == 2)
        & outcomes["method"].isin(["G-Catalog", "G-EDFS"])
    ]
    middle_table = (
        middle.groupby("method")[[
            "profit", "total_expenditure", "qualifying_coverage", "sites",
            "local_sites", "stranded_expenditure",
        ]]
        .mean()
        .to_dict(orient="index")
    )
    return {
        "greedy_current_vs_catalog_points": {
            **_cluster_summary(greedy, "difference"),
            "positive_case_share": float((greedy["difference"] > 0).mean()),
            "paired_cases": int(len(greedy)),
        },
        "current_greedy_vs_prior_points": {
            **_cluster_summary(prior, "difference", seed=2028),
            "positive_case_share": float((prior["difference"] > 0).mean()),
            "paired_cases": int(len(prior)),
        },
        "middle_budget": middle_table,
    }


def _cell_summaries(frame: pd.DataFrame, keys: list[str], value: str) -> list[dict]:
    outputs = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = {name: item for name, item in zip(keys, key)}
        row.update(_cluster_summary(group, value, seed=2030 + len(outputs)))
        outputs.append(row)
    return outputs


def _hardening_summary(path: Path) -> dict:
    contrasts = pd.read_csv(path / "paired_contrasts.csv")
    contrasts["difference_points"] = 100 * contrasts["revenue_normalized_profit_difference"]
    experiment3 = contrasts[contrasts["experiment"] == 3]
    experiment4 = contrasts[contrasts["experiment"] == 4]
    correction = pd.read_csv(path / "experiment5_outcomes.csv")
    correction["difference_points"] = 100 * correction["revenue_normalized_profit_difference"]
    return {
        "experiment3_cost_sensitivity": _cell_summaries(
            experiment3,
            ["fixed_cost_multiplier", "variable_cost_multiplier"],
            "difference_points",
        ),
        "experiment4_resource_sensitivity": _cell_summaries(
            experiment4,
            ["compute_ratio", "throughput_ratio"],
            "difference_points",
        ),
        "experiment5_correction": _cell_summaries(
            correction,
            ["detection_fraction", "refund_fraction"],
            "difference_points",
        ),
    }


def _prior_summary(path: Path) -> list[dict]:
    frame = pd.read_csv(path / "outcomes.csv")
    frame["difference_points"] = 100 * frame["revenue_normalized_difference_from_current"]
    return _cell_summaries(frame, ["sample_count"], "difference_points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--hardening", type=Path, default=DEFAULT_HARDENING)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args()
    result = {
        "statistical_unit": "generated physical environment",
        "confidence_interval": "cluster bootstrap over complete environments, 20,000 draws",
        "pilot": _pilot_summary(args.pilot),
        "hardening": _hardening_summary(args.hardening),
        "prior_stability": _prior_summary(args.prior),
    }
    target = args.hardening / "summary.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
