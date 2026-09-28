"""Validate strengthened evidence files and accounting invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    checks = {}

    pilot_diagnostics = pd.read_csv(args.results / "diagnostics_v1" / "decision_pairs.csv")
    diagnostics = pd.read_csv(
        args.results / "final_decision_diagnostics_v1" / "decision_pairs.csv"
    )
    correction = pd.read_csv(args.results / "diagnostics_v1" / "correction_accounting.csv")
    checks["decision_pairs"] = len(diagnostics)
    checks["overlap_pairs"] = int((pilot_diagnostics.kind == "overlap").sum())
    checks["correction_conditions"] = len(correction)
    checks["max_decision_accounting_residual"] = float(diagnostics.accounting_residual.abs().max())
    checks["max_correction_accounting_residual"] = float(correction.accounting_residual.abs().max())
    assert checks["decision_pairs"] == 1500
    assert checks["overlap_pairs"] == 120
    assert checks["correction_conditions"] == 720
    assert checks["max_decision_accounting_residual"] < 1e-7
    assert checks["max_correction_accounting_residual"] < 1e-7
    null = correction[(correction.detection_fraction == 0) & (correction.refund_fraction == 1)]
    checks["correction_null_max"] = float(null.profit_advantage_from_early_information.abs().max())
    assert checks["correction_null_max"] < 1e-7

    replay = json.loads((args.results / "edfs_replay_v1" / "summary.json").read_text())
    checks["edfs_replay"] = replay["status"]
    assert replay["status"] == "PASS" and replay["fast_public_internet_negative_control"] == "PASS"

    controlled = pd.read_csv(args.results / "overlap_control_v1" / "pairs.csv")
    checks["controlled_overlap_cases"] = len(controlled)
    checks["controlled_overlap_max_greedy_gap_points"] = float(
        controlled.informed_greedy_gap_points.max()
    )
    checks["controlled_overlap_catalog_optimal_fraction"] = float(
        (controlled.catalog_exact_status == "optimal").mean()
    )
    checks["controlled_overlap_max_catalog_gap"] = float(
        controlled.catalog_exact_gap.max()
    )
    assert checks["controlled_overlap_cases"] == 120
    assert (controlled.informed_exact_status == "optimal").all()
    assert checks["controlled_overlap_max_catalog_gap"] <= 0.021
    assert np.allclose(controlled.supported_local_share, 4 / 6)
    assert np.allclose(controlled.supported_regional_share, 1 / 6)
    assert np.allclose(controlled.supported_central_share, 1 / 6)

    overlap_contrasts = pd.read_csv(
        args.results / "evidence_summary_v1" / "controlled_overlap_regime_contrasts.csv"
    )
    checks["controlled_overlap_contrasts"] = len(overlap_contrasts)
    assert checks["controlled_overlap_contrasts"] == 6
    assert (overlap_contrasts.paired_environments == 20).all()

    hardening = pd.read_csv(args.results / "hardening_diagnostics_v1" / "outcomes.csv")
    checks["hardening_rows"] = len(hardening)
    assert checks["hardening_rows"] == 3240
    for field in ("selected_compute_capacity", "selected_interface_capacity",
                  "compute_utilization", "interface_utilization", "no_deployment",
                  "individually_affordable_sites"):
        assert field in hardening.columns

    prior = pd.read_csv(args.results / "prior_stability_v3" / "outcomes.csv")
    checks["prior_rows"] = len(prior)
    checks["prior_sample_counts"] = sorted(prior.sample_count.unique().tolist())
    checks["prior_max_gap"] = float(prior.mip_gap.max())
    checks["prior_optimal_fraction"] = float((prior.status == "optimal").mean())
    assert checks["prior_rows"] == 45
    assert checks["prior_sample_counts"] == [12, 24, 48]
    assert np.isfinite(prior.mip_gap).all()
    assert checks["prior_max_gap"] <= 0.05

    prior_summary = pd.read_csv(args.results / "evidence_summary_v1" / "prior_stability_summary.csv")
    assert {"ci_low", "ci_high"}.issubset(prior_summary.columns)
    assert (prior_summary.ci_low <= prior_summary.current_advantage_points).all()
    assert (prior_summary.current_advantage_points <= prior_summary.ci_high).all()

    requirements = pd.read_csv(args.results / "requirements_v1" / "requirements_outcomes.csv")
    scaling = pd.read_csv(args.results / "scaling_clean_v5" / "scaling_outcomes.csv")
    checks["requirement_rows"] = len(requirements)
    checks["scaling_rows"] = len(scaling)
    assert checks["requirement_rows"] == 1800
    assert checks["scaling_rows"] == 18
    assert set(scaling.planning_status).issubset({"local_optimum", "time_limit"})
    assert np.isfinite(requirements.select_dtypes(include=[np.number]).to_numpy()).all()
    assert np.isfinite(scaling.select_dtypes(include=[np.number]).to_numpy()).all()

    requirement_diagnostics = pd.read_csv(
        args.results / "evidence_summary_v1" / "requirement_catalog_diagnostics.csv"
    ).set_index("profile")
    checks["strict_catalog_predicted_support_rate"] = float(
        requirement_diagnostics.loc["strict", "predicted_supported_tuple_rate"]
    )
    checks["strict_catalog_no_deployment_rate"] = float(
        requirement_diagnostics.loc["strict", "catalog_no_deployment_rate"]
    )
    assert checks["strict_catalog_predicted_support_rate"] == 0.0
    assert checks["strict_catalog_no_deployment_rate"] == 1.0

    requirement_prior_path = args.results / "requirement_prior_v1" / "outcomes.csv"
    if requirement_prior_path.exists():
        requirement_prior = pd.read_csv(requirement_prior_path)
        checks["requirement_prior_rows"] = len(requirement_prior)
        checks["requirement_prior_profiles"] = sorted(requirement_prior.profile.unique().tolist())
        checks["requirement_prior_sample_counts"] = sorted(
            requirement_prior.sample_count.unique().tolist()
        )
        checks["requirement_prior_max_gap"] = float(requirement_prior.mip_gap.max())
        strict_48 = requirement_prior[
            (requirement_prior.profile == "strict")
            & (requirement_prior.sample_count == 48)
        ]
        checks["requirement_prior_strict_48_optimal_fraction"] = float(
            (strict_48.status == "optimal").mean()
        )
        assert checks["requirement_prior_rows"] == 135
        assert checks["requirement_prior_profiles"] == ["moderate", "relaxed", "strict"]
        assert checks["requirement_prior_sample_counts"] == [12, 24, 48]
        assert np.isfinite(requirement_prior.mip_gap).all()
        assert checks["requirement_prior_max_gap"] <= 0.06
        assert checks["requirement_prior_strict_48_optimal_fraction"] == 1.0

    final_rows = pd.read_csv(args.results / "final_v2" / "outcomes.csv")
    final_pairs = pd.read_csv(args.results / "final_v2" / "paired_contrasts.csv")
    checks["final_outcomes"] = len(final_rows)
    checks["final_pairs"] = len(final_pairs)
    checks["final_environments"] = final_pairs.environment_id.nunique()
    residual = (final_pairs.profit_difference - (
        final_pairs.revenue_difference - final_pairs.fixed_cost_difference
        - final_pairs.variable_cost_difference
    )).abs()
    checks["final_max_accounting_residual"] = float(residual.max())
    assert checks["final_outcomes"] == 3000
    assert checks["final_pairs"] == 1500
    assert checks["final_environments"] == 300
    assert checks["final_max_accounting_residual"] < 1e-7

    win_tie_loss = pd.read_csv(args.results / "evidence_summary_v1" / "principal_win_tie_loss.csv")
    assert int(win_tie_loss.loc[0, ["wins", "ties", "losses"]].sum()) == 1500

    output = {"status": "PASS", **checks}
    target = args.results / "evidence_summary_v1" / "validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
