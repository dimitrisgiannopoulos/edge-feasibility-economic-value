# Strengthened Evidence Summary

## Independent Principal Evaluation

- 300 independent environments and 1,500 paired budget cases.
- Mean information value: **12.20 revenue-normalized points**.
- Environment-clustered bootstrap 95% interval: **[11.45, 12.96]**.
- Current feasibility produces higher realized profit in **94.3%** of paired cases, ties in **5.0%**, and loses in **0.7%**.

## Decision Mechanism

- Footprint-change rates, additions, removals, and exact revenue/cost decompositions are in `decision_change_summary.csv`.
- The worked example was selected by the disclosed median-gain rule, not by maximum effect.
- Maximum profit-accounting residual across decision and correction diagnostics: **2.046e-12**.

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
