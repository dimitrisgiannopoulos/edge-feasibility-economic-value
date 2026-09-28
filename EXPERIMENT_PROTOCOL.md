# Economic-Value Experimental Protocol

Version 2, 2026-09-23. Pilot parameter ranges were fixed before the pilot run. Independent final seeds begin at 90000 and do not overlap development, pilot, hardening, prior, replay, or scaling seeds.

## Methods

| Method | Planning information | Selection method |
|---|---|---|
| Catalog + geography | Catalog, geography, calibrated delay rule | Marginal-profit greedy with deletions and swaps |
| Current feasibility | Current truthful resolved support map | The same greedy local search |
| Prior-informed | Sampled operator networks from the known structural family, not the current map | Two-stage scenario optimization |
| Exact validation | Catalog estimates or current support, matching the relevant treatment | Exact joint site selection |

The primary information contrast is

$$
\Delta_I=\Pi(\mathrm{current\ feasibility})-\Pi(\mathrm{catalog+geography}).
$$

Both terms use the same profit-aware planning routine and the same exact realized-support execution stage. Exact joint site selection is a validation check, not the principal method. The economic study depends on the truthful resolved binary map, not on an API name or serialization. EDFS is one concrete way to supply that map.

Profit differences are also divided by revenue obtainable if all demand qualified. This is called a revenue-normalized profit difference, not a percentage profit increase.

## Environment

- 24 demand areas, three MNOs, three ECSPs, and 48 candidate sites.
- 36 local, nine regional, and three central candidates.
- One moderate application profile with a 50 ms path-delay requirement in the pilot.
- Site-independent service revenue and synthetic monetary units on one common scale.
- Shared compute, site-interface capacity, and contracted MNO-site throughput quotas.
- Physical paths include local breakout and core hairpinning. Suitability is derived from path delay and resolved commercial conditions.
- The catalog delay rule is calibrated on 30 development environments with seeds 10000 through 10029. Pilot seeds start at 20000 and 30000.

## Experiment 0

Nineteen automated tests cover exhaustive subset agreement, operator-specific support, regional consolidation, shared compute, separate interface and quota limits, fixed-plus-variable budget accounting, binary conditional resolution, no-deployment optimality, repeatability, sampled-prior correctness, global operator-site fragmentation policies, post-deployment accounting, EDFS response resolution, scale-preserving demand construction, and valid time-bounded heuristic incumbents.

## Pilot Experiment 1

Twenty independent environments are used for each of three topology families and five budget levels. Capacity is relatively loose to isolate placement economics. This produces 300 paired cases and 1,500 method outcomes.

## Pilot Experiment 2

Matched environments preserve geography, demand, prices, capacities, budget, and the number of supported alternatives per demand group. They vary global MNO-site support as aligned, intermediate, or disjoint, and select supported sites in clustered or dispersed geographic patterns. The main overlap comparison fixes six alternatives. The scarcity comparison uses two, six, and twelve alternatives under intermediate overlap. This produces 200 paired cases and 1,000 method outcomes.

## Reporting Rules

- The generated environment is the independent replication unit.
- Comparisons are paired by seed and environment.
- Report means with paired confidence intervals, medians, and quantiles.
- Preserve negative and null outcomes.
- Report revenue, fixed costs, variable costs, coverage, footprint, stranded expenditure, runtime, and solver gaps.
- Do not call unserved demand an SLA violation.
- Do not describe synthetic monetary units as euros.
- When pooling repeated budget results, resample complete environments rather than treating each budget row as an independent environment.

## Strengthened Evidence Blocks

The `hardening_pilot_v1` block uses independent development seeds beginning at 40000. It crosses fixed-cost multipliers $\{0.5,1,2,4\}$ with variable-cost multipliers $\{0.5,1,2\}$, then crosses compute and throughput supply ratios $\{0.75,1.5,3\}$. A separate correction grid uses detection delays $\{0,0.01,0.10,0.25\}$ of the planning horizon and refundable commitment fractions $\{0,0.5,1\}$.

The initial prior-informed block used nested sets of 3, 6, and 12 sampled operator maps. The strengthened validation supersedes it with 12, 24, and 48 maps. Samples come from the known topology family but never include the current environment's support map.

Decision diagnostics replay all 1,500 final-evaluation budget pairs and all 120 matched overlap environments. They record added, removed, and retained sites, tier and MNO service changes, predicted versus realized service, and a component-wise profit reconciliation. The worked example is the positive middle-budget case nearest the median gain across the 300 final middle-budget environments, not the maximum-gain case.

A stricter overlap control gives every area-MNO group exactly four local, one regional, and one central supported alternative. Fixed cost, base variable cost, compute capacity, interface capacity, and contracted quota are normalized within tier. All 120 controlled cases are evaluated with both the profit-aware heuristic and exact site selection.

The prior-informed validation uses 15 independent environments, five per topology, and nested sample counts of 12, 24, and 48 support maps, with a 600-second exact-solver limit per count. This is a bounded stronger-comparator validation subset, not the principal statistical sample. Application-requirement validation uses 100 independent environments per topology and requirement at 20, 50, and 100 ms. Computational scaling uses one independent benchmark per topology at 24/48, 100/200, and 250/500 areas/sites under the restrictive budget level. Demand scales with the number of areas so demand per area and deployment economics do not collapse at larger sizes. Each information condition receives a 600-second planner limit, and completion status is reported with achieved profit and footprint. Each benchmark runs in a fresh process for attributable peak-memory measurement. Scaling is a practicality boundary and is not used for population-level economic inference.

The requirement audit reconstructs the catalog-predicted and actual support rates for all 900 requirement environments and reports no-deployment frequency by profile. Because the calibrated catalog estimator predicts no supported tuple at 20 ms, the 20 ms catalog comparison is explicitly treated as an empty-baseline boundary case. A matched stronger-comparator subset uses five requirement-sweep seeds per topology and profile, with nested prior sample counts of 12, 24, and 48 and a 600-second solver limit. This subset distinguishes the value of the current support map from the value of any informed operator-support prior.

The final principal evaluation uses 100 independent environments for each of three topology families and evaluates all five budgets, producing 1,500 paired comparisons. Topology families use disjoint seed ranges. Confidence intervals pooled across budgets resample complete environments so repeated budgets from one environment are never treated as independent replications.

The EDFS replay serializes and parses 1,152 area-site responses for one MNO and profile, resolves conditional support, and exactly reconstructs the simulator support map. A low-delay path available only through the public Internet is an explicit negative control and does not qualify for the modeled operator-managed connectivity service.

Fallback service is intentionally excluded. The study asks whether the requested service qualifies and earns its stated revenue. No degraded product is modeled or monetized.
