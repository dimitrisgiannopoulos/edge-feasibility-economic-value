# Economic Deployment Model

Version 3, used for the strengthened evidence campaign.

## Scope and Information Timing

The ASP makes one deployment decision for a fixed planning horizon. Demand, catalog prices, capacities, application requirements, commercial terms, and operator conditions remain fixed during that horizon. MNO declarations are treated as truthful and authoritative. The model does not add ASP-side verification, stale information, or strategic reporting.

The physical environment contains operator anchors, allowed paths, ECSP interconnections, policy conditions, and path delays. These hidden facts determine whether a demand group can use a site. The informed planner receives the resulting support map. Catalog planners receive locations, tiers, costs, capacities, demand, requirements, and the same spending budget, but not the current operator support map.

The economic comparison depends on a truthful resolved support map, not on an API name or serialization. EDFS provides one concrete pre-deployment response shape. An operator portal, aggregator, or another trusted planning service is equivalent if it produces the same binary map.

## Planning Horizon and Economic Units

The model covers one fixed planning period, interpretable as one month when all tariffs and demand inputs are calibrated monthly. In the synthetic experiments, one demand unit denotes one unit of aggregate qualifying service volume over that period, not one subscriber. The moderate-profile reference revenue is one synthetic monetary unit per served demand unit. Fixed costs, variable costs, budgets, and profit use that same scale.

The reported results are not euro estimates. Multiplying every monetary parameter and the budget by the same positive constant changes only the displayed scale, not the selected deployment. Economic conclusions instead depend on the tested fixed-cost-to-revenue and variable-cost-to-revenue ratios.

A `CONDITIONALLY_VIABLE` response denotes support subject to a known condition, such as a purchased breakout package. Conditions are resolved before optimization:

$$
F_{gs}\in\{0,1\}.
$$

It never denotes 50% of demand or a probability. Here, $g$ is an area-MNO-profile demand group and $s$ is an ECSP site. Frozen simulator outputs use the legacy internal labels `PARTIAL` and `UNVIABLE`; the replay adapter maps them to `CONDITIONALLY_VIABLE` and `NOT_VIABLE` before condition resolution.

## Objective

The ASP chooses site activation $x_s\in\{0,1\}$ and served demand $y_{gs}\ge0$. Revenue $r_g$ depends on delivering the requested service, not on site tier. Variable serving cost is $c_{gs}$ and fixed deployment and committed rental cost is $f_s$.

$$
\max \Pi = \sum_{g,s}(r_g-c_{gs})y_{gs}-\sum_s f_sx_s.
$$

Deploying nothing is feasible and yields zero profit. Unserved demand has no additional penalty in the pilot. Its economic consequence is lost service revenue.

Starting from no deployment guarantees nonnegative predicted profit under the planner's information. It does not guarantee nonnegative realized profit for a catalog planner whose support estimate is wrong.

## Constraints

Demand cannot be served twice:

$$
\sum_s y_{gs}\le d_g.
$$

Only active sites and supported tuples can carry demand:

$$
0\le y_{gs}\le d_gF_{gs}x_s.
$$

Compute and network-interface capacities are shared across all MNOs and demand groups:

$$
\sum_g \rho_g y_{gs}\le K_sx_s,
\qquad
\sum_g h_g y_{gs}\le L_sx_s.
$$

Separately exposed contracted throughput quota $Q_{ms}$ limits traffic for MNO $m$ at site $s$:

$$
\sum_{g:m(g)=m} h_gy_{gs}\le Q_{ms}x_s.
$$

The spending budget charges both fixed and variable expenditure:

$$
\sum_s f_sx_s+\sum_{g,s}c_{gs}y_{gs}\le B.
$$

$K_s$, $L_s$, and $Q_{ms}$ are explicit catalog or contract inputs. They are not inferred from topology-hiding feasibility information.

## Planning and Execution

Each planning method selects a footprint using only its permitted information. The selected sites and fixed costs are then frozen. Every method receives the same exact assignment LP under the realized binary support map and capacities. Consequently, measured differences concern pre-deployment commitments rather than deliberately poor runtime routing.

The exact MILP and fixed-footprint LP use HiGHS through SciPy. A tighter LP re-solves continuous recourse for every MILP footprint before accounting. The maximum observed reconciliation adjustment in the pilot is 0.2195 normalized units on a 10,000-unit maximum-revenue base.

## Profit-Aware Footprint Planner

The primary planner is a deterministic profit-aware heuristic. It is used unchanged under catalog/geography information and current resolved feasibility.

```text
selected = empty footprint
repeat
    for each affordable unselected site:
        solve the exact assignment LP for selected + site
        compute marginal predicted profit / additional fixed cost
    add the site with the largest positive score
until no profitable addition exists

repeat
    evaluate every one-site deletion and affordable one-for-one swap
    accept the move with the largest positive profit improvement
until no improving move exists
```

Every fixed-footprint assignment is solved exactly. The heuristic does not claim globally optimal site selection. Exact joint site-selection solves are retained on small and medium validation subsets to quantify the remaining heuristic gap.

Solver settings use SciPy/HiGHS with one solver thread per process, primal and dual LP feasibility tolerances of $10^{-9}$, and a deterministic tie order inherited from the site catalog. The fixed-footprint cache is bounded to 32 entries so large scaling cases do not retain thousands of one-use assignment matrices.

## EDFS Response Replay

The integration adapter maps `VIABLE` to one, maps `NOT_VIABLE` and `UNKNOWN` to zero, and resolves `CONDITIONALLY_VIABLE` from the known commercial or activation prerequisite. Missing candidates are unsupported when the request asks to include non-viable candidates. The replay uses the work-in-progress `retrieve-viable-edge-cloud-zones` response structure at source revision `bb147f7` and reproduces the simulator support map exactly. Its scope is a synthetic adapter for one MNO and one profile, not a live multi-operator deployment.

The intended service requires operator-managed connectivity to the edge site. A low-delay path over the public Internet therefore does not qualify by delay alone. The adapter includes an explicit negative control for that case.

## Post-Deployment Correction

The correction experiment divides the same planning period into a pre-detection fraction $\delta$ and a post-detection fraction $1-\delta$. A catalog planner commits first. At detection, the actual support map is revealed and the same informed planner may retain, remove, or add sites. Removed initial commitments return fraction $\rho$ of their fixed cost. Revenue and variable costs accrue in proportion to time, while fixed commitments are charged once. All phases share the original total budget.

The null control uses $\delta=0$ and $\rho=1$. It must reproduce the outcome obtained when current feasibility is known before commitment. Any residual difference in that control indicates inconsistent accounting.
