# Economic Value of Design-Time Edge Feasibility Exposure

Version `v0.1.1` of the reproducibility artifact for *The Economic Value of
Design-Time Feasibility Exposure for Multi-Operator Edge Deployment*.

The artifact contains the synthetic path and policy generator, catalog and
prior-informed comparators, common profit model, correction model, frozen
configurations, raw experiment outcomes, summary tables, and plotting code.
All site locations, service demand, network paths, and monetary values are
synthetic. No operator data or market-calibrated prices are included.

## Setup

Use Python 3.8 or later in a clean environment:

```bash
python -m pip install -r requirements.txt
```

## Verify the released results

The following commands use the supplied raw outcomes and do not rerun the
computationally expensive planning campaign:

```bash
python -m unittest economic_value.test_correctness economic_value.test_hardening
python -m economic_value.diagnose_informed_losses
python -m economic_value.analyze_extensions
python -m economic_value.validate_extensions
python -m economic_value.generate_extension_plots
```

`results/evidence_summary_v1/` contains the figure inputs and the exact
ten-case diagnosis of informed-greedy losses. Plots are written to
`figures/evidence_v1/`. The numerical runners never make plots.

## Regenerate the experiments

The frozen configurations are in `economic_value/config/`. The principal
independent evaluation is:

```bash
python -m economic_value.run_final_evaluation \
  --config economic_value/config/final_v2.json --output results/final_v2
```

The matched requirement comparator is:

```bash
python -m economic_value.run_prior_stability \
  --config economic_value/config/requirement_prior_v1.json \
  --output results/requirement_prior_v1
```

Additional runner commands and seed ranges are documented in
`EXPERIMENT_PROTOCOL.md`. Supplied raw CSV files let readers inspect and
replot the results without waiting for every solver run. Checkpoints were
omitted from the release because they duplicate those CSV outcomes; rerunning
the numerical modules creates fresh checkpoints.

## Scope

The 20 ms catalog estimator predicts no supported site, which is diagnosed
explicitly in `requirement_catalog_diagnostics.csv`. The 20 ms result against
that catalog baseline is an empty-baseline boundary; the prior-informed
comparison provides the stronger strict-requirement check. The largest
250-area/500-site scaling runs reach the time limit and return incumbents.

The study evaluates a truthful resolved support map. The EDFS-style response
replay is a synthetic integration check for one operator and profile.
