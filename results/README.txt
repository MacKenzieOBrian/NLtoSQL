results layout (hypothesis-only)

kept for primary hypothesis:
- results/baseline/runs
- results/analysis/per_item_metrics_primary_raw.csv

removed from active tree:
- legacy/archive snapshots
- agent output artifacts
- non-primary analysis exports and figures
- runs where ts_enabled=false and output was not model-only raw
- k=5 run artifacts

primary evidence rule:
- current analysis script source of truth is results/baseline/runs.
- if qlora outputs are produced elsewhere, copy selected run JSONs into results/baseline/runs before analysis.
- do not mix extension/legacy outputs into primary hypothesis statistics.
- TS interpretation rule: TS is computed only when TS is enabled (typically k=3) and the predicted SQL is executable (VA=True); TS for k=0 is intentionally NA in this design.

current coverage:
- llama base: k=0 and k=3, seeds 7/17/27 (6 runs).
- qwen base: k=0 seeds 7/17/27 and k=3 seeds 7/17/27 (6 runs).
- llama qlora: k=0 and k=3, seeds 7/17/27 (6 runs, in results/qlora/runs).
- qwen qlora: k=0 and k=3, seeds 7/17/27 (6 runs, in results/qlora/runs).
- react (llama + qlora adapter, k=3): seeds 7/17/27 (in results/agent/runs).
