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

current coverage note:
- baseline llama kept: k=0 and k=3, seeds 7/17/27.
- baseline qwen kept: k=0 seeds 7/17 and k=3 seeds 7/17/27.
- qlora qwen pending new runs.
- qlora llama pending new runs.
