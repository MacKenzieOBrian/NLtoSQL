# Examiner Q&A (Concise)

## Q1. Primary contribution?
A controlled, reproducible open-source comparison of prompting vs QLoRA for NL->SQL under constrained hardware, with paired statistics and failure analysis.

## Q2. How is this linked to Ojuri?
I replicate comparison structure and evaluation discipline, not proprietary stack parity.

Ref: `REFERENCES.md#ref-ojuri2025-agents`

## Q3. Why is ReAct not the main claim?
ReAct is used as execution infrastructure (validation, repair, traceability). Semantic claims still require EX/TS evidence.

## Q4. Which metrics matter most?
EX and TS first, then VA; EM is diagnostic.

## Q5. Statistical defensibility?
95% Wilson intervals, paired deltas, exact McNemar p-values.

## Q6. ReAct loop behavior?
`get_schema -> link_schema -> extract_constraints -> generate_sql -> validate_sql -> validate_constraints -> run_sql`, repair only on failure, return `no_prediction` if repair budget is exhausted.

## Q7. What remains hard?
Join path, aggregation scope, and value-linking errors.

## Q8. Strongest artifacts?
- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/run_manifest.csv`

## Q9. What is not claimed?
No claim of universal SOTA agent, proprietary parity, or cross-domain generalization from a single schema.
