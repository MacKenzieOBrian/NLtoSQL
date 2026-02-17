# Evaluation Framework (Focused Notes)

## Metric priority
1. EX (semantic execution equivalence)
2. TS (semantic robustness across suites)
3. VA (executability)
4. EM (diagnostic only)

## Why this order
- String-level SQL match is insufficient for semantic correctness.
- Execution-based evaluation is the main decision basis.

Refs: `REFERENCES.md#ref-yu2018-spider`, `REFERENCES.md#ref-zhong2020-ts`

## Statistical reporting standard
For each key comparison:
- rate + 95% Wilson CI
- paired delta on shared items
- exact McNemar p-value

Refs: `REFERENCES.md#ref-wilson1927`, `REFERENCES.md#ref-mcnemar1947`, `REFERENCES.md#ref-dror2018-significance`

## Seed-robust reporting
Report both:
- per-seed results
- mean/spread across seeds

Write stable claims only when improvements persist across seeds.

## Required evidence files
- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/per_item_metrics.csv`
- `results/analysis/run_manifest.csv`

## Interpretation rules
- Do not claim semantic gain from VA alone.
- Treat EM as supporting context, not primary evidence.
- Separate performance claims (EX/TS) from infrastructure claims (traceability/robustness behavior).
