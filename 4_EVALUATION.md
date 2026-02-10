# Evaluation Framework

This project evaluates semantics first, syntax second.

## Metric Priority

1. EX (execution equivalence on base DB)
2. TS (suite-based semantic robustness)
3. VA (executability)
4. EM (diagnostic string agreement)

## Literature Basis for Metric Hierarchy

- Spider established complex cross-domain evaluation pressure where SQL string form alone is insufficient: `REFERENCES.md#ref-yu2018-spider`.
- Distilled test-suite evaluation motivates behavior-focused semantic checks: `REFERENCES.md#ref-zhong2020-ts`.
- Execution-guided and agentic NL->SQL work supports explicit runtime feedback loops for reliability: `REFERENCES.md#ref-wang2018-eg-decoding`, `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-zhai2025-excot`.

Interpretation consequence:
- EM is retained for diagnostics,
- EX/TS drive semantic claims,
- VA is a validity floor, not a semantic endpoint.

## Metric Definitions

### VA
Whether predicted SQL executes under the `QueryRunner` safety policy.

Code:
- `nl2sql/query_runner.py`

### EX
Whether predicted and gold SQL return equivalent result multisets on the base DB.

Code:
- `nl2sql/eval.py:execution_accuracy`

### TS
Whether predicted and gold SQL agree across perturbed DB replicas.

Code:
- `nl2sql/eval.py:test_suite_accuracy_for_item`

### EM
Normalized string equality between predicted and gold SQL.

Code:
- `nl2sql/postprocess.py:normalize_sql`

## Statistical Reporting (Defensible Differences)

For each metric and run:
- report rate and 95% Wilson interval.

For paired run comparisons on identical examples:
- report delta in percentage points,
- report improved/degraded/tied counts,
- report exact McNemar p-value.

Code:
- `nl2sql/research_stats.py`
- `scripts/generate_research_comparison.py`

## Required Result Artifacts

- `results/analysis/overall_metrics_long.csv`
- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/per_item_metrics.csv`

These files are intended to drive all dissertation tables and plots.

## Claim-Evidence Rulebook

Use this checklist when writing results sections.

1. Any claim of "improvement" must include effect size and uncertainty.
2. Any cross-method claim should use paired evidence on shared examples.
3. Any mechanism claim should include error-type evidence (not only aggregate metrics).
4. Any ReAct claim must state whether it is an infrastructure claim (validity/traceability) or a semantic claim.

## Interpretation Rules

- Do not claim semantic improvement from VA alone.
- Treat EM as supporting evidence, not primary evidence.
- Prefer paired deltas over unpaired percentage comparison.
- Always include uncertainty and sample size in claims.

## Error Taxonomy Requirement

Every EX/VA failure is bucketed into one dominant class:
- invalid SQL
- join path
- aggregation
- value linking
- projection
- ordering/limit
- other semantic

Use taxonomy shifts to explain why metrics changed.

## Reproducibility Checklist

- fixed dataset and split
- fixed run config with logged seed
- archived JSON output per run
- shared evaluator across methods
- versioned script/notebook path in run metadata

## Minimal Claim Template

Under the same 200-item test set and evaluator, Method B improved EX by +X pp over Method A (paired McNemar p=Y), while remaining failures were dominated by [error classes].
