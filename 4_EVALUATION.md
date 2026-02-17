# Evaluation Framework

## Metric definitions and priority
1. **EX (Execution Accuracy)**: predicted SQL and gold SQL are semantically equivalent by execution output.
2. **TS (Test-Suite Accuracy)**: EX-style correctness across perturbed database suites (robust semantics).
3. **VA (Valid SQL Rate)**: SQL executes without runtime error.
4. **EM (Exact Match)**: normalized string equivalence; useful but not a semantic oracle.

Priority rule for claims: **EX/TS first, VA second, EM diagnostic**.

References:
- `REFERENCES.md#ref-yu2018-spider`
- `REFERENCES.md#ref-zhong2020-ts`

## Statistical definitions used in this project
- **Wilson 95% interval**: robust uncertainty interval for binary rates.
- **Paired delta**: `right_rate - left_rate` on identical examples.
- **Exact McNemar**: significance test using only discordant paired outcomes.

Why these choices:
- Binary metrics (VA/EM/EX/TS) are proportion outcomes.
- Pairing removes example-difficulty confounding.
- McNemar directly tests whether improvements exceed degradations.

References:
- `REFERENCES.md#ref-wilson1927`
- `REFERENCES.md#ref-mcnemar1947`
- `REFERENCES.md#ref-dror2018-significance`

## Comparison policy
Primary controlled comparisons:
- Baseline: `k=0 -> k=3` (few-shot effect)
- QLoRA: `k=0 -> k=3` (few-shot effect post-adaptation)
- Base vs QLoRA at matched `k`
- ReAct infra vs baseline `k=3` on overlap subset
- Model-family contrasts at matched `k`

## Interpretation rules
- Do not claim semantic gain from VA alone.
- Report both effect size (`delta_pct`) and uncertainty/significance.
- Treat non-significant gains as inconclusive.
- Separate method performance claims from infrastructure traceability claims.

## Current evidence snapshot (results/analysis, 2026-02-15)
- Base `k=0 -> k=3`: EX `48.5% -> 61.0%` (`+12.5pp`, McNemar `p=0.0001`).
- QLoRA `k=0 -> k=3`: EX `45.5% -> 55.0%` (`+9.5pp`, `p=0.0003`).
- Base vs QLoRA at `k=3`: EX `61.0% -> 55.0%` (`-6.0pp`, `p=0.0501`, borderline/non-significant at 0.05).
- ReAct infra vs base `k=3` on `n=20`: no significant EX/EM/VA improvement.

## Required dissertation evidence files
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/overall_metrics_wide.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/paired_deltas.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/failure_taxonomy.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/per_item_metrics.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/run_manifest.csv`
