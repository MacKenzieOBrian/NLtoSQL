# Examiner Q&A

This sheet is aligned with the current repository implementation.

## Q1. What is your primary contribution?

A controlled, reproducible comparison of NL->SQL improvements under constrained compute:
- prompting effect (`k=0` vs `k=3`)
- QLoRA fine-tuning effect (base vs adapted model)
- error-category analysis explaining metric movement

ReAct is included as execution infrastructure to stabilize validity and expose failure causes.

## Q1b. How do you replicate Ojuri while emphasizing open-source work?

I replicate Ojuri at the level of *comparison design* and *evaluation discipline* (prompting vs fine-tuning vs agent support, EX/TS-first interpretation, paired statistics), then run those contrasts on a fully open-source local stack with versioned artifacts and rerunnable scripts.

I do not claim proprietary-model parity; I claim directional trend replication under constrained open-source conditions.

Reference note:
- `11_REPLICATION_POSITIONING.md`

## Q2. Why is ReAct not your main claim?

Because the dissertation question is "what improves NL->SQL under constraints," not "how complex can the agent become." ReAct is used to enforce tool order, validation, and repair for robust evaluation.

## Q3. What metrics matter most and why?

- EX and TS are primary semantic metrics.
- VA measures executability.
- EM is diagnostic only.

Code pointers:
- `nl2sql/eval.py`
- `4_EVALUATION.md`

## Q4. How do you defend differences statistically?

- 95% Wilson intervals for per-run rates.
- Paired deltas on identical examples.
- Exact McNemar p-values for binary paired outcomes.

Code pointers:
- `nl2sql/research_stats.py`
- `scripts/generate_research_comparison.py`

## Q5. How do you ensure fair comparisons?

- same test set
- same evaluator
- same SQL safety policy
- same output artifact format
- explicit run metadata

## Q6. What does the ReAct core loop do?

`get_schema -> link_schema -> extract_constraints -> generate_sql -> validate_sql -> validate_constraints -> run_sql`, with `repair_sql` only on validation/execution failure.

If execution succeeds, the loop stops and returns that SQL.
If repair budget is exhausted without a successful execution, the loop returns `no_prediction` (it does not return known-failed SQL).

Code pointer:
- `nl2sql/react_pipeline.py`

## Q7. What remains hard even after improvements?

Semantic alignment errors: join-path mistakes, aggregation scope errors, and value-linking misses. Execution guidance improves validity but does not eliminate these categories.

## Q8. What are your strongest evidence artifacts?

- run JSONs in `results/`
- paired/comparison tables in `results/analysis/`
- failure taxonomy in `results/analysis/failure_taxonomy.csv`
- dated rationale in `LOGBOOK.md`

## Q9. What do you explicitly not claim?

- not a universal state-of-the-art Text-to-SQL agent
- not proven cross-domain generalization
- not full replacement for learned schema/linking models
