# Data & Benchmarks

This project’s evaluation is anchored on a **fixed benchmark** of NLQ→SQL pairs over ClassicModels. The dissertation focuses on reproducible comparisons between prompting/agents/fine-tuning, so dataset discipline matters as much as model choice.

For the “why” behind the split and metric choices, see `DECISIONS.md`.

## Current benchmark (ClassicModels-200)

- File: `data/classicmodels_test_200.json`
- Purpose: held-out evaluation for VA/EM/EX now, TS later.
- Status: gold SQL validated to execute successfully on the live ClassicModels DB.

### Record format

Each item is a JSON object:
- `nlq`: natural language question
- `sql`: gold MySQL query (SELECT-only)

Optional fields may be added later (e.g., difficulty tags, tables used, notes).

## Few-shot exemplars (evaluation hygiene)

Few-shot prompting uses NLQ→SQL exemplars **only for inference-time conditioning** (no training).

For dissertation-quality evaluation, exemplars should come from a **separate exemplar pool** (e.g., a train split or a curated exemplar bank) and must not leak the test item.

Current notebook runs support this workflow by passing an exemplar pool (planned); if exemplars are drawn from the benchmark itself, document it explicitly as an experimental condition.

After QLoRA fine-tuning, evaluating with both `k=0` and `k>0` is still meaningful:
- `k=0` isolates the effect of training adapters.
- `k>0` tests whether few-shot prompting provides additional gains on top of fine-tuning.

## Planned datasets

- Train (for QLoRA SFT): NLQ→SQL pairs with broad coverage (joins, aggregations, grouping/having, filters, sorting/limit).
- Distilled DB variants (for TS): schema-identical ClassicModels with different data so semantic equivalence can be tested beyond string match.
- Schema cache (optional): JSON dump of schema introspection for faster prompt building and reproducible schema context.

## Validation & QC

- Validate gold SQL by executing against the live DB (to avoid false negatives).
- Lint/format SQL consistently (so EX normalization is not trivially broken by whitespace).
- Deduplicate near-identical NLQs and ensure coverage across query patterns.
- Log dataset changes (hash/version) in `LOGBOOK.md`.

## Training set workflow (curated, strict)

For QLoRA, the training set must be **separate** from the benchmark test set. This is standard experimental hygiene: if training data overlaps the test benchmark, evaluation is no longer a fair measure of generalisation.

This repo includes a starter training set at `data/train/classicmodels_train_200.jsonl`. Use `notebooks/04_build_training_set.ipynb` to validate that file against:
- *leakage prevention* (train vs test exact-NLQ overlap)
- *executability* (VA)
- basic *safety* (SELECT-only)

If you want to extend/modify the training set, edit the JSONL file directly and re-run `notebooks/04_build_training_set.ipynb` until all rows execute successfully.

The default configuration targets a mixed difficulty distribution (easy/medium/hard). Difficulty is approximated from the SQL structure (joins, grouping/having, subqueries) and is used only to ensure coverage, not as a research metric.

Important limitation to record in the dissertation:
- DB validation (VA=True) guarantees executability, but it does not prove the NLQ and SQL are semantically aligned. You should manually spot-check a sample (e.g., 20–50 items), fix/reject mismatches, and document that QC step.

## Outputs produced by evaluation

Baseline notebooks write JSON outputs under `results/` (gitignored by default):
- `results/baseline/results_zero_shot_200.json`
- `results/baseline/results_few_shot_k3_200.json`

Each output contains per-item fields (`nlq`, `gold_sql`, `raw_sql`, `pred_sql`, `va`, `em`, `ex`, `error`) plus aggregate rates and run metadata (seed/k/timestamp/commit).

Metric note (Ojuri alignment):
- `em`/`em_rate` is exact string match (useful for debugging, but conservative).
- `ex`/`ex_rate` is execution accuracy by result comparison (runs predicted + gold SQL and compares outputs).
