# Data & Benchmarks

This project’s evaluation is anchored on a **fixed benchmark** of NLQ→SQL pairs over ClassicModels. The dissertation focuses on reproducible comparisons between prompting/agents/fine-tuning, so dataset discipline matters as much as model choice.

## Current benchmark (ClassicModels-200)

- File: `data/classicmodels_test_200.json`
- Purpose: held-out evaluation for VA/EX now, TS/result-equivalence later.
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

## Planned datasets

- Train (for QLoRA SFT): NLQ→SQL pairs with broad coverage (joins, aggregations, grouping/having, filters, sorting/limit).
- Distilled DB variants (for TS): schema-identical ClassicModels with different data so semantic equivalence can be tested beyond string match.
- Schema cache (optional): JSON dump of schema introspection for faster prompt building and reproducible schema context.

## Validation & QC

- Validate gold SQL by executing against the live DB (to avoid false negatives).
- Lint/format SQL consistently (so EX normalization is not trivially broken by whitespace).
- Deduplicate near-identical NLQs and ensure coverage across query patterns.
- Log dataset changes (hash/version) in `LOGBOOK.md`.

## Outputs produced by evaluation

Baseline notebooks write JSON artifacts under `results/` (gitignored by default):
- `results/baseline/results_zero_shot_200.json`
- `results/baseline/results_few_shot_k3_200.json`

Each output contains per-item fields (`nlq`, `gold_sql`, `raw_sql`, `pred_sql`, `va`, `ex`, `error`) plus aggregate rates and run metadata (seed/k/timestamp/commit).
