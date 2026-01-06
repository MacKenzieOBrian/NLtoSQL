# Architecture (Summary)

This is the dissertation-facing summary. Detailed notes are preserved in `ARCHITECTURE_DETAILS.md`.

## Goal

Build a reproducible NL→SQL evaluation pipeline over ClassicModels to compare:
- zero-shot prompting
- few-shot prompting
- QLoRA fine-tuning (adapters)
- agentic refinement (planned)

## Repo structure (why)

- `nl2sql/`: stable, importable evaluation harness (DB, schema, safety, prompting, metrics).
- `notebooks/`: Colab runners (experiments and artifacts).
- `data/`: benchmark test set + training set.
- `results/`: run outputs (gitignored by default; curate dissertation artifacts intentionally).

Rationale: keep evaluation logic stable and reviewable while notebooks remain thin runners. See `DECISIONS.md`.

## Metrics (Ojuri-aligned)

- `VA`: query executes successfully.
- `EM`: strict normalized string match (debugging baseline).
- `EX`: execution accuracy (compare predicted vs gold result sets).
- `TS`: test-suite accuracy across distilled DB variants (planned).

## Validity risks (and mitigations)

- Leakage (train/test overlap or exemplars leak test items) → leakage checks + explicit exemplar policy logging.
- LLM-assisted training noise → DB validation + manual spot-check sample and document QC.

