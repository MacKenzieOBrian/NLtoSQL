# NL→SQL Dissertation Project (ClassicModels)

This repo contains an end-to-end experimentation pipeline for **Natural Language to SQL** over the **ClassicModels** MySQL database.

The dissertation goal is to measure (and explain) performance differences between:
- **Zero-shot prompting** (no exemplars)
- **Few-shot prompting** (in-context exemplars, weights frozen)
- **Agentic SQL generation** (planned, ReAct-style refinement using a safe SQL execution tool)
- **Parameter-efficient fine-tuning** (planned, QLoRA)

## What’s in the repo

- `nl2sql/`: reusable “experiment harness” code (DB access, schema text, prompting, generation, evaluation).
- `notebooks/`: Colab notebooks that *run* experiments and produce dissertation-ready artifacts.
- `data/`: benchmark JSON (currently `data/classicmodels_test_200.json`).
- `results/`: local outputs (JSON runs, figures). Gitignored by default; see `results/README.md`.
- `DECISIONS.md`: decision record (what/why/where) for dissertation writing.
- `ARCHITECTURE.md`: condensed system overview.
- `ARCHITECTURE_DETAILS.md`: full architecture notes (appendix).
- `CONFIG.md`: runtime setup, env vars, and reproducibility checklist.
- `DATA.md`: dataset conventions and evaluation hygiene.
- `LOGBOOK.md`: condensed project log.
- `LOGBOOK_REFLECTIONS.md`: full log + reflections (appendix).
- `NOTES.md`: open questions / TODOs.

## Quickstart (Colab baseline)

1. Set env vars in Colab (or enter when prompted): `INSTANCE_CONNECTION_NAME`, `DB_USER`, `DB_PASS`, `DB_NAME`, `HF_TOKEN`.
2. Open and run: `notebooks/02_baseline_prompting_eval.ipynb`
3. Outputs are written to `results/baseline/` (gitignored by default).

## Quickstart (QLoRA)

1. Generate a strict, DB-validated training set: `notebooks/04_build_training_set.ipynb` → `data/train/classicmodels_train_200.jsonl`
2. Fine-tune + evaluate adapters: `notebooks/05_qlora_train_eval.ipynb`

## Evaluation metrics

- **VA (Validity)**: predicted SQL executes successfully (via `QueryRunner`).
- **EM (Exact Match)**: normalized SQL string match vs gold SQL (strict, conservative).
- **EX (Execution Accuracy)**: execute predicted SQL and compare results to the gold SQL results (Ojuri-style execution accuracy).
- **TS / test-suite accuracy**: planned next metric (compare results across distilled DB variants).
