# NL->SQL Dissertation Project (ClassicModels)

This repo contains an end-to-end experimentation pipeline for **Natural Language to SQL** over the **ClassicModels** MySQL database.

## Canonical Reading Order (New Decision Log Format)
1. `1_LITERATURE.md`
2. `2_METHODOLOGY.md`
3. `3_AGENT_DESIGN.md`
4. `4_EVALUATION.md`
5. `5_ITERATIVE_REFINEMENTS.md`
6. `6_LIMITATIONS.md`
7. `LOGBOOK.md`

## What's in the repo

- `nl2sql/`: reusable experiment harness code (DB access, schema text, prompting, generation, evaluation).
- `notebooks/`: Colab notebooks that run experiments and produce outputs.
- `data/`: benchmark JSON.
- `results/`: outputs (JSON runs, figures). In this repo, results are kept in git for reproducibility; see `results/README.md`.
- `REFERENCES.md`: bibliography list.

## Quickstart (Colab baseline)

1. Set env vars in Colab when prompted: `INSTANCE_CONNECTION_NAME`, `DB_USER`, `DB_PASS`, `DB_NAME`, `HF_TOKEN`.
2. Open and run: `notebooks/02_baseline_prompting_eval.ipynb`
3. Outputs are written to `results/baseline/`.

## Quickstart (QLoRA)

1. Validate (and optionally edit) the provided training set: `notebooks/04_build_training_set.ipynb`
2. Fine-tune + evaluate adapters: `notebooks/05_qlora_train_eval.ipynb`

## Evaluation metrics

- **VA (Validity)**: predicted SQL executes successfully (via `QueryRunner`).
- **EM (Exact Match)**: normalized SQL string match vs gold SQL (strict, diagnostic).
- **EX (Execution Accuracy)**: execute predicted SQL and compare results to the gold SQL results.
- **TS / test-suite accuracy**: suite-based semantic check implemented in `nl2sql/eval.py`.

Code: `nl2sql/query_runner.py`, `nl2sql/eval.py`
