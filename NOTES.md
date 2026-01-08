# Notes / TODOs

## Immediate next step (after baselines)

- Run `notebooks/04_build_training_set.ipynb` to validate `data/train/classicmodels_train_200.jsonl` (VA + leakage + SELECT-only).
- Run `notebooks/05_qlora_train_eval.ipynb` to fine-tune adapters and evaluate on the fixed 200-item benchmark.
- Compare QLoRA vs baseline using the saved JSON outputs under `results/`.

## Quality control to document (dissertation)

- Manual spot-check: verify a sample of NLQ→SQL pairs are semantically aligned (VA alone is not enough).
- Record run metadata: commit hash, model id, hyperparams, GPU type/runtime, and any prompt template changes.

## Planned extensions

- Add TS (test-suite accuracy) using distilled ClassicModels variants (schema-identical, different data).
- Add an agentic/refinement mode (tool-using SQL correction loop) and evaluate it with the same harness.
