## Logbook (Condensed)

This file records key project changes and why they were made (for dissertation traceability). More detailed notes live in `LOGBOOK_REFLECTIONS.md`.

### 2026-01-08 — Training + validation workflow simplified

- Added a starter QLoRA training set at `data/train/classicmodels_train_200.jsonl` (200 NLQ→SQL pairs) with strict train/test separation from `data/classicmodels_test_200.json`.
- Remade `notebooks/04_build_training_set.ipynb` into a **validator** notebook (leakage check + SELECT-only + executability/VA check on the live ClassicModels DB).
- Updated documentation to reflect the new workflow: curate/edit the JSONL directly, validate it, then run QLoRA (`notebooks/05_qlora_train_eval.ipynb`).

### Baseline checkpoint (n=200)

- Zero-shot: `VA≈0.81`, `EX≈0.00`
- Few-shot (k=3): `VA≈0.855`, `EX≈0.325`
