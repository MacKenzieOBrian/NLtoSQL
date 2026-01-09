# Training Data (QLoRA)

The **benchmark test set** for this project is `data/classicmodels_test_200.json`. That file is reserved for evaluation only and must not be used for fine-tuning.

For QLoRA SFT, create a separate training set (recommended starter size: **200 examples**):
- `data/train/classicmodels_train_200.jsonl`

## Format

JSON Lines (one object per line):

```json
{"nlq":"…","sql":"SELECT …;"}
```

## How to validate / maintain (strict)

This repo includes a starter training set at `data/train/classicmodels_train_200.jsonl`.

Use `notebooks/04_build_training_set.ipynb` to validate that file:
- enforce SELECT-only output
- reject any exact NLQs that overlap the benchmark test NLQs
- validate each SQL by executing it against the live ClassicModels DB (VA must be True)
- deduplicate by NLQ

The resulting file is consumed by `notebooks/05_qlora_train_eval.ipynb`.

## Why “mixed difficulty”

Fine-tuning on only complex joins can hurt basic SQL reliability, while training on only trivial queries does not teach the model the multi-table reasoning needed for enterprise-style questions. A mixed set provides broad coverage with a small dataset (e.g., 200 examples), and the difficulty buckets are used only to steer coverage.

For the broader rationale (splits/metrics/repro), see `DECISIONS.md`.
