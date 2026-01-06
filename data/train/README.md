# Training Data (QLoRA)

The **benchmark test set** for this project is `data/classicmodels_test_200.json`. That file is reserved for evaluation only and must not be used for fine-tuning.

For QLoRA SFT, create a separate training set (recommended starter size: **200 examples**):
- `data/train/classicmodels_train_200.jsonl`

## Format

JSON Lines (one object per line):

```json
{"nlq":"…","sql":"SELECT …;"}
```

## How to build (strict, LLM-assisted)

Use `notebooks/04_build_training_set.ipynb` to:
- generate candidate NLQ→SQL pairs with an LLM
- enforce SELECT-only output
- validate each SQL by executing it against the live ClassicModels DB (VA must be True)
- reject any NLQs that overlap the benchmark test NLQs
- deduplicate and target a mixed difficulty distribution (easy/medium/hard)

The resulting file is consumed by `notebooks/05_qlora_train_eval.ipynb`.

