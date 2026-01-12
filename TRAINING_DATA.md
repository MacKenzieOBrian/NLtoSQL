# Training Data (ClassicModels)

Purpose: supply supervised pairs for QLoRA while preserving a clean 200-item test set for VA/EX comparison.

## Sources and Validation
- Manually curated + templated NL→SQL over ClassicModels.
- Stored in `data/train/classicmodels_train_200.jsonl`; test in `data/classicmodels_test_200.json`.
- Validation: execution against the live DB (VA check) and exact NLQ de-dup vs the test set to prevent leakage [18].

## Coverage Targets
- Multi-table joins, GROUP BY/HAVING, filters, temporal conditions, ordering/limits.
- Mirrors business-style queries common to text-to-SQL benchmarks [18], [19].

## Rationale
- Small, domain-specific set keeps QLoRA feasible on 8–12GB GPUs [12], [4], [5].
- Strict split enables fair baseline vs fine-tune comparison (few-shot vs QLoRA, k=0/k=3), following Ojuri et al. (2025) [10].
