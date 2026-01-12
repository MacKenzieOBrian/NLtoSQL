# Evaluation Framework

This project reproduces the evaluation methodology used in Ojuri et al. (2025) to enable direct comparison between proprietary and open-source NL→SQL systems.

The evaluation measures three dimensions of correctness:

---

## 1. Valid SQL (VA)

### Definition
Determines whether generated SQL is syntactically valid and executable by the database engine.

### Measurement
Binary outcome: `valid` if the query executes without syntax errors.

### Rationale
SQL syntax errors are common in single-shot decoding. Measuring VA isolates syntactic failure modes.

### Literature
Execution-based syntax validation used in text-to-SQL benchmarks (Zhong et al., 2020).

---

## 2. Execution Accuracy (EX)

### Definition
Determines whether the executed query returns the correct result set on the evaluation database.

### Measurement
For each query:  
`EX = 1` if returned rows match ground-truth rows, else `0`.

### Rationale
Execution accuracy evaluates functional correctness rather than surface string similarity. Two SQL strings can differ but return the same results.

### Literature
Execution accuracy is the primary metric in Spider and related NL→SQL tasks (Yu et al., 2018).

---

## 3. Test-Suite Accuracy (TS)

### Definition
Measures logical correctness by running the query against multiple perturbed versions of the database with modified data.

### Measurement
`TS = 1` only if outputs match across all modified database instances.

### Rationale
A query that accidentally produces correct results on the original DB may fail under data perturbations, revealing semantic errors.

### Literature
Test-Suite metric introduced by Zhong et al. (2020); adopted in Ojuri et al. (2025).

---

## Dataset

The benchmark consists of 200 NL→SQL pairs over the classicmodels database. This matches the dataset used in Ojuri et al. (2025) and supports direct comparison.

---

## Evaluation Pipeline (Implementation)

In this repository, evaluation is implemented via:

- `nl2sql/eval.py` (VA/EX/EM utilities; TS planned)
- Notebooks: `02_baseline_prompting_eval.ipynb` and `05_qlora_train_eval.ipynb`

The pipeline:

1. Generate SQL via chosen strategy
2. Execute SQL via `QueryRunner`
3. Compare output against expected result
4. Compute VA, EX, EM (TS planned)
5. Aggregate scores across 200 samples

---

## Computational Feasibility Evaluation

In addition to correctness metrics, this project evaluates:

- VRAM usage
- training time
- inference latency

These measurements assess whether NL→SQL is viable under realistic hardware constraints such as Colab GPUs.

---

## Summary

This evaluation framework enables reproduction of proprietary-agent performance and supports empirical comparison between:

- zero-shot prompting
- few-shot prompting
- PEFT fine-tuning
- agentic refinement
