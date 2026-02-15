# Experiment Execution Plan (Concise)

## Control rule
Change one axis at a time:
1. model (`MODEL_ID`)
2. prompting (`K_VALUES`, `SEEDS`)
3. prompt variant
4. schema variant
5. exemplar strategy
6. TS toggle

## Core grid
- `K_VALUES=[0,3,5]`
- `SEEDS=[7,17,27]`
- Methods: baseline + qlora
- Models: llama3_8b_instruct + qwen2_5_7b_instruct

## Run order
1. Baseline quick check (`k=[0,3]`, `seed=[7]`)
2. QLoRA quick check (`k=[0,3]`, `seed=[7]`)
3. Baseline full grid
4. QLoRA full grid
5. Optional TS checks (`k=3`, small seed set)
6. Regenerate analysis outputs

## Notebooks
- Baseline: `notebooks/02_baseline_prompting_eval.ipynb`
- QLoRA: `notebooks/05_qlora_train_eval.ipynb`
- ReAct infra: `notebooks/03_agentic_eval.ipynb`
- Comparison: `notebooks/06_research_comparison.ipynb`

## Completion check
- confirm coverage in `results/analysis/run_manifest.csv`
- regenerate summary via `scripts/generate_research_comparison.py`

## Required output artifacts
- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/run_manifest.csv`
