# Experiment Execution Plan

## Purpose

Run controlled comparisons for:
- prompting effect (`k=0` vs few-shot),
- fine-tuning effect (base vs QLoRA),
- optional ReAct infrastructure checks,
- optional model-family robustness checks.

## Control Rule

Change one axis at a time:
1. model (`MODEL_ID`),
2. prompting depth (`K_VALUES`, `SEEDS`),
3. prompt (`PROMPT_VARIANT`),
4. schema context (`SCHEMA_VARIANT`),
5. exemplar strategy (`EXEMPLAR_STRATEGY`),
6. TS toggle (`ENABLE_TS`).

Keep all other knobs fixed when making a claim.

## Run Order (default)

1. Baseline quick (`k=0,3`, seed `7`).
2. QLoRA quick (`k=0,3`, seed `7`).
3. Baseline full sweep (`k=[0,1,3,5,8]`, seeds `[7,17,27,37,47]`).
4. QLoRA full sweep (same grid).
5. Optional TS check on `k=3` (small seed set).
6. Regenerate analysis artifacts.

## Notebooks

- Baseline: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/02_baseline_prompting_eval.ipynb`
- QLoRA: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/05_qlora_train_eval.ipynb`
- ReAct infra: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/03_agentic_eval.ipynb`
- Comparison: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/06_research_comparison.ipynb`

## Post-run command

```bash
python /Users/mackenzieobrian/MacDoc/Dissertation/scripts/generate_research_comparison.py --out-dir /Users/mackenzieobrian/MacDoc/Dissertation/results/analysis
```

## Minimum artifacts for writing

- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/run_manifest.csv`
