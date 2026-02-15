# Next Steps Checklist

## Immediate run sequence

| Step | Notebook | Change only this | Keep fixed | Output check |
| --- | --- | --- | --- | --- |
| 1 | `notebooks/02_baseline_prompting_eval.ipynb` | Qwen full sweep: `MODEL_ID`, `MODEL_ALIAS`, `K_VALUES=[0,1,3,5,8]`, `SEEDS=[7,17,27,37,47]`, `RUN_TAG=qwen2_5_7b_e1_k_sweep` | `PROMPT_VARIANT=default`, `SCHEMA_VARIANT=full`, `EXEMPLAR_STRATEGY=all` | `results/baseline/runs/qwen2_5_7b_e1_k_sweep_<timestamp>/` |
| 2 | same notebook | TS check: `K_VALUES=[3]`, `SEEDS=[7]`, `ENABLE_TS=True`, `TS_FOR_K_VALUES=[3]`, `TS_N=10` | all other baseline knobs | JSON includes `ts` and `ts_rate` |
| 3 | `notebooks/05_qlora_train_eval.ipynb` | Qwen quick QLoRA: `MODEL_ID`, `MODEL_ALIAS`, `K_VALUES=[0,3]`, `SEEDS=[7]`, `RUN_TAG=qwen2_5_7b_qlora_main` | prompt/schema/exemplar defaults | `results/qlora/model_family/qwen2_5_7b_instruct_qlora_k0.json` and `_k3.json` |
| 4 | same notebook | Qwen full QLoRA sweep: `K_VALUES=[0,1,3,5,8]`, `SEEDS=[7,17,27,37,47]`, `RUN_TAG=qwen2_5_7b_qlora_e1_k_sweep` | same prompt/schema/exemplar | full QLoRA run folder + summaries |
| 5 | `notebooks/06_research_comparison.ipynb` | no config changes | n/a | refreshed `results/analysis/*.csv` + figures |

## Run hygiene

- Use `COPY_MODEL_FAMILY=True` for cross-model comparisons.
- Use `COPY_CANONICAL=False` for non-primary sweeps.
- Change one axis at a time when making claims.
