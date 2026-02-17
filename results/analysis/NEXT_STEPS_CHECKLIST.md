# Next Steps Checklist (Concise)

## Current status (2026-02-15)
- Logged: Qwen baseline full-sweep partial import in `results/baseline/runs/qwen2_5_7b_e1_k_sweep_20260215_imported/`.
- Logged backfill: `results/baseline/runs/qwen2_5_7b_e1_k0_backfill_20260215_imported/results_k0_seed17.json` and `results_k0_seed27.json`.
- Logged combos: `(k,seed) = (0,7), (0,17), (0,27), (3,7), (3,17), (3,27), (5,7), (5,17), (5,27)`.
- Qwen baseline grid status: complete.

## Core run completion
1. Complete baseline grid for both models (`k=[0,3,5]`, `seeds=[7,17,27]`).
2. Complete QLoRA grid for both models with same `k/seed` setup.
3. Run optional TS checks at `k=3` after EX grid is complete.

## Immediate next step
- Start Qwen QLoRA quick run (`k=[0,3]`, `seed=[7]`) to validate end-to-end QLoRA path before full sweep.

## Run hygiene
- Change one knob at a time.
- Use model-specific `RUN_TAG` names.
- Keep `PROMPT_VARIANT=default`, `SCHEMA_VARIANT=full`, `EXEMPLAR_STRATEGY=all` for core comparisons.

## After runs
- Regenerate analysis with `scripts/generate_research_comparison.py`.
- Confirm coverage in `results/analysis/run_manifest.csv`.
- Write claims from `paired_deltas.csv` + `failure_taxonomy.csv`.
