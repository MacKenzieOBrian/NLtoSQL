# Next Steps Checklist (Concise)

## Core run completion
1. Complete baseline grid for both models (`k=[0,3,5]`, `seeds=[7,17,27]`).
2. Complete QLoRA grid for both models with same `k/seed` setup.
3. Run optional TS checks at `k=3` after EX grid is complete.

## Run hygiene
- Change one knob at a time.
- Use model-specific `RUN_TAG` names.
- Keep `PROMPT_VARIANT=default`, `SCHEMA_VARIANT=full`, `EXEMPLAR_STRATEGY=all` for core comparisons.

## After runs
- Regenerate analysis with `scripts/generate_research_comparison.py`.
- Confirm coverage in `results/analysis/run_manifest.csv`.
- Write claims from `paired_deltas.csv` + `failure_taxonomy.csv`.
