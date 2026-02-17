# Next Steps Checklist

## Current state (snapshot-aware)
Completed and logged:
- Llama baseline: `k=0`, `k=3`.
- Llama QLoRA eval: `k=0`, `k=3`.
- Qwen baseline: `k=0`, `k=3`, `k=5` (core imported runs).
- Qwen QLoRA quick run: `k=0,3`, `seed=7` from `results/qlora/runs/qwen2_5_7b_qlora_main_20260217_111905Z/`.
- ReAct infrastructure run: `n=20` slice for diagnostic comparison.

Key analysis outputs already generated:
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/run_manifest.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/paired_deltas.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/failure_taxonomy.csv`

## Priority order to finish dissertation evidence
1. Complete missing matched cells for model-family fairness (same `k`/seed policy across compared methods).
2. Add seed robustness for final headline comparisons (minimum planned seed set).
3. Run TS checks on the stable `k=3` checkpoints used for semantic claims.
4. Regenerate analysis with `/Users/mackenzieobrian/MacDoc/Dissertation/scripts/generate_research_comparison.py`.
5. Freeze a final claim table (delta + CI + McNemar) and use that as dissertation source of truth.

## Run hygiene rules
- Change one knob per experiment.
- Use model-specific `RUN_TAG` values.
- Keep prompt/schema/exemplar constant for core method comparisons.
- Log hardware/runtime constraints (including OOM events) as part of reproducibility evidence.

## Writing workflow after each new run batch
1. Update artifacts in `results/analysis/`.
2. Confirm run coverage in `run_manifest.csv`.
3. Update claim statements in markdown using paired results, not raw VA only.
