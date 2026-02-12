# Experiment Execution Plan

This runbook turns the dissertation plan into a repeatable sequence with minimal notebook edits.

## Primary Goal

Establish defensible differences for:
- prompting (`k=0` vs `k>0`),
- fine-tuning (base vs QLoRA),
- execution infrastructure (ReAct as validity support, not primary semantic method).

## Seed Rule (important)

- `k=0`: no exemplars are sampled, so seed has no practical effect.
- `k>0`: seed changes which exemplar set is sampled.
- "5 seeds per k" means: run the same setup 5 times with different seeds, then summarize mean/variance and intervals.

## Where to Run

- Baseline + ablations: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/02_baseline_prompting_eval.ipynb`
- QLoRA + ablations: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/05_qlora_train_eval.ipynb`
- ReAct infrastructure run: `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/03_agentic_eval.ipynb`
- Comparison tables/figures: `/Users/mackenzieobrian/MacDoc/Dissertation/scripts/generate_research_comparison.py`

Both baseline/QLoRA notebooks now expose:
- `K_VALUES`
- `SEEDS`
- `EXEMPLAR_STRATEGY`
- `PROMPT_VARIANT`
- `SCHEMA_VARIANT`
- `run_*_grid(...)` helpers that save per-run JSONs and summary CSVs.

## Standard Run Order

### E1: Core comparison with stability

Run both notebooks with:
- `K_VALUES = [0, 3]`
- `SEEDS = [7, 11, 19, 23, 31]`
- `PROMPT_VARIANT = "default"`
- `SCHEMA_VARIANT = "full"`
- `EXEMPLAR_STRATEGY = "all"`

Outputs:
- `results/baseline/runs/<tag_timestamp>/...`
- `results/qlora/runs/<tag_timestamp>/...`
- each folder includes per-run JSON and `grid_summary*.csv`.

### E2: k-sweep

Run both notebooks with:
- `K_VALUES = [0, 1, 2, 3, 4, 5]`
- same 5 seeds
- default prompt/schema/exemplar controls.

Use this to show how gains saturate with more demonstrations.

### E3: exemplar strategy ablation (few-shot only)

Run both notebooks with:
- `K_VALUES = [3]`
- same 5 seeds
- `EXEMPLAR_STRATEGY` in:
  - `"all"`
  - `"brief_sql"`
  - `"join_heavy"`
  - `"agg_heavy"`

Use this to quantify sensitivity to exemplar composition.

### E4: prompt/schema ablation

Run both notebooks with:
- `K_VALUES = [3]`
- same 5 seeds
- prompt variants:
  - `"default"`
  - `"schema_only_minimal"`
  - `"no_routing_hints"`
- schema variants:
  - `"full"`
  - `"first_80_lines"`
  - `"first_40_lines"`

Use this to show how prompt scaffolding and schema detail affect EX/TS.

### E5: ReAct infrastructure check

Run `/Users/mackenzieobrian/MacDoc/Dissertation/notebooks/03_agentic_eval.ipynb` in core mode only:
- schema setup once,
- generate -> validate -> execute,
- repair only on validation/execution failure,
- stop on first successful execution.

Interpretation rule:
- use ReAct primarily for VA/traceability stabilization,
- avoid claiming it fully solves semantic alignment.

## Canonical Files for Cross-Method Comparison

Keep these as the "active" comparison set:
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/baseline/baseline_k0.json`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/baseline/baseline_k3.json`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/qlora/qlora_k0.json`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/qlora/qlora_k3.json`
- latest ReAct run JSON under `/Users/mackenzieobrian/MacDoc/Dissertation/results/agent/`

Archive all other run copies under:
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/runs/`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/archive/`

## Post-Run Analysis

From repo root:

```bash
python scripts/generate_research_comparison.py
```

Expected artifacts:
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/overall_metrics_long.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/overall_metrics_wide.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/paired_deltas.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/failure_taxonomy.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/summary.md`

## Reporting Template

For each dissertation claim:
1. state the exact contrast (for example base k=3 vs base k=0),
2. report VA/EX/TS/EM + 95% interval,
3. include paired delta and McNemar significance where applicable,
4. explain change using error taxonomy shifts.
