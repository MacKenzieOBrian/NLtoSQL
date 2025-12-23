# Notes (decisions, assumptions, TODOs)

This file is intentionally less formal than `ARCHITECTURE.md`/`CONFIG.md`. It captures “why we did it this way” and what still needs to be done before dissertation write-up.

## Current structure (how work is organised)

- Implementation lives in `nl2sql/` (DB, schema, prompting, eval).
- Notebooks in `notebooks/` orchestrate runs and generate dissertation artifacts.
- Outputs go to `results/` (gitignored by default; commit only curated artifacts).

## Baselines first (before agents or fine-tuning)

- Goal: establish a transparent prompting baseline before adding more powerful methods.
- Definition: “few-shot” = inference-time exemplars only (weights frozen).
- Control: deterministic decoding and fixed seeds for repeatable VA/EX runs.

## Evaluation hygiene (important)

- VA is an apparatus check (does it execute?).
- EX is strict string match (conservative; undercounts semantically correct SQL).
- Next metric: result-equivalence / TS proxy (execute gold vs pred and compare results).
- Few-shot exemplar policy: for dissertation-quality evaluation, exemplars must come from a **non-test** pool and must not include the evaluated item.

## Agent plan (ReAct-style)

- QueryRunner is the tool: Action = execute candidate SQL; Observation = errors/row counts/columns.
- Log traces for analysis: prompt, intermediate SQL, error messages, final SQL.

## QLoRA plan (later)

- Use 4-bit NF4 loading to keep training feasible on academic GPUs.
- Log: LoRA config + VRAM + runtime + seed + dataset version.

## Reproducibility checklist (what to always record)

- git commit hash, notebook name, run timestamp
- model id + loading config
- seed, `k`, prompt version, post-processing toggles
- dataset version/hash
