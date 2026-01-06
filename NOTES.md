# Notes (open questions / TODOs)

This file is intentionally short and action-oriented. The main decision rationale lives in `DECISIONS.md`.

## Current structure (how work is organised)

- Implementation lives in `nl2sql/` (DB, schema, prompting, eval).
- Notebooks in `notebooks/` orchestrate runs and generate dissertation artifacts.
- Outputs go to `results/` (gitignored by default; commit only curated artifacts).

## Immediate TODOs

- Re-run baselines after metric changes and archive JSON artifacts (VA/EM/EX) from Colab.
- Add TS-style evaluation (distilled DB variants + denotation comparison).
- Create an exemplar pool separate from the test benchmark for few-shot runs (train/dev split).

## Agent plan (ReAct-style)

- QueryRunner is the tool: Action = execute candidate SQL; Observation = errors/row counts/columns.
- Log traces for analysis: prompt, intermediate SQL, error messages, final SQL.

## QLoRA TODOs

- Decide train size beyond 200 (if needed) and document (manual + synthetic mix).
- Add a small dev split for hyperparameter selection (separate from test).
- Log final QLoRA hyperparams and training compute (steps, LR, seq length, GPU).

## Reproducibility checklist (what to always record)

- git commit hash, notebook name, run timestamp
- model id + loading config
- seed, `k`, prompt version, post-processing toggles
- dataset version/hash
