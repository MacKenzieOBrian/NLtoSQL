# Decision Record (Dissertation)

This is the single place to understand **what decisions were made, why, and where they are implemented**.

## Research design

| Decision | Why (justification) | Where |
|---|---|---|
| Compare prompting vs fine-tuning | Establish inference-only baselines before claiming fine-tuning uplift; matches the comparative framing in Ojuri et al. | `notebooks/02_baseline_prompting_eval.ipynb`, `notebooks/05_qlora_train_eval.ipynb` |
| Fixed ~200-item test benchmark | Small enough for repeated controlled experiments; large enough to observe patterns | `data/classicmodels_test_200.json` |
| Strict train/test separation | Prevents leakage; preserves validity of generalisation claims | `notebooks/04_build_training_set.ipynb`, `data/train/README.md` |

## Why we run (k=0) and (k=3) in multiple phases

| Phase | What changes | Why it matters |
|---|---|---|
| Baseline, `k=0` | Prompt only (no exemplars); weights fixed | Establishes a “prompt-only” floor. |
| Baseline, `k=3` | Prompt includes exemplars; weights fixed | Measures inference-time prompt conditioning uplift without training. |
| QLoRA, `k=0` | Weights changed via adapters; no exemplars | Measures training uplift on its own (closest to “fine-tuned model only”). |
| QLoRA, `k=3` | Adapters + exemplars | Tests whether prompting still helps after fine-tuning; useful for selecting a final deployment mode and for dissertation analysis. |

## Metrics (Ojuri-aligned)

| Metric | Decision | Why | Where |
|---|---|---|---|
| `VA` | Executability | Required to measure whether generated SQL can run on a live DB | `nl2sql/query_runner.py`, `nl2sql/eval.py` |
| `EM` | Exact string match | Useful debugging signal, but conservative | `nl2sql/eval.py` |
| `EX` | Execution accuracy (result comparison) | Matches Ojuri’s “execution accuracy” definition | `nl2sql/eval.py` (`execution_accuracy`) |
| `TS` | Test-suite accuracy (planned) | Stronger semantic check across distilled DB variants | Documented in `ARCHITECTURE.md`/`DATA.md` |

## Safety and DB access

| Decision | Why | Where |
|---|---|---|
| Read-only enforcement (block DDL/DML tokens) | Protects the DB from destructive outputs | `nl2sql/query_runner.py`, `nl2sql/eval.py` |
| Cloud SQL Connector + SQLAlchemy creator | Secure access without exposing DB endpoints; Colab-compatible | `nl2sql/db.py` |

## Reproducibility

| Decision | Why | Where |
|---|---|---|
| Colab as primary runtime | GPU availability + consistent environment for repeated runs | `CONFIG.md`, `notebooks/` |
| Pinned dependencies | Prevents Colab binary drift changing results | `requirements.txt` |
| Deterministic decoding for baselines | Avoids sampling noise in VA/EM/EX comparisons | `nl2sql/llm.py`, baseline notebooks |
| Outputs under `results/` (gitignored) | Prevents accidental large commits; curate dissertation artifacts intentionally | `.gitignore`, `results/README.md` |

## Training data (QLoRA)

| Decision | Why | Where |
|---|---|---|
| LLM-assisted generation + DB validation | Scales data creation while ensuring VA=True | `notebooks/04_build_training_set.ipynb` |
| Mixed difficulty targets | Broad coverage (easy/medium/hard) with a small dataset | `notebooks/04_build_training_set.ipynb` |

## What to cite in the dissertation

- Metrics and evaluation framing: Ojuri et al. (VA / execution accuracy / test-suite accuracy).
- Test-suite accuracy concept: Zhong et al. (2020) style denotation comparison across DB variants.
