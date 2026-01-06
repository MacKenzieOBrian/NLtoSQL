# Configuration & Reproducibility

This document describes how to reproduce experiments (baseline prompting now; agent + QLoRA planned). It is written to support dissertation-quality runs: fixed dependencies, deterministic decoding, and traceable run metadata.

For the “why” behind key choices (metrics, splits, safety, reproducibility), see `DECISIONS.md`.

## Project structure (why it changed)

The repo is intentionally split into:
- `nl2sql/` (importable “experiment harness”): stable, reviewable code for DB access, safe execution, prompting, and evaluation.
- `notebooks/` (Colab runners): orchestrate runs, save artifacts, and generate dissertation tables/figures without duplicating core logic.
- `data/` (benchmarks) and `results/` (run outputs): keep inputs and outputs separate; `results/` is gitignored by default to avoid committing large artifacts by accident.

This makes runs easier to reproduce: the notebook becomes a thin runner, while evaluation logic lives in version-controlled modules.

## Quickstart (Colab baseline)

1. Use a GPU runtime (T4/A100).
2. Fresh clone the repo into `/content` (recommended) and record the commit hash.
3. Install pinned deps from `requirements.txt`, then restart the runtime.
4. Authenticate:
   - GCP: `google.colab.auth.authenticate_user()` (or ADC locally)
   - Hugging Face: `notebook_login()` or `HF_TOKEN`
5. Run: `notebooks/02_baseline_prompting_eval.ipynb`
6. Outputs are saved under `results/baseline/` (gitignored by default; see `results/README.md`).

## Quickstart (QLoRA)

1. Build a training set that does not overlap the 200-item benchmark:
   - Run `notebooks/04_build_training_set.ipynb` to generate and DB-validate `data/train/classicmodels_train_200.jsonl`.
2. Fine-tune and evaluate adapters:
   - Run `notebooks/05_qlora_train_eval.ipynb` (saves adapters to `results/adapters/` and eval JSONs to `results/qlora/`).

## Environment variables

Set these before running notebooks (or enter when prompted):

| Variable | Purpose | Example |
|----------|---------|---------|
| `INSTANCE_CONNECTION_NAME` | Cloud SQL instance identifier | `modified-enigma-476414-h9:europe-west2:classicmodels` |
| `DB_USER` | MySQL username | `root` |
| `DB_PASS` | MySQL password | — |
| `DB_NAME` | Database name | `classicmodels` |
| `HF_TOKEN` | Hugging Face access token (gated models) | `REDACTED` |

## Dependencies

Pinned in `requirements.txt` to avoid Colab binary drift. The most failure-prone pins are `numpy`, `pandas`, `torch`, `triton`, and `bitsandbytes`.

Recommended Colab flow:
1. `pip install -r requirements.txt`
2. Restart runtime
3. Re-run notebook from the top

## Authentication

### Google Cloud (Cloud SQL)
- Colab: `from google.colab import auth; auth.authenticate_user()`
- Local: `gcloud auth application-default login` (ADC) or a service account key (avoid committing secrets).

### Hugging Face (gated model)
- Meta Llama 3 Instruct is gated: access approval + token are both required.
- Use `from huggingface_hub import notebook_login; notebook_login()` or set `HF_TOKEN`.

## Baseline prompting settings

- Model: `meta-llama/Meta-Llama-3-8B-Instruct` (token gated)
- Loading: 4-bit NF4 where possible (fits Colab GPUs, aligns with planned QLoRA)
- Decoding: deterministic for reporting (`do_sample=False`, bounded `max_new_tokens`, `pad_token_id=eos_token_id`)

## Evaluation metrics (Ojuri-style)

- **VA (Validity)**: predicted SQL executes successfully (syntactic/engine validity check).
- **EM (Exact Match)**: normalized SQL string match vs gold SQL (strict, conservative).
- **EX (Execution Accuracy)**: execute predicted SQL and compare its result set to the gold SQL result set.
- **TS (Test-Suite Accuracy)**: planned; compare predicted vs gold results across multiple distilled DB variants.

## Reproducibility checklist (log per run)

Record alongside results:
- git commit hash
- model id + quantization settings
- prompt template version (and any post-processing toggles)
- `k` (few-shot exemplars), random seed, benchmark version/hash
- GPU type/runtime notes (Colab)

## QLoRA (planned)

This project will compare prompting vs QLoRA SFT later. Key knobs to report in the dissertation:
- LoRA rank `r`, `alpha`, dropout, target modules
- batch size + grad accumulation, max seq length, LR/scheduler, warmup
- quantization config (4-bit NF4) and dtype

## Why these choices

See `DECISIONS.md` (kept separate so this file stays a “how to run” checklist).
