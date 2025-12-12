## 1. Overview

This document provides configuration details required to reproduce all experiments in the project, including:

- Environment variables  
- Authentication steps  
- Hardware/GPU assumptions  
- Python dependencies  
- Cloud SQL access requirements  
- HuggingFace model access  

Quick reading note: everything here is geared to inference-time prompting first (few-shot baseline) with the model frozen, so any gains are from prompt design and post-processing, not from training. Pins + seeds + deterministic decoding are the guardrails for reproducibility.

## 2. Environment Variables

The following variables must be defined **before** running the notebook:

| Variable | Purpose | Example |
|----------|---------|---------|
| `INSTANCE_CONNECTION_NAME` | Cloud SQL instance identifier | `modified-enigma-476414-h9:europe-west2:classicmodels` |
| `DB_USER` | MySQL username | `root` |
| `DB_PASS` | MySQL password | — |
| `DB_NAME` | Database name | `classicmodels` |
| `HF_TOKEN` | HuggingFace access token for gated models | `REDACTED_HF_TOKEN` |

## 3. Runtime / Notebook Setup (Colab)
- Runtime: GPU-enabled (T4/A100), Python 3.12.  
- GitHub: start each session with a **fresh clone** in `/content` (`rm -rf /content/NLtoSQL && git clone ...`) to avoid stale files; record commit hash (`git rev-parse --short HEAD`).  
- Missing files (requirements.txt, JSON test set, .md docs): pull from repo; notebook cells reference these files.  
- After installs: **restart runtime** to clear stale C-extensions from Colab base images.

## 4. Python Dependencies
Pinned in `requirements.txt`. Critical pins and reasons:
- `torch==2.2.2`, `transformers==4.37.2`, `bitsandbytes==0.42.0`, `triton==2.2.0`, `pandas==2.2.1`
- `numpy==1.26.4` — fixes the binary incompatibility error (`ValueError: numpy.dtype size changed...`) seen with newer NumPy wheels.

Workflow:
1) `pip install -r requirements.txt`  
2) Restart runtime  
3) Verify versions: NumPy, Pandas, Torch align (avoids C-extension crashes).  

## 5. Hugging Face Authentication (Gated Models)
- Use `from huggingface_hub import notebook_login; notebook_login()` or set `HF_TOKEN`.  
- Meta-Llama-3-8B-Instruct is **gated**: token + approved access on the model page are both required.  
- 403 after login usually means authorization pending; request access with affiliation (university/independent).

## Models
- Base: `meta-llama/Meta-Llama-3-8B-Instruct`
- Tokenizer: same.

## QLoRA idea as my understanding (see definitions below)
- 4-bit quantization.
- LoRA r/alpha/dropout: TBD (e.g., r 16–64). Target attention/MLP.
- SFT on curated NLQ-SQL; batch/gradacc tuned to VRAM; log peak VRAM + runtime; capture seed.

## Core QLoRA knobs
- **Quantization (4-bit)**: Store model weights in 4-bit to fit on smaller GPUs; compute still uses higher precision (e.g., bfloat16).
- **LoRA rank (r)**: Size of the low-rank adapter matrices; higher r = more capacity, more VRAM. Common range 16–64.
- **Alpha (LoRA alpha)**: Scaling factor for the adapter; larger alpha amplifies the adapter contribution. Often set to 2–4× r.
- **Dropout**: Probability of dropping adapter activations during training to reduce overfitting; small values (e.g., 0.05–0.1) are typical.
- **Target modules**: Which layers get adapters (e.g., attention projections, MLP projections). Needs to match model architecture.
- **Grad checkpointing**: Trades compute for memory by re-computing activations on backward pass; useful to fit bigger models.
- **Batch size / Grad accumulation**: Effective batch = micro-batch × grad accum steps; adjust to stay under VRAM limits.
- **Learning rate / Scheduler**: Small LR for SFT (e.g., 5e-5 to 2e-4); cosine/linear schedulers are common.
- **Max seq length**: Truncation length for inputs; longer costs more VRAM.
- **Warmup steps**: Small ramp-up of LR to stabilize early training.
- **Seed**: For reproducibility; log it with run configs.

## Prompting
- Few-shot baseline: schema + table blurbs + 2–4 exemplars + NLQ.
- ReAct: tool description, schema context, and running Thought/Action/Observation trace.

## Model Access & Loading
- Model: `meta-llama/Meta-Llama-3-8B-Instruct` (gated).
- Auth: Hugging Face token required. Request access on the model page. In notebooks, run `from huggingface_hub import notebook_login; notebook_login()` or set `HUGGINGFACE_HUB_TOKEN`/`HF_TOKEN` and pass `token=True` on load.
- Loading: 4-bit (NF4) quantization via bitsandbytes, `device_map="auto"` (GPU-backed), use chat template (`apply_chat_template`), pad-token fallback to EOS if needed, deterministic decoding (`temperature=0`) for reproducible evaluation.

## Few-Shot Baseline Run (for VA/EX)
- Build prompt: schema summary + 2–4 NLQ→SQL exemplars + new NLQ (fixed template/versioned).  
- Generation: deterministic (`do_sample=False`, omit `temperature/top_p`, modest `max_new_tokens`, set `pad_token_id=eos_token_id`).  
- Evaluation: run generated SQL through QueryRunner against `data/classicmodels_test_200.json`; compute VA/EX.  
- Provenance: log commit hash, prompt version, generation params, and hardware in notebook + LOGBOOK to mirror Ojuri-style academic reporting.

## Reproducibility and Inference-Only Guardrails
- Model usage is **strictly inference-time**: no fine-tuning, adapters, or parameter updates during baselines.  
- Fix seeds for exemplar selection and keep decoding deterministic so VA/EX shifts reflect prompt changes, not randomness.  
- Capture run metadata (commit, prompt template, hardware) alongside outputs to make results re-runnable for the dissertation.

