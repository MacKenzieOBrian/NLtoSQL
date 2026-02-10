# QLoRA Configuration and Run Logging

This document explains the default QLoRA setup used in `notebooks/05_qlora_train_eval.ipynb` and provides a run log format for dissertation reporting.

## Scope

- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Training data: `data/train/classicmodels_train_200.jsonl`
- Adapter output: `results/adapters/qlora_classicmodels`
- Goal: stable, reproducible adaptation under Colab-level compute constraints.

## TrainingArguments Rationale

| Setting | Value | Why this is used | Tradeoff |
| --- | --- | --- | --- |
| `per_device_train_batch_size` | `1` | Fits 8B + 4-bit + LoRA on limited VRAM (for example Colab T4). | Higher gradient noise per step. |
| `gradient_accumulation_steps` | `8` | Restores a practical effective batch size (`1 x 8 = 8`) without extra VRAM. | Slower wall-clock per optimizer step. |
| `learning_rate` | `1e-4` | Standard LoRA SFT range for fast adaptation on small supervised sets. | Can overfit if epochs are too high. |
| `num_train_epochs` | `3` | Enough exposure for a 200-example corpus without long over-training cycles. | May still overfit narrow patterns; verify with held-out checks. |
| `warmup_ratio` | `0.05` | Reduces unstable early updates at the start of training. | Slightly delays full-rate learning. |
| `logging_steps` | `10` | Gives frequent loss checkpoints for auditability in short runs. | Slightly noisier logs. |
| `save_steps` | `200` | Avoids many intermediate checkpoints in short jobs. | Limited rollback points during one run. |
| `save_total_limit` | `2` | Keeps storage bounded in Colab/workspace runs. | Fewer historical checkpoints retained. |
| `bf16` | `use_bf16` | Uses bf16 when hardware supports it (Ampere+), improving numerical behavior. | Not available on many free GPUs. |
| `fp16` | `not use_bf16` | Fallback precision for GPUs without bf16 support (for example T4). | More overflow/underflow risk than bf16. |
| `optim` | `"paged_adamw_8bit"` | Memory-efficient optimizer aligned with QLoRA constraints. | Slightly different optimizer dynamics vs full AdamW. |
| `report_to` | `[]` | Keeps runs self-contained without external tracker dependencies. | No automatic dashboard history. |

## LoRA Adapter Rationale

| Setting | Value | Why this is used |
| --- | --- | --- |
| `r` | `32` | Moderate adapter capacity for semantic mapping without full-model tuning. |
| `lora_alpha` | `64` | Common scaling for `r=32`; keeps update magnitude balanced. |
| `lora_dropout` | `0.05` | Light regularization to reduce memorization on small datasets. |
| `bias` | `"none"` | Standard PEFT choice to minimize trainable parameter count. |
| `target_modules` | `q_proj`, `v_proj` | Focuses adaptation on attention projections with good memory/performance tradeoff. |

## Logging Rule for Dissertation Claims

For every reported QLoRA run, record the exact resolved values used at runtime, especially precision mode:

- `bf16=True, fp16=False` or `bf16=False, fp16=True`
- GPU type and CUDA availability
- exact package versions (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`)
- commit hash and output adapter path

Do not report only code expressions such as `bf16=use_bf16`; report the realized values.

## Run Record Template

```md
### QLoRA Run Record: <RUN_ID>

- Date: <YYYY-MM-DD>
- Commit: <git short hash>
- Notebook: `notebooks/05_qlora_train_eval.ipynb`
- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Train file: `data/train/classicmodels_train_200.jsonl` (n=<N>, overlap check=<pass/fail>)
- Output dir: `results/adapters/<adapter_name>`

TrainingArguments:
- per_device_train_batch_size=1
- gradient_accumulation_steps=8
- learning_rate=1e-4
- num_train_epochs=3
- warmup_ratio=0.05
- logging_steps=10
- save_steps=200
- save_total_limit=2
- optim="paged_adamw_8bit"
- bf16=<True/False used>
- fp16=<True/False used>
- report_to=[]

LoRA:
- r=32
- lora_alpha=64
- lora_dropout=0.05
- bias="none"
- target_modules=["q_proj","v_proj"]

Environment:
- GPU: <name>
- CUDA available: <True/False>
- torch=<version>
- transformers=<version>
- peft=<version>
- trl=<version>
- bitsandbytes=<version>

Evaluation outputs:
- `results/qlora/results_zero_shot_200.json`
- `results/qlora/results_few_shot_k3_200.json`
```

## Optional Hardening for Repeated-Run Statistics

If you want tighter rerun comparability, add the following into `TrainingArguments`:

- `seed=7`
- `data_seed=7`

This does not change the methodological framing, but it reduces run-to-run variance caused by data order and stochastic kernels.
