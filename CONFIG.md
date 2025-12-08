# CONFIG (notes to self)

## Environment
- Colab + Google Cloud SQL (classicmodels).
- Connector: `cloud-sql-python-connector[pymysql]` + SQLAlchemy creator.
- Hardware: jot GPU model, VRAM, RAM per run in LOGBOOK.
- Secrets: env/interactive for now; move to Secret Manager later.
- Env vars (dev convenience values):
  - `INSTANCE_CONNECTION_NAME=modified-enigma-476414-h9:europe-west2:classicmodels`
  - `DB_USER=root`
  - `DB_PASS=<your_password>`
  - `DB_NAME=classicmodels`
  - `GOOGLE_CLOUD_PROJECT=modified-enigma-476414-h9`

## Deps to pin (requirements.txt)
- python 3.10+
- torch==2.2.2
- transformers==4.37.2
- accelerate==0.27.2
- bitsandbytes==0.42.0
- peft==0.10.0
- trl==0.7.10
- datasets==2.16.1
- google-api-core==2.11.1
- cloud-sql-python-connector[pymysql]==1.18.5
- SQLAlchemy==2.0.7
- pymysql==1.1.0
- cryptography==42.0.5
- pandas==2.2.1

## Models
- Base: `meta-llama/Meta-Llama-3-8B-Instruct`
- Tokenizer: same.

## QLoRA sketch (see definitions below)
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

## Evaluation
- Metrics: VA, EX, TS.
- Data: 200-sample test set; distilled variants for TS.
- Logging: save run configs (seed, hyperparams, dataset snapshot hashes) with metrics.
