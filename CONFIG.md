# CONFIG (notes to self)

## Environment
- Colab + Google Cloud SQL (classicmodels).
- Connector: `cloud-sql-python-connector[pymysql]` + SQLAlchemy creator.
- Hardware: jot GPU model, VRAM, RAM per run in LOGBOOK.
- Secrets: env/interactive for now; move to Secret Manager later.

## Deps to pin (add to requirements cell/file)
- python 3.10+
- torch (match Colab GPU)
- transformers (pin)
- accelerate (pin)
- bitsandbytes (pin)
- peft (pin)
- trl (pin)
- datasets (pin)
- cloud-sql-python-connector[pymysql]
- SQLAlchemy==2.0.7
- pymysql
- cryptography==41.0.0
- pandas

## Models
- Base: `meta-llama/Meta-Llama-3-8B-Instruct`
- Tokenizer: same.

## QLoRA sketch (see TECH_NOTES.md for definitions)
- 4-bit quantization.
- LoRA r/alpha/dropout: TBD (e.g., r 16–64). Target attention/MLP.
- SFT on curated NLQ-SQL; batch/gradacc tuned to VRAM; log peak VRAM + runtime; capture seed.

## Prompting
- Few-shot baseline: schema + table blurbs + 2–4 exemplars + NLQ.
- ReAct: tool description, schema context, and running Thought/Action/Observation trace.

## Evaluation
- Metrics: VA, EX, TS.
- Data: 200-sample test set; distilled variants for TS.
- Logging: save run configs (seed, hyperparams, dataset snapshot hashes) with metrics.
