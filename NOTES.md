# Notes and Decision Rationale 

## Docs + Git
- Commit in small chunks: scaffolds, data prep scripts, baseline few-shot, QLoRA config, training runs, eval. Push to `nl2sql` when ready; leave `origin` alone.
- Docs split: ARCHITECTURE (ReAct/agent/tools/flow), CONFIG (env + deps + QLoRA params + GPU/VRAM), DATA (train/test/distilled provenance), LOGBOOK (day-to-day).
- VS Code env: set `terminal.integrated.env.osx` with DB creds + project so notebooks run locally without prompts; use ADC (`gcloud auth application-default login`) or service account key.

## Baseline vs Fine-Tune
- Start with a few-shot baseline (schema + table blurbs + 2–4 exemplars) like Ojuri, so I can show why fine-tuning wins.
- Main path is QLoRA (4-bit + LoRA adapters) for feasibility and reproducibility.

## Agent Shape
- ReAct loop: Thought → Action (SQL via QueryRunner) → Observation → refine. QueryRunner stays the tool with read-only guardrails.
- Log traces so I can show how the agent reasons and fixes itself.
- QueryRunner rationale: read-only executor with guardrails and metadata is the “Act” tool; it turns LLM SQL into DB calls safely and produces the Observation (errors/row counts/previews/history) needed for ReAct self-correction and evaluation.
- Colab sync tips: always git pull before installs; set env vars/ADC/quota project; if NumPy binary mismatch appears, force reinstall once and restart. IAM needed: Cloud SQL Client + Service Usage on the project.

## Data
- Classicmodels only. Need a 200-sample test set plus a bigger train set covering joins/aggregations/filters. Keep a schema cache JSON for prompts.
- If I generate synthetic pairs with a bigger model, I’ll note prompts/filters in DATA.md.

## Evaluation
- Metrics: VA (syntax), EX (correctness on live classicmodels), TS (distilled DB consistency). Aim EX > 80, TS > 70.
- Distilled DBs: schema-identical with varied data for TS.

## Security/Ops
- Secrets via env/prompt for now; move to Secret Manager later. Tools stay read-only.
- Log GPU model, VRAM peak/avg, and runtimes for QLoRA runs in CONFIG + LOGBOOK.

## Artifacts
- Save screenshots/figures when I demo baselines, training curves, and evals; diagrams where helpful.

## Papers & Justifications 
- **Few-shot baseline first** — Following Brown et al. (2020) and Mosbach et al. (2023), few-shot prompting is treated as an inference-time baseline that does not modify model parameters. Ojuri et al. (2025) argue prompt-based baselines are necessary to quantify the true contribution of agents or fine-tuning. Consequently, this project implements a schema-grounded few-shot NL→SQL baseline prior to any parameter-efficient fine-tuning (QLoRA).  
- **Schema grounding** — Prior work shows schema grounding is critical for Text-to-SQL (Li et al., 2023; Zhu et al., 2024; Hong et al., 2025; RESDSQL). The schema is dynamically extracted from INFORMATION_SCHEMA and embedded directly in the prompt; columns are ordered to prioritise identifiers/PKs as a lightweight inductive bias without changing model parameters.  
- **Deterministic decoding for VA/EX** — For VA/EX evaluation, deterministic decoding (`do_sample=False`) removes sampling noise so differences reflect prompt/model changes, consistent with benchmark practice (Zhong et al., 2020; Gao et al., 2025; Ojuri et al., 2025).  
- **Post-processing (SQL extraction + minimal projection)** — Constrained/execution-aware handling is standard (PICARD; ExCoT; Ojuri et al., 2025). Outputs are reduced to a single `SELECT ...;` and minimal projection for list-style queries to ensure executability and comparability; this does not alter model weights.  
- **Execution harness** — Following Spider and VA/EX/TS practice (Yu et al., 2018; Zhong et al., 2020; Ojuri et al., 2025), generated queries are executed against a live DB via QueryRunner, which logs success/error/metadata for reproducible evaluation.  
- **4-bit + QLoRA feasibility** — Parameter-efficient methods (Ding et al., 2023; Goswami et al., 2024) show 4-bit NF4 + QLoRA retain performance while fitting academic GPUs; this keeps baselines and later fine-tuning on the same stack.  
- **Agent readiness** — The current harness prepares for ReAct-style agents (Yao et al., 2023; Xi et al., 2025); QueryRunner is the “Act” tool, enabling iterative reasoning later.  
- **Canonical sentence to reuse** — “In this project, ‘few-shot learning’ refers exclusively to inference-time prompt conditioning using exemplar NLQ–SQL pairs; the underlying language model parameters remain fixed throughout all baseline experiments (Brown et al., 2020; Mosbach et al., 2023).”
