# Notes and Decision Rationale (kept casual but clear)

## Docs + Git
- Commit in small chunks: scaffolds, data prep scripts, baseline few-shot, QLoRA config, training runs, eval. Push to `nl2sql` when ready; leave `origin` alone.
- Docs split: ARCHITECTURE (ReAct/agent/tools/flow), CONFIG (env + deps + QLoRA params + GPU/VRAM), DATA (train/test/distilled provenance), LOGBOOK (day-to-day).

## Baseline vs Fine-Tune
- Start with a few-shot baseline (schema + table blurbs + 2–4 exemplars) like Ojuri, so I can show why fine-tuning wins.
- Main path is QLoRA (4-bit + LoRA adapters) for feasibility and reproducibility.

## Agent Shape
- ReAct loop: Thought → Action (SQL via QueryRunner) → Observation → refine. QueryRunner stays the tool with read-only guardrails.
- Log traces so I can show how the agent reasons and fixes itself.

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
