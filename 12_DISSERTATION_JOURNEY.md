# Dissertation Journey Guide

This file connects the full project journey from research framing to final write-up.

## Journey Overview

| Phase | Core question | Main artifacts |
| --- | --- | --- |
| Planning and scope | What is the contribution boundary? | `LOGBOOK.md`, `11_REPLICATION_POSITIONING.md` |
| Evaluation design | How will improvements be measured fairly? | `4_EVALUATION.md`, `5_RESEARCH_GROUNDING_MAP.md` |
| Baseline experiments | What does prompting alone achieve? | `notebooks/02_baseline_prompting_eval.ipynb`, `results/baseline/` |
| QLoRA experiments | What does lightweight adaptation add? | `notebooks/05_qlora_train_eval.ipynb`, `results/qlora/` |
| Agentic infrastructure | What does execution guidance add operationally? | `notebooks/03_agentic_eval.ipynb`, `results/agent/` |
| Comparative synthesis | Which differences are statistically defensible? | `notebooks/06_research_comparison.ipynb`, `results/analysis/` |

## Writing Workflow (Chapter Draft Order)

1. Start with contribution boundary:
   - Use `11_REPLICATION_POSITIONING.md` to draft replication stance and scope limits.
2. Draft methodology and evaluation:
   - Use `4_EVALUATION.md` and `5_RESEARCH_GROUNDING_MAP.md` to justify metrics and inference policy.
3. Draft results sections by comparison axis:
   - Prompting effect (`k=0` vs `k>0`)
   - Fine-tuning effect (base vs QLoRA)
   - Agentic infrastructure effect (ReAct as support)
4. Add statistical validity:
   - Pull intervals and paired significance from `results/analysis/paired_deltas.csv`.
5. Add error mechanism interpretation:
   - Use `results/analysis/failure_taxonomy.csv`.
6. Finalize conclusion and limitations:
   - Reuse scope guardrails from `6_LIMITATIONS.md` and `11_REPLICATION_POSITIONING.md`.

## Minimal Evidence Bundle for Submission

- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/run_manifest.csv`
- `LOGBOOK.md`
- `EXAMINER_QA.md`

## Demo Walkthrough (Concise)

1. Problem and research boundary (open-source replication under constraints).
2. Method comparison setup (baseline vs QLoRA vs ReAct infrastructure).
3. Controlled-run evidence (same dataset, same evaluator, paired tests).
4. Key metric deltas and significance.
5. Failure-taxonomy explanation of residual errors.
6. Final contribution statement and limits.
