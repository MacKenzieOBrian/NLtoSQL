NLtoSQL Dissertation Project (Open-Source, Reproducible)

Overview
- This repository contains a research workflow for NL-to-SQL under constrained hardware.
- The core comparison evaluates prompting, QLoRA adaptation, and execution-guided infrastructure.
- Experiments are run through notebooks and reproducible JSON artifacts in `results/`.

Repository Layout
- `nl2sql/` : core pipeline, evaluation, guardrails, and tool interfaces.
- `notebooks/` : experiment execution notebooks (baseline, QLoRA, ReAct, comparison).
- `scripts/` : analysis/report generation helpers.
- `results/` : run artifacts and analysis outputs used for reporting.

Branching Model
- `master` : active experimental branch (lab workflow, iterative updates).
- `codex/release-order` : curated release-order branch with a cleaner research narrative.

Quick Start
1. Install dependencies from `requirements.txt`.
2. Configure DB environment variables used by the notebooks.
3. Run notebooks in order:
   - `notebooks/02_baseline_prompting_eval.ipynb`
   - `notebooks/05_qlora_train_eval.ipynb`
   - `notebooks/03_agentic_eval.ipynb`
   - `notebooks/06_research_comparison.ipynb`
4. Generate analysis artifacts with:
   - `python scripts/generate_research_comparison.py`

Primary Outputs
- `results/analysis/overall_metrics_wide.csv`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `results/analysis/per_item_metrics.csv`

