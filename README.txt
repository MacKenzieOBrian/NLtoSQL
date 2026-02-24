NLtoSQL Dissertation Project (Model-Only Evaluation)

Overview
- This repository runs an NL-to-SQL study on constrained hardware.
- The main hypothesis compares base vs QLoRA models and zero-shot (k=0) vs few-shot (k=3).
- Evaluation is model-only raw output scoring (VA/EM/EX, optional TS), without SQL cleaning layers.
- Constrained decoding and reliability cleanup layers are available as optional extension toggles in the eval notebooks.

Repository Layout
- `nl2sql/` : core prompt/model/db/evaluation modules.
- `notebooks/` : dataflow demo, baseline runs, QLoRA train/eval, and stats notebook.
- `scripts/` : analysis script for supervisor-required statistics.
- `results/` : run artifacts and analysis outputs.

Quick Start
1. Install dependencies from `requirements.txt`.
2. Configure database environment variables for the notebooks.
3. Run notebooks in order:
   - `notebooks/01_no_model_dataflow_demo.ipynb`
   - `notebooks/02_baseline_prompting_eval.ipynb`
   - `notebooks/05_qlora_train_eval.ipynb`
   - `notebooks/06_research_comparison.ipynb`
4. Generate stats files from per-item metrics:
   - `python scripts/generate_research_comparison.py`

Primary Outputs
- `results/analysis/stats_mean_median_shapiro.csv`
- `results/analysis/stats_paired_ttests.csv`
