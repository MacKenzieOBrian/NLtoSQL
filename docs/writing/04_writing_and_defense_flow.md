# Writing and Defense Flow

The most effective writing flow is an evidence-led progression from baseline control to QLoRA adaptation, ending in statistical synthesis. This keeps the narrative coherent for examiners and directly aligned to the final hypothesis.

For defense and demo settings, anchor explanations to current module boundaries and outputs:
- generation and extraction behavior: `nl2sql/core/llm.py`
- prompt construction: `nl2sql/core/prompting.py`
- safe execution checks: `nl2sql/core/query_runner.py`
- evaluation contract: `nl2sql/evaluation/eval.py`
- run orchestration: `notebooks/02_baseline_prompting_eval.ipynb` and `notebooks/05_qlora_train_eval.ipynb`
- statistics outputs: `results/analysis/stats_mean_median_shapiro.csv` and `results/analysis/stats_paired_ttests.csv`

A practical writing rule is to keep each claim paragraph self-contained: setup, metric change, statistical result, interpretation, and artifact path. This makes your argument auditable.

AI-assistance disclosure should stay concise: identify where boilerplate/scaffolding help was used, and state that method design, scope changes, evaluation choices, and final claims were researcher-directed.

Final defense posture: this project does not claim proprietary parity or universal cross-domain generalization. It claims a reproducible open-source pathway with controlled comparisons and statistically grounded interpretation under constrained hardware.
