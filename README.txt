NLtoSQL Dissertation Project
============================

This archive is the final project handover. It explains the repository layout,
ownership/provenance of the code, how AI use was documented, and how to run the
official experiment and analysis workflow.


1) Project summary
------------------
- Goal: evaluate NL-to-SQL performance on the ClassicModels database under
  constrained hardware.
- Main comparisons:
  - base model vs QLoRA
  - zero-shot (`k=0`) vs few-shot (`k=3`)
  - optional ReAct extension
- Official evidence path:
  1. run the fixed scripts in `scripts/`
  2. manually copy the chosen JSON outputs into `results/final_pack/`
  3. run `python scripts/build_final_analysis.py`
  4. inspect `results/final_analysis/*.csv`
- Official baseline/QLoRA seed policy:
  - `k=0` -> `[7]` only, because this condition is deterministic
  - `k=3` -> `[7, 17, 27, 37, 47]`
  - `TS` remains enabled only for `k=3`


2) Repository structure
-----------------------
- `data/`
  - fixed evaluation and training data
- `nl2sql/core/`
  - prompt construction, generation, validation, safe execution
- `nl2sql/agent/`
  - local ReAct loop and agent prompts
- `nl2sql/evaluation/`
  - scoring, fixed grid execution, manual-pack loading, simple statistics
- `nl2sql/infra/`
  - DB/model/notebook support code
- `notebooks/`
  - runnable mirrors and support material, not the official evidence path
- `scripts/`
  - fixed rerun entrypoints and final analysis builder
- `results/final_pack/`
  - manually selected official JSON evidence files
- `results/final_analysis/`
  - official CSV outputs built from `final_pack`

Official submission surface:
- `nl2sql/`
- `scripts/`
- `README.txt`
- `technical_description.md`
- `results/final_pack/`
- `results/final_analysis/`

Support material that is useful to inspect but not needed to justify the final claims:
- `notebooks/`
- `notebooks/01_demo.ipynb`
- `diagrams.md`
- `writing_code_alignment_report.md`


3) Code ownership and provenance
--------------------------------
Author-owned core research logic:
- `nl2sql/core/*`
- `nl2sql/agent/*`
- `nl2sql/evaluation/eval.py`
- `nl2sql/evaluation/grid_runner.py`
- experiment design, metric definitions (`VA`, `EM`, `EX`, `TS`), run policy,
  and interpretation of results

Author-directed support/orchestration code:
- `nl2sql/infra/*`
- `nl2sql/evaluation/final_pack.py`
- `nl2sql/evaluation/simple_stats.py`
- fixed scripts in `scripts/`
- notebook orchestration in `notebooks/02`, `03`, `05`, and reporting notebook `06`

External dependencies/services:
- libraries in `requirements.txt`, including `transformers`, `peft`,
  `bitsandbytes`, `trl`, `sqlalchemy`, `cloud-sql-python-connector`,
  `pandas`, `numpy`, and `scipy`
- model weights/tokenizers downloaded at runtime from Hugging Face hubs
- official library/platform documentation used as implementation references,
  especially for Transformers, SQLAlchemy, Python standard-library features,
  and the Google Cloud SQL connector

Third-party or supervisor source code:
- no standalone third-party source files are vendored into this repository
- no supervisor-provided source modules are included directly in the archive


4) AI use documentation
-----------------------
This README is the primary record of generative AI use for the submitted archive.

AI was used mainly for:
- scaffolding project structure and helper-module layout
- refactoring repeated notebook logic into local support files
- generating boilerplate for scripts, wrappers, and documentation drafts
- small-scale typing assistance through editor/tab completion in some areas

AI was not used to decide:
- the research questions or dissertation objectives
- the benchmark scope and comparison structure
- the evaluation metrics or final run-selection policy
- the interpretation of results or dissertation conclusions

Areas with AI-assisted scaffolding or drafting:
- `nl2sql/infra/*`
- `scripts/*.py`
- `nl2sql/evaluation/final_pack.py`
- `nl2sql/evaluation/simple_stats.py`
- `scripts/build_final_analysis.py`
- notebook orchestration and project documentation

Core research logic remained author-led:
- `nl2sql/core/*`
- `nl2sql/agent/*`
- `nl2sql/evaluation/eval.py`
- `nl2sql/evaluation/grid_runner.py`
- experimental design, run policy, and interpretation


5) Build and run
----------------
Prerequisites:
- Python 3.10+
- access to a ClassicModels MySQL database
- Hugging Face token where required
- CUDA-capable environment for QLoRA training (typically Colab GPU)

Environment variables used by notebooks:
- `INSTANCE_CONNECTION_NAME`
- `DB_USER`
- `DB_PASS`
- `DB_NAME` (default `classicmodels`)
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

Local setup:
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

Colab setup (recommended for QLoRA):
1. `bash scripts/colab_setup.sh`
2. restart the runtime once if prompted

Official rerun order:
1. `python scripts/run_baseline_llama.py`
2. `python scripts/run_baseline_qwen.py`
3. `python scripts/run_qlora_llama.py`
4. `python scripts/run_qlora_qwen.py`
5. `python scripts/run_react_llama.py` and/or `python scripts/run_react_qwen.py`
6. manually copy the official JSON files into `results/final_pack/`
7. `python scripts/build_final_analysis.py`
8. open `notebooks/06_research_comparison.ipynb` for reporting only

Official baseline/QLoRA run counts:
- one canonical `k=0` JSON per condition
- multiple `k=3` JSON files per stochastic condition; the target seed set is `7, 17, 27, 37, 47`
- ReAct remains one fixed descriptive run at `k=3`, seed `7`

Official analysis outputs:
- `results/final_analysis/manifest.csv`
- `results/final_analysis/per_item.csv`
- `results/final_analysis/summary_by_condition.csv`
- `results/final_analysis/pairwise_tests.csv`
- `pairwise_tests.csv` contains only the two formal `k=3` baseline-vs-QLoRA `EX` comparisons

Runnable notebook mirrors:
- `notebooks/02_baseline_prompting_eval.ipynb`
- `notebooks/03_agentic_eval.ipynb`
- `notebooks/05_qlora_train_eval.ipynb`
- these notebooks are convenience wrappers and checks, not the official evidence path


6) Reproducibility and handover notes
-------------------------------------
- commit the code/docs snapshot before reruns
- confirm `git status --short` is clean before starting official runs
- keep only the official cited JSON files in `results/final_pack/`
- inspect `results/final_analysis/manifest.csv` before writing claims
- use least-privilege DB credentials and do not hardcode secrets

Useful companion documents:
- `REFERENCES.md`
- `technical_description.md`
