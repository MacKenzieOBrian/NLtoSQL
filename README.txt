NLtoSQL Dissertation Project - Handover README
==============================================

This covers project structure, code ownership, build/run steps,
and where AI use is documented.


1) Project summary
------------------
- Goal: evaluate NL-to-SQL performance on ClassicModels under constrained hardware.
- Main comparison: base model vs QLoRA, and zero-shot (`k=0`) vs few-shot (`k=3`).
- Primary evaluation mode: `model_only_raw` (no optional cleanup layers).
- Primary metrics: VA, EM, EX, and optional TS when enabled.


2) Repository structure
-----------------------
Top-level layout:
- `data/`
  - `classicmodels_test_200.json`: fixed 200-item evaluation set.
  - `train/classicmodels_train_200.jsonl`: QLoRA training set.
- root project documentation:
  - `README.txt`, `REFERENCES.md`, `documentation.md`.
- `nl2sql/`
  - `core/`: prompt, generation, SQL cleanup, validation, query execution.
  - `evaluation/`: scoring, run discovery, comparison statistics, analysis helpers.
  - `agent/`: ReAct extension logic.
  - `infra/`: notebook setup, DB/auth helpers, and orchestration wrappers.
- `notebooks/`
  - `02_baseline_prompting_eval.ipynb`: baseline experiments.
  - `03_agentic_eval.ipynb`: extension path (agentic/reliability).
  - `04_build_training_set.ipynb`: train-set validation/build workflow.
  - `05_qlora_train_eval.ipynb`: QLoRA training + evaluation.
  - `06_research_comparison.ipynb`: analysis notebook wrapper.
- `scripts/`
  - `collect_research_tables.py`: flatten saved run JSON files into raw tables.
  - `format_research_outputs.py`: turn the raw per-item table into stats outputs.
  - `generate_research_comparison.py`: all-in-one wrapper for the two-stage analysis path.
  - `colab_setup.sh`: Colab runtime dependency bootstrap.
- `results/`
  - active hypothesis experiment artifacts and analysis outputs.
  - see `results/README.txt` for the kept primary-result subset.

Primary evidence folders:
- `results/baseline/runs` (base model runs, Llama and Qwen)
- `results/qlora/runs` (QLoRA runs, Llama and Qwen)
- `results/agent/runs` (ReAct extension runs)
- `results/analysis` (computed CSVs from the research comparison workflow)


3) Code ownership and provenance
--------------------------------
Own code (author-developed in this repository):
- `nl2sql/core/*`, `nl2sql/evaluation/*`, `nl2sql/agent/*`, `nl2sql/infra/*`, and notebook orchestration.
- `scripts/generate_research_comparison.py` as a CLI wrapper.
- project notebook workflows and run plans.
- result management and reporting scripts/files.

Not own code (external dependencies/services):
- Python libraries in `requirements.txt`, including but not limited to:
  - `transformers`, `peft`, `bitsandbytes`, `accelerate`, `trl`, `datasets`
  - `sqlalchemy`, `cloud-sql-python-connector`, `pymysql`
  - `pandas`, `numpy`, `scipy`
- model weights/tokenizers downloaded at runtime from Hugging Face model hubs
  (for example Llama/Qwen checkpoints), not redistributed in this archive.
- Cloud SQL connector and SQLAlchemy integration patterns are based on official
  documentation; implementation is adapted into project code in `nl2sql/infra/db.py`.

Third-party or supervisor source code:
- No standalone third-party source files are vendored directly into this repository.
- No supervisor-provided code files are included as direct source modules.

Compatibility wrappers:
- None are required in the current simplified layout.
- The active code paths live under `nl2sql/core`, `nl2sql/agent`, `nl2sql/evaluation`, and `nl2sql/infra`.


4) AI use documentation
-----------------------
AI was used for implementation scaffolding, boilerplate, and documentation
drafting. Research decisions — hypothesis design, experiment scope, run
selection, result interpretation, and dissertation conclusions — are my own.

File-level disclosure:
- `nl2sql/evaluation/research_runs.py`, `nl2sql/evaluation/research_stats.py`,
  `nl2sql/evaluation/research_comparison.py`, and `scripts/generate_research_comparison.py`
  - What this workflow does:
    - discovers run JSON files under `results/**/results_k*_seed*.json` and ReAct runs
      under `results/agent/runs/**/results_react_200.json` or `results_react_eval.json`.
    - filters primary baseline/QLoRA runs to one explicit `primary_eval_profile`
      (default: `model_only_raw`) so analysis does not silently mix raw and reliability-layer runs.
    - normalizes run metadata into `condition_id` (`llama|qwen` x `base|qlora|react` x `k=0|3`).
    - builds per-item analysis rows and run manifest rows.
    - computes mean/median/std and paired-comparison statistics.
    - applies Benjamini-Hochberg FDR correction within each metric family.
    - writes analysis CSV outputs under `results/analysis/`.
  - AI-assisted (plumbing): CLI scaffold, JSON/file ingestion, metadata normalization,
    deduplication, pandas joins/grouping, Wilcoxon/t-test/CI/Cohen's d wrappers,
    BH FDR helper, and CSV export wiring.
  - Human decisions (research logic): source-of-truth run folder, supported matrix
    (`llama`/`qwen`, `base`/`qlora`/`react`, `k in {0,3}`), primary eval profile
    (`model_only_raw` for dissertation claims), predefined hypothesis comparisons,
    choice of primary test (Wilcoxon), and interpretation of statistical results.

- `scripts/colab_setup.sh`
  - AI-assisted: shell script boilerplate and install block structure.
  - Human decisions: pinned versions and runtime acceptance.

- `scripts/collect_research_tables.py`, `scripts/format_research_outputs.py`,
  `scripts/generate_research_comparison.py`,
  `nl2sql/infra/experiment_helpers.py`, `nl2sql/infra/db.py`,
  `nl2sql/infra/notebook_utils.py`, `nl2sql/infra/model_loading.py`
  - AI-assisted: helper/orchestration boilerplate for notebook setup, run metadata,
    DB/auth prompts, model-loading scaffolds, shared eval wrappers, and the
    two-stage analysis scripts.
  - Human decisions: experiment options, profile choices, run plans, and which
    outputs count as the raw evidence versus the formatted analysis.

- `notebooks/02_baseline_prompting_eval.ipynb`
  - AI-assisted: repeated run-loop scaffolding and output wiring.
  - Human decisions: seeds, `k` values, run plans, and ablation settings.

- `notebooks/03_agentic_eval.ipynb` (extension/historical path)
  - AI-assisted: repeated evaluation cell structure and summary wiring.
  - Human decisions: agent settings and interpretation of extension outcomes.

- `notebooks/05_qlora_train_eval.ipynb`
  - AI-assisted: training/evaluation loop scaffolding and output wiring.
  - Human decisions: model presets, adapter workflow, and run priorities.

- `nl2sql/validation.py` and `nl2sql/*.py` wrappers
  - AI-assisted: wrapper/re-export boilerplate.
  - Human decisions: validation policy location and compatibility strategy.

Concrete examples of boilerplate-style AI assistance:
- research comparison plumbing:
  `discover_primary_runs`, `discover_react_runs`, `build_tables_from_runs`,
  `compute_mean_median_std`, `_bh_fdr_adjust`, `compute_paired_tests`,
  `generate`, `main`.
- notebook run-plan and grid-loop cells in baseline/QLoRA notebooks.
- small helper modules and script wrappers around the raw-table and stats-table stages.

Not AI-owned outcomes:
- research framing and final evidence claims.
- final run inclusion/exclusion policy for primary vs extension results.
- dissertation narrative and conclusions.


5) Build prerequisites
----------------------
- Python 3.10+ (3.12 used in recent runs).
- Access to a MySQL ClassicModels database (Cloud SQL in notebook defaults).
- Hugging Face token for gated model access when required.
- For QLoRA training: CUDA-capable environment (typically Colab GPU).

Environment variables used by notebooks:
- `INSTANCE_CONNECTION_NAME`
- `DB_USER`
- `DB_PASS`
- `DB_NAME` (default `classicmodels`)
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`


6) Build and run instructions
-----------------------------
A) Local setup
1. Create environment:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`

B) Colab setup (recommended for QLoRA)
1. Run:
   - `bash scripts/colab_setup.sh`
2. Restart runtime once when prompted by the script output.

C) Notebook execution order
1. `notebooks/02_baseline_prompting_eval.ipynb` (base-model experiments)
2. `notebooks/03_agentic_eval.ipynb` (agentic evaluation)
3. `notebooks/04_build_training_set.ipynb` (if rebuilding training data)
4. `notebooks/05_qlora_train_eval.ipynb` (QLoRA train/eval)
5. `notebooks/06_research_comparison.ipynb` (analysis wrapper)

D) Scripted analysis build
Run from repo root:
- Stage 1 (raw tables): `python scripts/collect_research_tables.py --primary-eval-profile model_only_raw`
- Stage 2 (stats tables): `python scripts/format_research_outputs.py`
- Optional all-in-one wrapper: `python scripts/generate_research_comparison.py --primary-eval-profile model_only_raw`

Outputs:
- `results/analysis/per_item_metrics_primary_raw.csv`
- `results/analysis/run_manifest.csv`
- `results/analysis/stats_mean_median_std.csv`
- `results/analysis/stats_paired_ttests.csv`


7) Reproducibility and run-policy notes
---------------------------------------
- Primary dissertation claims should use `model_only_raw` runs only.
- If rerunning baseline or QLoRA notebooks, keep the notebook grid plan set to the
  default raw-profile configuration and regenerate the analysis tables with
  `--primary-eval-profile model_only_raw`.
- Keep notebook run plans explicit and persist the generated JSON artifacts.
- The simplest mental model is:
  1. notebooks save raw run JSON files
  2. stage 1 scripts flatten them into raw tables
  3. stage 2 scripts format those tables into hypothesis/statistics outputs
- Do not mix extension/non-primary runs into primary hypothesis statistics.
- When TS is required, ensure `enable_ts=True` and that TS databases are available.


8) Operational and security notes
---------------------------------
- Do not hardcode DB credentials or tokens in notebooks/files.
- Notebooks are designed to prompt via input/getpass when env vars are absent.
- SQL execution includes safety controls in runtime code; still use least-privilege DB users.


9) Where to start
-----------------
If picking this up fresh, read:
- this file (`README.txt`)
- `results/README.txt` (artifact folder policy)
- `REFERENCES.md` (literature and method anchors)
