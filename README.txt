NLtoSQL Dissertation Project - Handover README
==============================================

This README is written as a handover document for another developer.
It explains project structure, code ownership/provenance, build/run steps,
and where AI use is documented.


1) Project summary
------------------
- Goal: evaluate NL-to-SQL performance on ClassicModels under constrained hardware.
- Main comparison: base model vs QLoRA, and zero-shot (`k=0`) vs few-shot (`k=3`).
- Primary evaluation mode: model-only outputs (no optional cleanup layers).
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
  - canonical runtime/evaluation code (`core/`, `evaluation/`).
  - compatibility wrappers at package root preserve legacy import paths.
- `notebooks/`
  - `01_no_model_dataflow_demo.ipynb`: pipeline walkthrough/demo.
  - `02_baseline_prompting_eval.ipynb`: baseline experiments.
  - `03_agentic_eval.ipynb`: extension path (agentic/reliability).
  - `04_build_training_set.ipynb`: train-set validation/build workflow.
  - `05_qlora_train_eval.ipynb`: QLoRA training + evaluation.
  - `06_research_comparison.ipynb`: analysis notebook wrapper.
- `scripts/`
  - `generate_research_comparison.py`: statistical outputs used in reporting.
  - `colab_setup.sh`: Colab runtime dependency bootstrap.
- `results/`
  - active hypothesis experiment artifacts and analysis outputs.
  - see `results/README.txt` for the kept primary-result subset.

Primary evidence folders:
- `results/baseline/runs`
- `results/analysis`
- `results/qlora/runs` (staging area only; selected QLoRA runs are copied into
  `results/baseline/runs` for unified analysis)


3) Code ownership and provenance
--------------------------------
Own code (author-developed in this repository):
- `nl2sql/core/*`, `nl2sql/evaluation/*` and orchestration logic in notebooks.
- `scripts/generate_research_comparison.py`.
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
  documentation; implementation is adapted into project code in `nl2sql/core/db.py`.

Third-party or supervisor source code:
- No standalone third-party source files are vendored directly into this repository.
- No supervisor-provided code files are included as direct source modules.

Compatibility wrappers:
- `nl2sql/*.py` wrapper modules re-export canonical modules to keep older imports working.
- These wrappers are project-maintained glue code, not external dependencies.


4) AI use documentation (policy record)
---------------------------------------
This section is the official AI-use record for this repository.

Scope boundary:
- AI was used for implementation scaffolding, boilerplate, refactoring support,
  and documentation drafting.
- Human-owned decisions: research question, hypothesis design, experiment scope,
  run selection, acceptance criteria, interpretation, and final dissertation claims.

File-level disclosure:
- `scripts/generate_research_comparison.py`
  - What the script does:
    - discovers run JSON files under `results/baseline/runs/**/results_k*_seed*.json`.
    - normalizes run metadata into `condition_id` (`llama|qwen` x `base|qlora` x `k=0|3`).
    - builds per-item analysis rows and run manifest rows.
    - computes mean/median, Shapiro-Wilk checks, and paired t-tests on predefined contrasts.
    - writes analysis CSV outputs under `results/analysis/`.
  - AI-assisted (plumbing): CLI scaffold, JSON/file ingestion, metadata normalization,
    deduplication, pandas joins/grouping, and CSV export wiring.
  - Human decisions (research logic): source-of-truth run folder, supported matrix
    (`llama`/`qwen`, `base`/`qlora`, `k in {0,3}`), inclusion policy
    (`model_only_raw` by default), predefined hypothesis comparisons, and interpretation
    of statistical results in the dissertation.

- `scripts/colab_setup.sh`
  - AI-assisted: shell script boilerplate and install block structure.
  - Human decisions: pinned versions and runtime acceptance.

- `nl2sql/experiment_helpers.py`
  - AI-assisted: helper structure for model aliases and option handling.
  - Human decisions: allowed experiment options and how they are used.

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
- `generate_research_comparison.py`:
  `parse_args`, `discover_runs`, `build_tables_from_runs`, `_join_for_pair`,
  `compute_mean_median_shapiro`, `compute_paired_ttests`, `generate`, `main`.
- notebook run-plan and grid-loop cells in baseline/QLoRA notebooks.
- simple compatibility wrappers (`from ... import *`) for legacy import paths.

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
1. `notebooks/01_no_model_dataflow_demo.ipynb` (optional demo/traceability)
2. `notebooks/02_baseline_prompting_eval.ipynb` (base-model experiments)
3. `notebooks/04_build_training_set.ipynb` (if rebuilding training data)
4. `notebooks/05_qlora_train_eval.ipynb` (QLoRA train/eval)
5. `notebooks/06_research_comparison.ipynb` (analysis wrapper)

D) Scripted analysis build
Run from repo root:
- `python scripts/generate_research_comparison.py`
- Optional: `python scripts/generate_research_comparison.py --runs-root results/baseline/runs`

Outputs:
- `results/analysis/per_item_metrics_primary_raw.csv`
- `results/analysis/run_manifest.csv`
- `results/analysis/stats_mean_median_shapiro.csv`
- `results/analysis/stats_paired_shapiro.csv`
- `results/analysis/stats_paired_ttests.csv`


7) Reproducibility and run-policy notes
---------------------------------------
- Primary dissertation claims should use model-only runs (optional reliability flags off).
- Keep run plans explicit (`RUN_PLAN`, `CUSTOM_PLAN`) and persist generated JSON artifacts.
- Do not mix extension/non-primary runs into primary hypothesis statistics.
- When TS is required, ensure `enable_ts=True` and that TS databases are available.


8) Operational and security notes
---------------------------------
- Do not hardcode DB credentials or tokens in notebooks/files.
- Notebooks are designed to prompt via input/getpass when env vars are absent.
- SQL execution includes safety controls in runtime code; still use least-privilege DB users.


9) Contact surface for maintainers
----------------------------------
If handing over to another developer, first review:
- this file (`README.txt`)
- `results/README.txt` (artifact folder policy)
- `REFERENCES.md` (literature and method anchors)
