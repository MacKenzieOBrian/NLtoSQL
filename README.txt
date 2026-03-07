NLtoSQL Dissertation Project - Handover README
==============================================

This covers project structure, code ownership, build/run steps,
and where AI use is documented.


1) Project summary
------------------
- Goal: evaluate NL-to-SQL performance on ClassicModels under constrained hardware.
- Main comparison: base model vs QLoRA, and zero-shot (`k=0`) vs few-shot (`k=3`).
- Primary evaluation mode: `model_only_raw` (no optional cleanup layers).
- Official analysis outputs: descriptive summaries for VA, EX, and TS, plus EX-only pairwise tests.


2) Repository structure
-----------------------
Top-level layout:
- `data/`
  - `classicmodels_test_200.json`: fixed 200-item evaluation set.
  - `train/classicmodels_train_200.jsonl`: QLoRA training set.
- root project documentation:
  - `README.txt`, `REFERENCES.md`, `technical_description.md`.
- `nl2sql/`
  - `core/`: prompt, generation, SQL cleanup, validation, query execution.
  - `evaluation/`: scoring, fixed grid execution, manual-pack loading, and simple statistics.
  - `agent/`: ReAct extension logic.
  - `infra/`: notebook setup, DB/auth helpers, and orchestration wrappers.
- `notebooks/`
  - `02_baseline_prompting_eval.ipynb`: baseline experiments.
  - `03_agentic_eval.ipynb`: ReAct extension path.
  - `04_build_training_set.ipynb`: train-set validation/build workflow.
  - `05_qlora_train_eval.ipynb`: QLoRA training + evaluation.
  - `06_research_comparison.ipynb`: reporting-only notebook for the final CSV outputs.
- `scripts/`
  - `run_baseline_llama.py`, `run_baseline_qwen.py`: fixed baseline campaigns.
  - `run_qlora_llama.py`, `run_qlora_qwen.py`: fixed QLoRA campaigns.
  - `run_react_llama.py`, `run_react_qwen.py`: fixed ReAct campaigns.
  - `build_final_analysis.py`: official manual-pack analysis builder.
  - `colab_setup.sh`: Colab runtime dependency bootstrap.
- `results/`
  - active hypothesis experiment artifacts and analysis outputs.

Primary evidence folders:
- `results/baseline/runs` (base model runs, Llama and Qwen)
- `results/qlora/runs` (QLoRA runs, Llama and Qwen)
- `results/agent/runs` (ReAct extension runs)
- `results/final_pack` (manually selected official JSON files for dissertation analysis)
- `results/final_analysis` (official CSV outputs built from `final_pack`)


3) Code ownership and provenance
--------------------------------
Own code (author-developed in this repository):
- `nl2sql/core/*`, `nl2sql/evaluation/*`, `nl2sql/agent/*`, `nl2sql/infra/*`, and notebook orchestration.
- `scripts/build_final_analysis.py` as the official dissertation analysis script.
- project notebook workflows and wrappers.
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
- `nl2sql/evaluation/final_pack.py`, `nl2sql/evaluation/simple_stats.py`,
  and `scripts/build_final_analysis.py`
  - What this workflow does:
    - reads only the manually selected JSON files placed in `results/final_pack/`.
    - validates their canonical filenames and fixed raw-output policy.
    - builds a manifest table and a flat per-item table.
    - computes per-condition summaries, Wilcoxon tests, and
      BH-FDR adjusted p-values.
    - writes the official CSV outputs under `results/final_analysis/`.
  - AI-assisted (plumbing): manual-pack loader scaffold, pandas reshaping,
    Wilcoxon wrapper, BH-FDR helper, CSV export wiring, and CLI wrapper structure.
  - Human decisions (research logic): canonical filename contract, supported matrix
    (`llama`/`qwen`, `base`/`qlora`/`react`, `k in {0,3}`), raw-only primary policy,
    fixed comparison list, and interpretation of the resulting statistics.

- `scripts/colab_setup.sh`
  - AI-assisted: shell script boilerplate and install block structure.
  - Human decisions: pinned versions and runtime acceptance.

- `scripts/run_baseline_llama.py`, `scripts/run_baseline_qwen.py`,
  `scripts/run_qlora_llama.py`, `scripts/run_qlora_qwen.py`,
  `scripts/run_react_llama.py`, `scripts/run_react_qwen.py`,
  `scripts/build_final_analysis.py`,
  `nl2sql/infra/experiment_helpers.py`, `nl2sql/infra/db.py`,
  `nl2sql/infra/notebook_utils.py`, `nl2sql/infra/model_loading.py`
  - AI-assisted: helper/orchestration boilerplate for notebook setup, run metadata,
    DB/auth prompts, model-loading scaffolds, fixed campaign scripts, and the
    manual-pack analysis script.
  - Human decisions: experiment settings, fixed rerun recipe, and which
    outputs count as the official evidence.

- `notebooks/02_baseline_prompting_eval.ipynb`
  - AI-assisted: repeated run-loop scaffolding and output wiring.
  - Human decisions: seeds, `k` values, and the fixed rerun settings.

- `notebooks/03_agentic_eval.ipynb` (extension/historical path)
  - AI-assisted: repeated evaluation cell structure and summary wiring.
  - Human decisions: agent settings and interpretation of extension outcomes.

- `notebooks/05_qlora_train_eval.ipynb`
  - AI-assisted: training/evaluation loop scaffolding and output wiring.
  - Human decisions: model presets, adapter workflow, and run priorities.

Concrete examples of boilerplate-style AI assistance:
- fixed run-script scaffolding and manual-pack CSV wiring.
- notebook grid-loop scaffolding in the baseline/QLoRA notebooks.
- small helper modules around the final-pack loader and EX-only statistics stage.

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

C) Official rerun order
1. `python scripts/run_baseline_llama.py`
2. `python scripts/run_baseline_qwen.py`
3. `python scripts/run_qlora_llama.py`
4. `python scripts/run_qlora_qwen.py`
5. `python scripts/run_react_llama.py` and/or `python scripts/run_react_qwen.py`
6. manually copy the official JSON files into `results/final_pack/`
7. `python scripts/build_final_analysis.py`
8. `notebooks/06_research_comparison.ipynb` (reporting only)

Runnable notebook mirrors:
- `notebooks/02_baseline_prompting_eval.ipynb`
- `notebooks/03_agentic_eval.ipynb`
- `notebooks/05_qlora_train_eval.ipynb`
- These mirror the fixed scripts for walkthrough/demo use, but they are not the authoritative evidence path.

D) Official analysis outputs
- `results/final_analysis/manifest.csv`
- `results/final_analysis/per_item.csv`
- `results/final_analysis/summary_by_condition.csv`
- `results/final_analysis/pairwise_tests.csv`


7) Reproducibility and run-policy notes
---------------------------------------
- Freeze the source snapshot before reruns:
  1. commit the current code/docs state
  2. record the commit hash in dissertation notes
  3. confirm `git status --short` shows no source changes before the rerun starts
- The fixed dissertation rerun matrix is the fixed script set listed in Section 6C.
- The notebooks remain as runnable mirrors of those same settings.
- Persist the generated JSON artifacts from those fixed reruns.
- Manually copy only the official JSON files into `results/final_pack/`.
- The simplest mental model is:
  1. fixed scripts save raw run JSON files
  2. `final_pack/` holds the exact files cited in the dissertation
  3. `build_final_analysis.py` turns those files into the official CSV tables
- After rebuilding the analysis, inspect `results/final_analysis/manifest.csv`
  and the summary/test CSVs before writing claims.
- Do not mix extension/non-primary runs into the primary baseline/QLoRA matrix.
- The fixed rerun recipe uses TS for `k=3` and for the full ReAct run, so the
  perturbed TS databases must be available.
- Full rerun and write-up checklists live in:
  - `DISSERTATION_RERUN_PROTOCOL.md`
  - `DISSERTATION_EVALUATION_TEMPLATE.md`


8) Operational and security notes
---------------------------------
- Do not hardcode DB credentials or tokens in notebooks/files.
- Notebooks are designed to prompt via input/getpass when env vars are absent.
- SQL execution includes safety controls in runtime code; still use least-privilege DB users.


9) Where to start
-----------------
If picking this up fresh, read:
- this file (`README.txt`)
- `REFERENCES.md` (literature and method anchors)
- `technical_description.md` (architecture and implementation narrative)
