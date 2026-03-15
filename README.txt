NLtoSQL Dissertation Archive
============================

This archive contains the code, notebooks, and result files for the NL-to-SQL
dissertation project. The project compares prompting, QLoRA, and a bounded
ReAct-style extension on the ClassicModels database.


1) Structure
------------
- `nl2sql/`
  - main package
- `scripts/`
  - official rerun entrypoints
- `notebooks/`
  - development and notebook mirrors
- `results/final_pack/`
  - official JSON evidence files
- `results/final_analysis/`
  - final CSV outputs built from `final_pack`


2) Main code vs helper code
---------------------------
Main project code:
- `nl2sql/core/*`
- `nl2sql/agent/*`
- `nl2sql/evaluation/eval.py`
- `nl2sql/evaluation/grid_runner.py`

Helper / analysis / menial-task code:
- `scripts/build_final_analysis.py`
- `nl2sql/evaluation/final_pack.py`
- `nl2sql/evaluation/simple_stats.py`
- `nl2sql/infra/*`
- `notebooks/04_build_training_set.ipynb`
- `notebooks/06_research_comparison.ipynb`

Notebook experiment surfaces:
- `notebooks/02_baseline_prompting_eval.ipynb`
- `notebooks/03_agentic_eval.ipynb`
- `notebooks/05_qlora_train_eval.ipynb`

The notebooks were used heavily during development, but the `scripts/` folder
is the official rerun surface for the final archive.


3) Ownership and provenance
---------------------------
The main research logic is my own work. That includes the prompt design, schema
summary approach, guarded SQL execution path, ReAct-style loop, evaluation
logic, and the overall experiment design.

Some support code is more scaffolding-heavy. That mostly applies to the helper
scripts, analysis scripts, formatting utilities, and some notebook orchestration
code listed above.

There is no supervisor source code copied into this repo, and there are no
vendored third-party source modules. External code comes through libraries in
`requirements.txt`, model weights downloaded at runtime, and the GitHub repos
and docs listed below that influenced the implementation.


4) Influenced by
----------------
- `ysymyth/ReAct` - influenced the Thought / Action / Observation loop shape.
  https://github.com/ysymyth/ReAct
- `huggingface/transformers` - influenced chat templating and generation usage.
  https://github.com/huggingface/transformers
- `huggingface/peft` - influenced the LoRA / QLoRA adapter workflow.
  https://github.com/huggingface/peft
- `taoyds/spider` - influenced the evaluation framing for text-to-SQL.
  https://github.com/taoyds/spider
- `taoyds/test-suite-sql-eval` - influenced the test-suite style evaluation.
  https://github.com/taoyds/test-suite-sql-eval
- `QwenLM/Qwen` - influenced the Qwen model setup and reference points.
  https://github.com/QwenLM/Qwen


5) Build and run
----------------
Requirements:
- Python 3.10+
- ClassicModels MySQL database
- Hugging Face token where needed
- CUDA-capable environment for QLoRA training

Install:
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

Environment variables used by the notebooks and scripts:
- `INSTANCE_CONNECTION_NAME`
- `DB_USER`
- `DB_PASS`
- `DB_NAME`
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

Official run order:
1. `python scripts/run_baseline_llama.py`
2. `python scripts/run_baseline_qwen.py`
3. `python scripts/run_qlora_llama.py`
4. `python scripts/run_qlora_qwen.py`
5. `python scripts/run_react_llama.py`
6. `python scripts/run_react_qwen.py`
7. copy the chosen JSON files into `results/final_pack/`
8. `python scripts/build_final_analysis.py`


6) AI use
---------
AI use for this archive is documented in `AI_USE.md`.

That note explains where AI helped with scaffolding, helper scripts, output
analysis, formatting, and debugging support, and where the main project logic
was still written and checked by me.
