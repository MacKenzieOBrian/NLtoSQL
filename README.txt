NLtoSQL dissertation archive

this repo has the code notebooks and results for my nl to sql project on the classicmodels database
the main comparison is prompting vs qlora plus a bounded react style extension

quick structure
`nl2sql/` main package
`scripts/` final rerun entry points
`notebooks/` development notebooks
`results/final_pack/` curated json run files used for final reporting
`results/final_analysis/` csv outputs generated from final_pack

official reported runs in `results/final_pack/`
non agentic `k=0` conditions use 1 seed
non agentic `k=3` conditions use 10 seeds
react `k=3` conditions use 5 seeds

main research code is in
`nl2sql/core/*`
`nl2sql/agent/*`
`nl2sql/evaluation/eval.py`
`nl2sql/evaluation/grid_runner.py`

helper and analysis code is in
`nl2sql/infra/*`
`nl2sql/evaluation/final_pack.py`
`nl2sql/evaluation/simple_stats.py`
`scripts/build_final_analysis.py`
plus supporting notebooks

the notebooks were used during development
for final reruns use the scripts folder

ownership and provenance
the core method logic is my own work including prompt design schema summary guarded execution react loop evaluation setup and experiment design
there is no supervisor source code copied into this repo and no vendored third party modules
external dependencies are from `requirements.txt` and runtime model downloads

main external references used during implementation
ReAct repo https://github.com/ysymyth/ReAct
Transformers https://github.com/huggingface/transformers
PEFT https://github.com/huggingface/peft
Spider https://github.com/taoyds/spider
test suite eval https://github.com/taoyds/test-suite-sql-eval
Qwen repo https://github.com/QwenLM/Qwen

build and run
requirements python 3.10+ classicmodels mysql and a hugging face token
for qlora training use a cuda capable environment

setup
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`

env vars used by scripts and notebooks
`INSTANCE_CONNECTION_NAME`
`DB_USER`
`DB_PASS`
`DB_NAME`
`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

official run order
1. `python scripts/run_baseline_llama.py`
2. `python scripts/run_baseline_qwen.py`
3. `python scripts/run_qlora_llama.py`
4. `python scripts/run_qlora_qwen.py`
5. `python scripts/run_react_llama.py`
6. `python scripts/run_react_qwen.py`
7. copy selected json files into `results/final_pack/`
8. `python scripts/build_final_analysis.py`

ai use is documented in AI_USE.md
each ai assisted section is marked inline with a comment in the form  # ai note copilot: "prompt used"
these inline comments are the per piece prompt record required under aias level 3
all unlisted code is hand written
