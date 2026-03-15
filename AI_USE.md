ai use note (aias level 3)

i used github copilot for limited scaffolding only
all final code was hand typed by me and edited to fit this project
if i did not understand a suggestion i rejected it

default rule
all code in this repo is hand coded unless it is explicitly listed below and marked with an inline ai scaffolding comment in source

attestation
i used github copilot for small scaffold blocks only
i reviewed and adapted those blocks and kept the final logic policy and integration decisions myself

exact places where ai scaffolding was used (small subparts only)
`nl2sql/infra/experiment_helpers.py` `configure_react_notebook` `_evaluate_react_ablation` `run_react_notebook_eval`
`nl2sql/infra/training_set.py` `filter_training_records` `run_training_set_validation`
`nl2sql/evaluation/final_pack.py` `_parse_filename` `_validate_payload` `build_tables_from_pack`
`nl2sql/evaluation/simple_stats.py` `_coerce_per_item` `compare_runs` `build_summary_by_condition` `build_pairwise_tests`
`scripts/build_final_analysis.py` `main`
`scripts/run_baseline_llama.py` `main`
`scripts/run_baseline_qwen.py` `main`
`scripts/run_qlora_llama.py` `main`
`scripts/run_qlora_qwen.py` `main`
`scripts/run_react_llama.py` `main`
`scripts/run_react_qwen.py` `main`

prompt types i used for scaffolding
give me the structure of functions for x
what functions would need to be made for x to happen
give me a plan/template for this script flow

inline comments in code are the source of truth for exact ai assisted subparts
everything not listed above is hand coded
