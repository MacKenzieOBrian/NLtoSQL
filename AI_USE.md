ai use note (aias level 3)

i used github copilot at level 3 (ai collaboration) for scaffold help and autocomplete suggestions during project coding
i planned the structure chose the approach reviewed every suggestion and rewrote or rejected anything that did not fit
if i did not understand a suggestion i changed it or did not use it

source of truth
code in this repo is treated as my own unless marked with an inline ai note copilot comment in source
each comment includes the exact prompt used in the form  # ai note copilot: "prompt text"
these comments are the per-piece prompt record required under aias level 3

attestation sentence
project coding used github copilot at aias level 3 for limited scaffold support and autocomplete suggestions. i planned the implementation reviewed all suggestions critically and kept responsibility for the final code integration testing and technical decisions.

files with inline ai note copilot comments and prompts

nl2sql/infra/experiment_helpers.py
  QLORA_EXPERIMENT_PRESETS  configure_react_notebook  train_qlora_adapter  _evaluate_react_ablation  run_react_notebook_eval

nl2sql/infra/training_set.py
  filter_training_records  run_training_set_validation

nl2sql/evaluation/simple_stats.py
  _coerce_per_item  compare_runs  build_summary_by_condition  build_pairwise_tests

nl2sql/evaluation/final_pack.py
  _validate_payload  build_tables_from_pack

nl2sql/evaluation/eval.py
  execution_accuracy  test_suite_accuracy_for_item  _build_item_pool  categorize_failure

nl2sql/agent/react_pipeline.py
  _parse_react_output  _compact_execution_error  _build_initial_messages  run_react_pipeline

nl2sql/agent/agent_tools.py
  schema_to_text

nl2sql/agent/prompts.py
  REACT_SYSTEM_PROMPT

nl2sql/core/schema.py
  NAME_LIKE_RE  build_schema_summary

nl2sql/core/prompting.py
  SYSTEM_INSTRUCTIONS  make_few_shot_messages

nl2sql/core/postprocess.py
  RANKING_HINT_RE  _strip_order_by_limit

scripts/build_final_analysis.py  run_baseline_llama.py  run_baseline_qwen.py  run_qlora_llama.py  run_qlora_qwen.py  run_react_llama.py  run_react_qwen.py
  main in each script

all unlisted code is hand written by me
