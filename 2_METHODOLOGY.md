# Methodology Engineering Log (Reformatted)

Each decision is presented in a four-part explanation format.

---

### Decision 2.1 - Use ClassicModels as the fixed-schema benchmark

**Plain-language**  
Mixing schemas made results hard to interpret. A fixed schema makes improvements attributable to the method rather than the data.

**Technical description**  
ClassicModels is used as the evaluation substrate. A consistent schema summary is built for prompting, and a fixed test set is evaluated across all stages.

**Code locations**  
`data/classicmodels_test_200.json`  
`data/train/classicmodels_train_200.jsonl`  
`nl2sql/schema.py` (`build_schema_summary`)  
`notebooks/05_qlora_train_eval.ipynb` ("## 2) Load benchmark + training set")

**Justification**  
Spider (Yu et al., 2018) emphasizes controlled evaluation within fixed schemas, and Ojuri et al. (2025) use ClassicModels for agent evaluation. The trade-off is limited claims about cross-domain generalization.

---

### Decision 2.2 - Use small supervised splits (200 train / 200 test)

**Plain-language**  
Full-scale fine-tuning is infeasible under Colab-class constraints. A small, clean split is enough to measure relative improvements.

**Technical description**  
A 200/200 split is used for training and testing. Leakage checks and validation are performed in the dataset prep notebook.

**Code locations**  
`data/train/classicmodels_train_200.jsonl`  
`data/classicmodels_test_200.json`  
`notebooks/04_build_training_set.ipynb` ("## 3) Leakage + safety checks")

**Justification**  
PEFT work shows gains with limited data (Ding et al., 2023; Goswami et al., 2024). The trade-off is that small data underestimates full-capacity performance.

---

### Decision 2.3 - Fix the prompt format and use deterministic decoding

**Plain-language**  
Prompt drift and sampling noise made results unstable. Deterministic decoding keeps comparisons fair.

**Technical description**  
A fixed system prompt and deterministic decoding (`do_sample=False`) are used for baselines. This isolates the effect of method changes.

**Code locations**  
`nl2sql/prompting.py` (`SYSTEM_INSTRUCTIONS`, `make_few_shot_messages`)  
`nl2sql/llm.py` (`generate_sql_from_messages`, `do_sample=False`)  
`notebooks/02_baseline_prompting_eval.ipynb`

**Justification**  
Controlled ICL baselines are required for fair comparison (Brown et al., 2020; Mosbach et al., 2023). The trade-off is that deterministic decoding can understate best-case performance.

---

### Decision 2.4 - Use QLoRA adapters instead of full fine-tuning

**Plain-language**  
Full fine-tuning exceeds the available VRAM. QLoRA makes training feasible while keeping evaluation consistent.

**Technical description**  
QLoRA adapters are trained in a dedicated notebook and evaluated using the same harness as baselines. The base model remains fixed.

**Code locations**  
`notebooks/05_qlora_train_eval.ipynb` ("## 4) Load base model (4-bit) + configure QLoRA", "## 6) Train (SFT with TRL)")  
`nl2sql/eval.py` (`eval_run`)

**Justification**  
QLoRA is a standard PEFT method with strong empirical support (Ding et al., 2023; Goswami et al., 2024). The trade-off is that adapters may not fully capture complex relational reasoning with limited data.

---

### Decision 2.5 - Evaluate agentic refinement without changing weights

**Plain-language**  
Prompting and QLoRA still produced semantic errors. Execution feedback can improve correctness without retraining.

**Technical description**  
A ReAct-style loop generates, executes, and refines SQL candidates using execution feedback. Only control logic is added; weights are unchanged.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` ("## ReAct execution-guided pipeline (best version so far)")  
`nl2sql/query_runner.py` (`QueryRunner.run`)  
`nl2sql/agent_utils.py` (`intent_constraints`, `semantic_score`)

**Justification**  
ReAct (Yao et al., 2023) and ExCoT (Zhai et al., 2025) motivate execution feedback loops. The trade-off is that control logic cannot fully correct deep semantic gaps without stronger priors.
