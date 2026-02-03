# Literature Engineering Log (Reformatted)

Each decision is presented in a four-part explanation format.

---

### Decision 1.1 - Center the review on execution-centric evaluation (VA/EX/TS)

**Plain-language**  
Early drafts focused on exact match and prompt wording, but baseline runs showed VA much higher than EX. The review needed to explain why valid SQL can still be semantically wrong.

**Technical description**  
Baseline evaluations showed many VA=1, EX=0 cases. This indicated that string matching was not the right primary lens. The literature framing was shifted to emphasize execution-based evaluation and test-suite ideas, while EM was retained only as a diagnostic signal.

**Code locations**  
`nl2sql/eval.py` (`execution_accuracy`, `test_suite_accuracy_for_item`, `eval_run`)  
`nl2sql/query_runner.py` (`QueryRunner.run`)  
`notebooks/03_agentic_eval.ipynb` ("## ReAct execution-guided pipeline (best version so far)")

**Justification**  
Zhong et al. (2020) show that execution-based and test-suite evaluation are required to handle SQL surface variability. Yu et al. (2018) establish execution-centric evaluation in NL->SQL benchmarks. The trade-off is that all execution metrics depend on database fidelity, so the framing explicitly acknowledges that limitation.

---

### Decision 1.2 - Use a prompt -> PEFT -> agent ladder as the organizing frame

**Plain-language**  
The literature was initially a flat list of methods, which made the experimental stages look arbitrary. A staged ladder clarifies why each experiment exists.

**Technical description**  
The review was reorganized to mirror the experimental sequence: prompting (ICL), then PEFT (QLoRA), then agentic control with execution feedback. This aligns the narrative with the actual notebooks and evaluation runs.

**Code locations**  
`notebooks/02_baseline_prompting_eval.ipynb` (baseline prompting)  
`notebooks/05_qlora_train_eval.ipynb` (QLoRA training and eval)  
`notebooks/03_agentic_eval.ipynb` (agentic loop)

**Justification**  
Brown et al. (2020) motivate ICL baselines, Ding et al. (2023) and Goswami et al. (2024) motivate PEFT, and Yao et al. (2023) plus Zhai et al. (2025) motivate feedback-driven agent loops. The trade-off is that this ladder can under-represent hybrid methods, but it keeps the study defensible and attributable.

---

### Decision 1.3 - Treat proprietary agents as upper bounds, not baselines

**Plain-language**  
Comparing directly to closed, proprietary agents weakens reproducibility. They are better described as upper bounds rather than baselines.

**Technical description**  
All evaluations are kept within the open harness. Proprietary outputs are not used inside the pipeline, and comparisons are framed qualitatively.

**Code locations**  
`nl2sql/eval.py` (`eval_run`)  
`notebooks/02_baseline_prompting_eval.ipynb`  
`notebooks/03_agentic_eval.ipynb`

**Justification**  
Surveys of LLM agents highlight reproducibility constraints and the need for open evaluation (Xi et al., 2025). Strong results in proprietary systems (Ojuri et al., 2025) are informative but not reproducible here. The trade-off is that state-of-the-art comparisons remain qualitative.

---

### Decision 1.4 - Emphasize schema linking as a dominant error source

**Plain-language**  
Trace reviews repeatedly showed wrong-table joins, even when SQL executed successfully. The literature needed to reflect schema linking as a central bottleneck.

**Technical description**  
Error logs and EX failures clustered around table selection and join path errors. A lightweight schema-subset prompt and join hints were introduced to reduce the scope of schema linking errors without retraining.

**Code locations**  
`nl2sql/agent_utils.py` (`build_schema_subset`)  
`nl2sql/schema.py` (`build_schema_summary`)  
`notebooks/03_agentic_eval.ipynb` ("Top-down: Prepare schema summary and a small debug slice")

**Justification**  
RESDSQL (Li et al., 2023) and surveys (Zhu et al., 2024; Hong et al., 2025) identify schema linking as a primary bottleneck. The approach here is heuristic and transparent, which keeps it auditable but limits generalization.
