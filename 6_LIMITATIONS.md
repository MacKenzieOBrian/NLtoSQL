# Limitations Engineering Log (Reformatted)

Each limitation is presented in a four-part explanation format.

---

### Decision 6.1 - Keep TS suite-based rather than fully distilled

**Plain-language**  
A full distilled test-suite implementation was out of scope, so TS uses perturbed replicas instead.

**Technical description**  
`test_suite_accuracy_for_item` evaluates gold and predicted SQL across multiple TS DBs and requires consistent matches.

**Code locations**  
`nl2sql/eval.py` (`test_suite_accuracy_for_item`)  
`notebooks/03_agentic_eval.ipynb` (TS engine factory + evaluation loop)

**Justification**  
Zhong et al. (2020) motivate test-suite evaluation, but distillation is complex. The trade-off is that suite-based TS depends on perturbation quality.

---

### Decision 6.2 - Use heuristic schema linking rather than a learned linker

**Plain-language**  
A learned schema linker was out of scope. The tool‑driven loop currently uses full schema context; the keyword‑subset linker remains legacy only.

**Technical description**  
`get_schema` returns a structured schema and `schema_to_text` renders it for prompting. The keyword‑based `build_schema_subset` exists but is not wired into the tool‑driven loop yet.

**Code locations**  
`nl2sql/agent_tools.py` (`get_schema`, `schema_to_text`)  
Legacy: `nl2sql/agent_utils.py` (`build_schema_subset`)  
Legacy: `nl2sql/schema.py` (`build_schema_summary`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent._build_react_prompt`)

**Justification**  
Schema linking is a known bottleneck (Li et al., 2023; Zhu et al., 2024). The trade-off is heuristic coverage and weaker generalization.

---

### Decision 6.3 - Limit reflection to a single bounded step

**Plain-language**  
Unbounded reflection risks drift and opaque behavior. The tool‑driven loop uses a bounded step budget and only repairs on explicit execution errors.

**Technical description**  
`react_sql` iterates up to `REACT_MAX_STEPS`. When `run_sql` fails, the loop forces a `repair_sql` action and re‑runs guardrails before the next execution attempt.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` (`react_sql`)  
`nl2sql/agent_tools.py` (`run_sql`, `repair_sql`)  
`nl2sql/agent_utils.py` (`intent_constraints`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent.reflect_sql`)

**Justification**  
ReAct and ExCoT motivate feedback, but do not require unlimited revisions (Yao et al., 2023; Zhai et al., 2025). The trade-off is that some errors may need multiple reflections and will be missed.
