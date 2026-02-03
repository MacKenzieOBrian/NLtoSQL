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
A learned schema linker was out of scope, so a transparent keyword-based subset is used.

**Technical description**  
`build_schema_subset` uses keyword hints to reduce schema scope and provide join hints in the prompt.

**Code locations**  
`nl2sql/agent_utils.py` (`build_schema_subset`)  
`nl2sql/schema.py` (`build_schema_summary`)  
`notebooks/03_agentic_eval.ipynb` (prompt build step)

**Justification**  
Schema linking is a known bottleneck (Li et al., 2023; Zhu et al., 2024). The trade-off is heuristic coverage and weaker generalization.

---

### Decision 6.3 - Limit repair to a single bounded step

**Plain-language**  
Unbounded repair risks drift and opaque behavior. A single repair attempt keeps the loop interpretable.

**Technical description**  
`repair_sql` generates a small set of fixes and accepts only those that pass execution and intent gates.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` (helper layer `repair_sql`)  
`nl2sql/agent_utils.py` (`intent_constraints`)  
`scripts/run_full_pipeline.py` (single repair attempt path)

**Justification**  
ReAct and ExCoT motivate feedback, but do not require unlimited revisions (Yao et al., 2023; Zhai et al., 2025). The trade-off is that some errors may need multiple repairs and will be missed.
