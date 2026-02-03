# Agent Design Engineering Log (Reformatted)

Each decision is presented in a four-part explanation format.

---

### Decision 3.1 - Enforce single-SELECT outputs (clean_candidate)

**Plain-language**  
Raw model outputs often include explanations or multiple statements, which causes false VA=0. A strict cleaner keeps only the first valid SELECT.

**Technical description**  
`clean_candidate` extracts the first SELECT, removes prompt echo, enforces presence of FROM, and rejects junk patterns. This prevents formatting noise from dominating VA/EX.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` helper layer ("## 6. Helper Layer: Staged Controls, Candidate Generation, and Error-Aware Repair")  
`nl2sql/llm.py` (`extract_first_select`)

**Justification**  
Constrained decoding ideas (Scholak et al., 2021) motivate filtering invalid continuations. The trade-off is that an incomplete first SELECT can still slip through.

---

### Decision 3.2 - Introduce projection contracts to stabilize EX

**Plain-language**  
EX was failing due to extra or reordered columns even when logic was correct. Projection contracts align output shape with the NLQ.

**Technical description**  
If the NLQ explicitly lists fields, `enforce_projection_contract` drops extra columns and preserves NLQ order. This is a post-generation clamp, not a model change.

**Code locations**  
`nl2sql/agent_utils.py` (`enforce_projection_contract`)  
`notebooks/03_agentic_eval.ipynb` helper layer (postprocess step)

**Justification**  
Execution-based metrics are sensitive to surface variation (Zhong et al., 2020). This clamp reduces false negatives but can over-restrict implicit multi-field queries.

---

### Decision 3.3 - Add intent constraints (query-type gate)

**Plain-language**  
Executable SQL can still answer the wrong question type. An intent gate rejects mismatched structures early.

**Technical description**  
`intent_constraints` checks for aggregates, GROUP BY, and ORDER/LIMIT patterns to align query structure with NLQ intent.

**Code locations**  
`nl2sql/agent_utils.py` (`intent_constraints`)  
`notebooks/03_agentic_eval.ipynb` (inside `evaluate_candidate`)

**Justification**  
Execution success does not guarantee semantic correctness (Zhong et al., 2020). The trade-off is possible false rejections on ambiguous NLQs.

---

### Decision 3.4 - Use schema-subset prompting to reduce wrong-table errors

**Plain-language**  
Wrong-table joins were a dominant error mode. Reducing schema scope improves table selection without retraining.

**Technical description**  
`build_schema_subset` uses keyword hints to select a smaller schema summary and join hints for the prompt.

**Code locations**  
`nl2sql/agent_utils.py` (`build_schema_subset`)  
`nl2sql/schema.py` (`build_schema_summary`)  
`notebooks/03_agentic_eval.ipynb` (prompt build step)

**Justification**  
Schema linking is a known bottleneck (Li et al., 2023; Zhu et al., 2024). The trade-off is heuristic coverage and weaker generalization.

---

### Decision 3.5 - Add semantic reranking over executable candidates

**Plain-language**  
When multiple candidates execute, a simple semantic score helps pick the one most aligned with the NLQ.

**Technical description**  
`semantic_score` and `count_select_columns` produce an explainable score; the top candidate is selected.

**Code locations**  
`nl2sql/agent_utils.py` (`semantic_score`, `count_select_columns`)  
`notebooks/03_agentic_eval.ipynb` (inside `evaluate_candidate`)

**Justification**  
Reranking strategies are common in NL->SQL pipelines (Zhu et al., 2024; Gao et al., 2025). The trade-off is that lexical overlap is not true semantic parsing.

---

### Decision 3.6 - Add error-aware repair with DB error feedback

**Plain-language**  
If all candidates fail, a bounded repair step can correct common schema or syntax mistakes.

**Technical description**  
`repair_sql` feeds the bad SQL, error message, and schema into the model and tests a small set of fixes. Repairs still pass through intent gates.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` helper layer (`repair_sql`)  
`nl2sql/query_runner.py` (`QueryRunner.run`)  
`scripts/run_full_pipeline.py` (single repair attempt path)

**Justification**  
Execution feedback is central to ExCoT (Zhai et al., 2025) and aligns with ReAct (Yao et al., 2023). The trade-off is potential drift toward executable but irrelevant SQL, so repair is bounded and gated.

---

### Decision 3.7 - Provide a deterministic fallback candidate

**Plain-language**  
Some NLQs still fail after repair. A deterministic fallback prevents empty outputs.

**Technical description**  
`vanilla_candidate` produces a deterministic few-shot output using the baseline prompt.

**Code locations**  
`nl2sql/agent_utils.py` (`vanilla_candidate`)  
`nl2sql/prompting.py` (`make_few_shot_messages`)

**Justification**  
ICL baselines are standard controls (Brown et al., 2020). The trade-off is that fallback can regress to baseline errors and does not use feedback.
