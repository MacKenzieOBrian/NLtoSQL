# Agent Design Engineering Log (Reformatted)

Each decision is presented in a four-part explanation format.

---

### Decision 3.1 - Enforce single-SELECT outputs (clean_candidate)

**Plain-language**  
Raw model outputs often include explanations or multiple statements, which causes false VA=0. A strict cleaner keeps only the first valid SELECT.

**Technical description**  
`clean_candidate` extracts the first SELECT, removes prompt echo, enforces presence of FROM, and rejects junk patterns. This prevents formatting noise from dominating VA/EX.

**Code locations**  
`nl2sql/agent_utils.py` (`clean_candidate_with_reason`, `clean_candidate`)  
`nl2sql/agent.py` (`ReactSqlAgent.evaluate_candidate`)  
`nl2sql/llm.py` (`extract_first_select`)

**Justification**  
Constrained decoding ideas (Scholak et al., 2021) motivate filtering invalid continuations. The trade-off is that an incomplete first SELECT can still slip through.

---

### Decision 3.2 - Introduce projection contracts to stabilize EX

**Plain-language**  
EX was failing due to extra or reordered columns even when logic was correct. Projection contracts align output shape with the NLQ.

**Technical description**  
If the NLQ explicitly lists fields, `enforce_projection_contract` drops extra columns and preserves NLQ order. The field list uses simple synonyms (including plural forms) and a context‑gated “codes” hint to reduce false misses. This is a post-generation clamp, not a model change.

**Code locations**  
`nl2sql/agent_utils.py` (`enforce_projection_contract`)  
`nl2sql/agent.py` (`ReactSqlAgent.postprocess_sql`)

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
`nl2sql/agent.py` (`ReactSqlAgent.evaluate_candidate`)

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
`nl2sql/agent.py` (`ReactSqlAgent._build_react_prompt`)

**Justification**  
Schema linking is a known bottleneck (Li et al., 2023; Zhu et al., 2024). The trade-off is heuristic coverage and weaker generalization.

---

### Decision 3.5 - Add semantic reranking over executable candidates

**Plain-language**  
When multiple candidates execute, a simple semantic score helps pick the one most aligned with the NLQ.

**Technical description**  
`semantic_score` and `count_select_columns` produce an explainable score; the top candidate is selected. A small “literal‑value” bonus is added when the SQL includes explicit NLQ values (e.g., “USA”, “San Francisco”) to favor correct filters.

**Code locations**  
`nl2sql/agent_utils.py` (`semantic_score`, `count_select_columns`)  
`nl2sql/agent.py` (`ReactSqlAgent.evaluate_candidate`)

**Justification**  
Reranking strategies are common in NL->SQL pipelines (Zhu et al., 2024; Gao et al., 2025). The trade-off is that lexical overlap is not true semantic parsing.

---

### Decision 3.6 - Add error-aware repair with DB error feedback

**Plain-language**  
If all candidates fail, a bounded repair step can correct common schema or syntax mistakes.

**Technical description**  
`repair_sql` feeds the bad SQL, error message, and schema into the model and tests a small set of fixes. Repairs still pass through intent gates.

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent.repair_sql`)  
`nl2sql/query_runner.py` (`QueryRunner.run`)  
`scripts/run_full_pipeline.py` (CLI sanity-check path uses the same agent module)

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

---

### Decision 3.8 - Make ReAct feedback explicit in prompts

**Plain-language**  
Earlier versions claimed a ReAct loop but did not pass any real action/observation history back to the model. This update makes the feedback loop concrete and auditable.

**Technical description**  
`_format_history_item` formats trace items as Action/Observation and `_build_react_prompt` now injects the last few items into the prompt. `evaluate_candidate` attaches an `obs` string for clean rejects, execution failures, and intent mismatches so the model gets a concise error signal on the next step.

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent._format_history_item`, `ReactSqlAgent._build_react_prompt`, `ReactSqlAgent.evaluate_candidate`)

**Justification**  
ReAct (Yao et al., 2023) and ExCoT (Zhai et al., 2025) emphasize using observations to steer revisions. The trade-off is prompt length, so only recent items are included.

---

### Decision 3.9 - Optional acceptance threshold for multi-step refinement

**Plain-language**  
If the best candidate looks weak, the loop should try another step instead of returning early.

**Technical description**  
`ReactConfig.accept_score` sets an optional score threshold. When set, `react_sql` keeps iterating until a candidate clears the threshold or the step budget is exhausted.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig.accept_score`, `ReactSqlAgent.react_sql`)

**Justification**  
This makes the multi-step loop meaningful without changing the generation method. The trade-off is extra runtime if the threshold is set too aggressively.
