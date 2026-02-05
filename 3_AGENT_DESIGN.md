# Agent Design Engineering Log (Reformatted)

**Current status note:** The evaluation notebook now uses a tool‑driven ReAct loop (`notebooks/03_agentic_eval.ipynb`) backed by `nl2sql/agent_tools.py` and `nl2sql/prompts.py`. The legacy candidate‑ranking loop in `nl2sql/agent.py` is kept for comparison and ablations; entries below that reference `ReactSqlAgent.*` reflect that legacy loop.
**Legend:** Decisions tagged **(Legacy)** refer to the candidate‑ranking loop in `nl2sql/agent.py` and are not executed in the tool‑driven notebook. Active tool‑loop decisions include 3.1–3.4, 3.6–3.8, 3.13 (cleaning only), and 3.16.

Each decision is presented in a four-part explanation format.

---

### Decision 3.1 - Enforce single-SELECT outputs (clean_candidate)

**Plain-language**  
Raw model outputs often include explanations or multiple statements, which causes false VA=0. A strict cleaner keeps only the first valid SELECT.

**Technical description**  
`clean_candidate` extracts the first SELECT, removes prompt echo, enforces presence of FROM, and rejects junk patterns. This prevents formatting noise from dominating VA/EX.

**Code locations**  
`nl2sql/agent_utils.py` (`clean_candidate_with_reason`, `clean_candidate`)  
`notebooks/03_agentic_eval.ipynb` (`_apply_guardrails`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent.evaluate_candidate`)  
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
`notebooks/03_agentic_eval.ipynb` (`_apply_guardrails`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent.postprocess_sql`)

**Justification**  
Execution-based metrics are sensitive to surface variation (Zhong et al., 2020). This clamp reduces false negatives but can over-restrict implicit multi-field queries.

---

### Decision 3.3 - Add intent constraints (query-type gate)

**Plain-language**  
Executable SQL can still answer the wrong question type. Intent alignment now supports a hard gate *or* a soft penalty when classification is uncertain.

**Technical description**  
`intent_constraints` checks for aggregates, GROUP BY, and ORDER/LIMIT patterns to align query structure with NLQ intent.  
`classify_intent` was expanded to catch common aggregate phrasing (“how many”, “number of”, “how much”) to reduce false lookup labels.  
In the tool‑driven loop, intent is applied as a post‑execution gate (`intent_ok`) before `finish`. The soft‑penalty option exists only in the legacy candidate loop.

**Code locations**  
`nl2sql/agent_utils.py` (`classify_intent`, `intent_constraints`)  
`notebooks/03_agentic_eval.ipynb` (`react_sql`)  
Legacy: `nl2sql/agent.py` (`ReactConfig.enforce_intent_constraints`, `ReactConfig.intent_penalty`, `ReactSqlAgent.evaluate_candidate`)

**Justification**  
Execution success does not guarantee semantic correctness (Zhong et al., 2020). A soft penalty preserves good SQL when intent heuristics are uncertain, while still discouraging structurally mismatched answers.

---

### Decision 3.4 - Use schema-subset prompting to reduce wrong-table errors

**Plain-language**  
Wrong-table joins were a dominant error mode. Reducing schema scope improves table selection without retraining.

**Technical description**  
`build_schema_subset` uses keyword hints to select a smaller schema summary and join hints for the prompt. The tool‑driven loop calls `link_schema` to apply this subset before SQL generation.

**Code locations**  
`nl2sql/agent_utils.py` (`build_schema_subset`)  
`nl2sql/agent_tools.py` (`link_schema`)  
`notebooks/03_agentic_eval.ipynb` (`react_sql`)

**Justification**  
Schema linking is a known bottleneck (Li et al., 2023; Zhu et al., 2024). The trade-off is heuristic coverage and weaker generalization.

---

### Decision 3.5 (Legacy) - Add semantic reranking over executable candidates

**Plain-language**  
When multiple candidates execute, a simple semantic score helps pick the one most aligned with the NLQ.

**Technical description**  
`semantic_score` and `count_select_columns` produce an explainable score; the top candidate is selected. A small “literal‑value” bonus is added when the SQL includes explicit NLQ values (e.g., “USA”, “San Francisco”) to favor correct filters. If the NLQ explicitly enumerates fields, a penalty is applied when the SQL omits any of those fields. When intent constraints are configured as *soft*, a small intent mismatch penalty is also subtracted.

**Code locations**  
`nl2sql/agent_utils.py` (`semantic_score`, `count_select_columns`)  
`nl2sql/agent.py` (`ReactSqlAgent.evaluate_candidate`)

**Justification**  
Reranking strategies are common in NL->SQL pipelines (Zhu et al., 2024; Gao et al., 2025). The trade-off is that lexical overlap is not true semantic parsing.

---

### Decision 3.6 - Add error-aware reflection with DB error feedback

**Plain-language**  
If all candidates fail, a bounded reflection step can correct common schema or syntax mistakes.

**Technical description**  
`repair_sql` feeds the bad SQL, error message, NLQ, and schema into the model and returns a revised SQL string. The tool loop runs `run_sql` after guardrails and gates `finish` on execution success and intent alignment.

**Code locations**  
`nl2sql/agent_tools.py` (`repair_sql`, `run_sql`)  
`notebooks/03_agentic_eval.ipynb` (`react_sql`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent.reflect_sql`)  
`nl2sql/query_runner.py` (`QueryRunner.run`)

**Justification**  
Execution feedback is central to ExCoT and aligns with ReAct; the trade-off is potential drift toward executable but irrelevant SQL, so reflection is bounded and gated.  
Refs: `REFERENCES.md#ref-zhai2025-excot`, `REFERENCES.md#ref-yao2023-react`.

---

### Decision 3.7 - Provide a deterministic fallback candidate

**Plain-language**  
Some NLQs still fail after reflection. A deterministic fallback prevents empty outputs.

**Technical description**  
`vanilla_candidate` produces a deterministic few-shot output using the baseline prompt. The tool loop uses it as a bounded fallback if the agent fails to reach `finish`.

**Code locations**  
`nl2sql/agent_utils.py` (`vanilla_candidate`)  
`notebooks/03_agentic_eval.ipynb` (`react_sql`)  
`nl2sql/prompting.py` (`make_few_shot_messages`)

**Justification**  
ICL baselines are standard controls (Brown et al., 2020). The trade-off is that fallback can regress to baseline errors and does not use feedback.

---

### Decision 3.8 - Make ReAct feedback explicit in prompts

**Plain-language**  
Earlier versions claimed a ReAct loop but did not pass any real action/observation history back to the model. This update makes the feedback loop concrete and auditable.

**Technical description**  
The tool‑driven loop emits a strict Thought/Action/Observation trace. The system prompt defines the loop and the LLM is required to output explicit tool calls; Python executes them and appends Observations verbatim.

**Code locations**  
`nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)  
`notebooks/03_agentic_eval.ipynb` (`react_sql`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent._format_history_item`, `ReactSqlAgent._build_react_prompt`, `ReactSqlAgent.evaluate_candidate`)

**Justification**  
ReAct and ExCoT emphasize using observations to steer revisions; the trade-off is prompt length, so only recent items are included.  
Refs: `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-zhai2025-excot`.

---

### Decision 3.9 (Legacy) - Optional acceptance threshold for multi-step refinement

**Plain-language**  
If the best candidate looks weak, the loop should try another step instead of returning early.

**Technical description**  
`ReactConfig.accept_score` sets an optional score threshold. When set, `react_sql` keeps iterating until a candidate clears the threshold or the step budget is exhausted.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig.accept_score`, `ReactSqlAgent.react_sql`)

**Justification**  
This makes the multi-step loop meaningful without changing the generation method. The trade-off is extra runtime if the threshold is set too aggressively.

---

### Decision 3.10 (Legacy) - Inject a small join exemplar into ReAct prompts

**Plain-language**  
Join errors were the dominant EX failure mode. A single concrete join example helps anchor the correct pattern.

**Technical description**  
The ReAct prompt builder includes a short exemplar block when `REACT_EXEMPLARS` is provided. The notebook builds a small exemplar set from the test set (including an office/city join) and injects it into the ReAct prompt.

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent._format_exemplars`, `_build_react_prompt`)  
`notebooks/03_agentic_eval.ipynb` (cell `# 2) Load schema summary + test set` for `REACT_EXEMPLARS`)

**Justification**  
Few-shot anchoring is a standard way to reduce join errors in NL->SQL (Brown et al., 2020; survey work on schema-grounded prompting). The trade-off is potential leakage if exemplars are drawn from the same test set, so this is logged and used as a controlled, explicit decision.

---

### Decision 3.11 (Legacy) - Hybrid candidate generation (greedy + sampled)

**Plain-language**  
Pure sampling produced too much syntax junk, but pure greedy reduced coverage. A hybrid strategy keeps one clean anchor while still exploring alternatives.

**Technical description**  
Per prompt, the loop generates one greedy candidate (deterministic) plus sampled candidates for the remainder of the per‑prompt budget. Sampling is still controlled by `ReactConfig.do_sample`, but the greedy anchor is always included.

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent.react_sql`, `ReactConfig.do_sample`, `ReactConfig.num_cands`)

**Justification**  
This balances syntactic reliability with diversity in a best‑of‑N setting. It also reduces wasted budget on low‑likelihood garbage without turning the loop into a single deterministic guess.

---

### Decision 3.12 (Legacy) - Shift budget from multi-step loops to more candidates

**Plain-language**  
Multi‑step refinement was not yielding consistent gains, so the default budget now favors more candidates in a single step.

**Technical description**  
Default `ReactConfig` uses `max_steps=1` and a higher `num_cands` to increase candidate coverage without extra loop iterations. These values remain configurable for ablations.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig.max_steps`, `ReactConfig.num_cands`)

**Justification**  
More initial candidates increase the chance of an executable, semantically aligned SQL while keeping runtime bounded and traces simpler.

---

### Decision 3.13 (Partial legacy) - Keyword‑soup rejection + intent‑aware reflection hints

**Plain-language**  
Many failures were pure keyword soup (“SELECT FROM WHERE LIMIT …”). The cleaner now rejects these early, and reflections get a small intent hint to avoid wrong query shapes.

**Technical description**  
`clean_candidate_with_reason` rejects empty/keyword‑only SELECT lists and high keyword‑to‑identifier ratios; this is still applied in the tool‑driven guardrails.  
Intent‑aware reflection hints were part of the legacy candidate loop and are not currently wired into the tool‑driven `repair_sql` prompt.

**Code locations**  
`nl2sql/agent_utils.py` (`clean_candidate_with_reason`)  
`notebooks/03_agentic_eval.ipynb` (`_apply_guardrails`, `react_sql`)  
Legacy: `nl2sql/agent.py` (`ReactSqlAgent.reflect_sql`)

**Justification**  
Early rejection reduces wasted execution/reflection attempts, and intent hints help reflections target aggregation structure. The trade‑off is occasional false rejects on complex subqueries.

---

### Decision 3.14 (Legacy) - Prefilter candidates before execution

**Plain-language**  
Executing every candidate is expensive and many are obviously weak. A lightweight prefilter ranks candidates and executes only the top subset.

**Technical description**  
Before hitting the database, candidates are cleaned and scored with the same lexical signal used for reranking. Only the top `max_exec_cands` are executed; the rest are skipped. This is an execution‑time optimization, not a change to model generation.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig.max_exec_cands`, `ReactSqlAgent._prefilter_candidates`, `ReactSqlAgent.react_sql`)

**Justification**  
Reduces wasted execution attempts on low‑plausibility SQL, improving runtime and lowering exposure to syntax‑junk candidates without sacrificing explainability.

---

### Decision 3.15 (Legacy) - Schema‑aware validation + error‑specific reflection hints

**Plain-language**  
Many candidates failed due to unknown tables or columns. We now reject those before execution and tailor reflections based on the DB error code.

**Technical description**  
Before execution, the loop checks `FROM/JOIN` table names and qualified `table.column` references against the schema summary.  
Repair prompts include a short “Fix guidance” line keyed to common MySQL errors (e.g., syntax, unknown column, ambiguous column).

**Code locations**  
Legacy candidate loop: `nl2sql/agent.py` (`ReactSqlAgent._schema_validate`, `ReactSqlAgent.validate_sql`, `ReactSqlAgent.reflect_sql`)

**Justification**  
Schema validation removes obvious invalid SQL early; error‑type hints make reflections more targeted without adding new tools.

---

### Decision 3.16 - Tool‑Driven ReAct Loop (Explicit Actions)

**Plain-language**  
The agent now follows an explicit Thought → Action → Observation loop where the LLM chooses tools and Python executes them.

**Technical description**  
The notebook defines a bounded tool loop. The LLM emits `Action: tool_name[json_args]` using a single system prompt. Python executes tool functions (`get_schema`, `link_schema`, `extract_constraints`, `generate_sql`, `validate_sql`, `validate_constraints`, `run_sql`, `repair_sql`, `finish`) and appends the Observation back into the trace. Guardrails run between generation/repair and validation/execution; `validate_sql` must pass before `validate_constraints`, `validate_constraints` must pass before `run_sql`, and `run_sql` must succeed before `finish`. Validation/execution failures force a `repair_sql` action, and trace summaries log action sequences and compliance flags.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` (tool loop `react_sql`)  
`nl2sql/agent_tools.py` (tool interface)  
`nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)

**Justification**  
Aligns the implementation with ReAct and agent‑mediated NL→SQL workflows by making tool choices explicit and auditable.  
Refs: `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-ojuri2025-agents`.
