# Iterative Refinements Engineering Log (Reformatted)

**Current status note:** The active tool‑driven ReAct loop lives in `notebooks/03_agentic_eval.ipynb`. Many decisions below reference the legacy candidate‑ranking loop in `nl2sql/agent.py` and are retained for historical ablations.

Each decision is presented in a four-part explanation format.

---

### Decision 5.1 - Stage-gated ablation of the agent loop

**Plain-language**  
When many controls were enabled at once, it was unclear which change improved EX. Stage gating isolates effects.

**Technical description**  
Controls are enabled incrementally via explicit configuration toggles (e.g., reflection on/off, schema subset on/off, projection contract on/off). This supports ablation-style runs where each change can be evaluated in isolation.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig`)  
`notebooks/03_agentic_eval.ipynb` (cell `# 6) Agent implementation (imported)` sets config values)

**Justification**  
Ablation is standard in NL->SQL evaluation (Zhu et al., 2024). The trade-off is additional run time and complexity.

---

### Decision 5.2 - Stop-on-semicolon and prompt-echo trimming

**Plain-language**  
Valid SQL was often followed by extra text, causing syntax errors and VA=0. Trimming fixes this without changing model weights.

**Technical description**  
The cleaner cuts at the first semicolon and removes prompt echo or trailing instructions before execution.

**Code locations**  
`nl2sql/agent.py` (`_StopOnSemicolon` in `ReactSqlAgent.generate_candidates`)  
`nl2sql/agent_utils.py` (`clean_candidate_with_reason`)  
`nl2sql/llm.py` (`extract_first_select`)

**Justification**  
Constrained decoding ideas (Scholak et al., 2021) motivate lightweight filtering. The trade-off is rare cases where valid semicolons are trimmed incorrectly.

---

### Decision 5.3 - Relax EX to row-only comparison

**Plain-language**  
EX was failing on correct results when column names or order differed. Comparing rows only better reflects semantic equivalence.

**Technical description**  
`execution_accuracy` compares row multisets using a Counter and does not require column name equality.

**Code locations**  
`nl2sql/eval.py` (`execution_accuracy`)  
`notebooks/03_agentic_eval.ipynb` (evaluation loop)

**Justification**  
Zhong et al. (2020) highlight surface-level variability. The trade-off is that column semantics can be ignored if values coincide.

---

### Decision 5.4 - Port TS harness into the core eval module

**Plain-language**  
Keeping TS only in the notebook made it harder to reuse and audit. Moving it into `nl2sql.eval` improves traceability.

**Technical description**  
TS functions are now in `nl2sql/eval.py` and imported by the notebook and scripts.

**Code locations**  
`nl2sql/eval.py` (`test_suite_accuracy_for_item`)  
`notebooks/03_agentic_eval.ipynb` (TS evaluation)

**Justification**  
Test-suite evaluation is a core metric (Zhong et al., 2020). The trade-off is that this TS remains suite-based, not fully distilled.

---

### Decision 5.5 - Expand intent cues + allow soft intent penalties

**Plain-language**  
The intent gate was rejecting good SQL because common aggregate phrasing (“how many”, “number of”) was misclassified as a lookup. The gate is now configurable as a soft penalty instead of a hard reject.

**Technical description**  
`classify_intent` includes regex cues for “how many / number of / how much” to reduce false lookup labels. In `ReactSqlAgent.evaluate_candidate`, intent mismatches can either hard‑reject or subtract a configurable penalty (`ReactConfig.enforce_intent_constraints`, `ReactConfig.intent_penalty`).

**Code locations**  
`nl2sql/agent_utils.py` (`classify_intent`, `intent_constraints`)  
`nl2sql/agent.py` (`ReactConfig.enforce_intent_constraints`, `ReactConfig.intent_penalty`, `ReactSqlAgent.evaluate_candidate`)

**Justification**  
Intent heuristics are useful but brittle. Soft penalties keep the signal while avoiding avoidable VA/EX loss on ambiguous NLQs.

---

### Decision 5.6 - Hybrid greedy + sampled candidates; rebalanced budget

**Plain-language**  
Sampling alone produced too many syntax errors. A greedy anchor plus sampled diversity improves candidate quality, and budget now prioritizes breadth over multi‑step retries.

**Technical description**  
Each prompt yields one greedy candidate and `per_prompt-1` sampled candidates (if enabled). Default `ReactConfig` now favors `max_steps=1` with a higher `num_cands` to increase coverage in a single step.

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent.react_sql`, `ReactConfig.max_steps`, `ReactConfig.num_cands`, `ReactConfig.do_sample`)

**Justification**  
This improves syntactic reliability without losing diversity and reduces time spent in low‑yield multi‑step loops.

---

### Decision 5.7 - Reduce keyword‑soup failures (decoding + cleaning + reflection hints)

**Plain-language**  
Most execution failures were pure syntax junk. We cut decoding entropy, reject keyword soup early, and nudge reflections toward the right query type.

**Technical description**  
Default decoding uses lower `temperature` and shorter `max_new_tokens`.  
`clean_candidate_with_reason` rejects empty/keyword‑only SELECT lists and high keyword‑to‑identifier ratios.  
`reflect_sql` includes a detected intent label to guide aggregate vs lookup fixes.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig.temperature`, `ReactConfig.max_new_tokens`, `ReactSqlAgent.reflect_sql`)  
`nl2sql/agent_utils.py` (`clean_candidate_with_reason`)

**Justification**  
Reduces unexecutable candidates while keeping generation diversity and bounded reflection.

---

### Decision 5.8 - Prefilter candidates before execution

**Plain-language**  
Many candidates are obviously weak before execution. Prefiltering cuts DB calls and focuses on plausible SQL.

**Technical description**  
Candidates are cleaned and given a lightweight lexical score, then only the top `max_exec_cands` are executed. This keeps the loop bounded while improving runtime efficiency.

**Code locations**  
`nl2sql/agent.py` (`ReactConfig.max_exec_cands`, `ReactSqlAgent._prefilter_candidates`)

**Justification**  
Reduces wasted execution on junk candidates and keeps traces focused on plausible hypotheses.

---

### Decision 5.9 - Remove tabular prompt variant

**Plain-language**  
Maintaining two prompt styles added complexity without clear benefit. The loop now uses a single ReAct prompt.

**Technical description**  
`ReactConfig.use_tabular_prompt` and `_build_tabular_prompt` were removed. The loop generates candidates from the single ReAct prompt.

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent._build_react_prompt`, `ReactSqlAgent.react_sql`)

**Justification**  
Reduces prompt‑engineering degrees of freedom and keeps the loop easier to audit and defend.

---

### Decision 5.10 - Schema validation + error‑specific reflection guidance

**Plain-language**  
Unknown tables/columns were a common failure mode. We now reject those before execution and give reflections targeted guidance.

**Technical description**  
The loop validates `FROM/JOIN` tables and qualified columns against the schema summary.  
Repair prompts add a short guidance line keyed to common MySQL errors (e.g., syntax 1064, unknown column 1054, ambiguous column 1052).

**Code locations**  
`nl2sql/agent.py` (`ReactSqlAgent._schema_validate`, `ReactSqlAgent.validate_sql`, `ReactSqlAgent.reflect_sql`)

**Justification**  
Pre‑execution validation reduces wasted DB calls and raises VA, while targeted reflection hints improve fix quality without new tooling.

---

### Decision 5.11 - Tool‑Driven ReAct Loop (Explicit Actions)

**Plain-language**  
The agent loop now makes tool choices explicit: the LLM emits actions and Python executes them.

**Technical description**  
The notebook defines a bounded ReAct loop using `agent_tools.py` and a single system prompt (`prompts.py`). The loop bootstraps with `get_schema`, then iterates Thought → Action(tool) → Observation. Guardrails run between `generate_sql`/`repair_sql` and `run_sql`, and execution must succeed before `finish`.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` (tool-driven `react_sql`)  
`nl2sql/agent_tools.py` (tools)  
`nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)

**Justification**  
Matches ReAct’s explicit tool abstraction and aligns with agent‑mediated NL→SQL workflows while keeping traces auditable.
