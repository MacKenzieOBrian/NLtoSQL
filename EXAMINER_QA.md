# Examiner Q&A (Hard Mode) - NL->SQL Agent

Use this as a viva practice deck:
1. Read the question out loud.
2. Answer in 20-40 seconds from memory.
3. Then check the "Model answer" to see what you missed.

All answers are written to match the current implementation in this repo (not an idealized system).

---

## Contents
- Model Loading and Quantization (Practical Engineering Questions)
- Database Execution and Safety (Engine + QueryRunner)
- Schema Summary and Heuristic Schema Linking
- Prompting, Generation, and Leakage Control
- Deterministic Postprocess and Constraints
- Generation Controls and Prompt Hygiene (Why Outputs Are Cleaner Than Raw LLM Text)
- Agent Loop (ReAct + Execution Feedback)
- Scoring, Gating, and Repair (What You Added Beyond Prompting)
- Evaluation (VA / EM / EX / TS)
- Dataset, Results, and Reporting
- QLoRA / PEFT
- Methodology / Threats to Validity

---


## Model Loading and Quantization (Practical Engineering Questions)

### Q: "Why did you run the model in 4-bit NF4 rather than full precision?"

Model answer (say this):
"This is a practical constraint: running an 8B instruct model in full precision often exceeds typical Colab GPU VRAM. 4-bit NF4 quantization reduces memory footprint enough to run evaluation and QLoRA experiments. I keep the evaluator and prompts consistent so the trade-off is mainly performance vs feasibility, not an uncontrolled change in methodology."

Code pointers:
- `notebooks/03_agentic_eval.ipynb` (model load cell `# 3) Load model ...`, `BitsAndBytesConfig`)
- `nl2sql/llm.py:generate_sql_from_messages` (deterministic generation wrapper)

Related literature / references:
- [Ding et al., 2023](REFERENCES.md#ref-ding2023-peft) (PEFT motivation)
- [Modal, 2025](REFERENCES.md#ref-modal2025-vram) (engineering constraints)

---

### Q: "How do you keep baseline generation reproducible?"

Model answer (say this):
"Baseline generation is deterministic by default: `nl2sql/llm.py:generate_sql_from_messages` calls `model.generate` with `do_sample=False`. That way baseline and QLoRA are comparable without sampling variance."

Code pointers:
- `nl2sql/llm.py:generate_sql_from_messages`

Related literature:
- [Brown et al., 2020](REFERENCES.md#ref-brown2020-gpt3) (few-shot prompting context)

---

## Database Execution and Safety (Engine + QueryRunner)

### Q: "How do you actually connect to the database from Python?"

Model answer (say this):
"I use the Cloud SQL Python Connector and wrap it in a SQLAlchemy Engine using the `creator=` hook. In `nl2sql/db.py:create_engine_with_connector`, I define `getconn()` that calls `connector.connect(..., 'pymysql', ...)`, then pass `creator=getconn` into `sqlalchemy.create_engine`. After that I execute SQL via SQLAlchemy connections with `conn.execute(sqlalchemy.text(sql))`."

Code pointers:
- `nl2sql/db.py:create_engine_with_connector`
- `nl2sql/db.py:safe_connection`

Related literature / docs:
- [Zhu et al., 2024](REFERENCES.md#ref-zhu2024-survey) (schema-grounded NL->SQL motivates reliable DB access)

---

### Q: "Why did you introduce QueryRunner instead of calling engine.execute everywhere?"

Model answer (say this):
"`QueryRunner` centralizes controlled execution for model-generated SQL. It blocks destructive tokens, returns structured success/error metadata, and optionally captures a small preview for debugging. It is the agent's 'Act' tool and also provides VA by exposing `success`."

Code pointers:
- `nl2sql/query_runner.py:QueryRunner` (`_safety_check`, `run`, `QueryResult`)

Related literature:
- [Yao et al., 2023](REFERENCES.md#ref-yao2023-react) (the 'Act' tool concept in ReAct)

---

### Q: "How do you prevent the model from deleting tables or modifying rows?"

Model answer (say this):
"At execution time I apply a simple, explicit safety guard: both the QueryRunner and the EX harness block DDL/DML tokens like DROP/DELETE/UPDATE/INSERT. This is not a fully secure SQL sandbox, but it is a pragmatic safety layer for controlled dissertation experiments."

Code pointers:
- `nl2sql/query_runner.py:QueryRunner._safety_check`
- `nl2sql/eval.py:_safety_check` (EX harness)

---

### Q: "Is this safety check sufficient in general?"

Model answer (say this):
"No. A token blocklist is not a complete SQL security solution. It is appropriate here because the evaluation is run in a controlled environment on a known schema, and the goal is to prevent accidental damage during experiments rather than defend against adversarial input."

Code pointers:
- `nl2sql/query_runner.py:QueryRunner._safety_check`

Related literature:
- [Xi et al., 2025](REFERENCES.md#ref-xi2025-agents) (agents interacting with tools require safety constraints)

---

## Schema Summary and Heuristic Schema Linking

### Q: "How do you build the schema text shown to the model?"

Model answer (say this):
"I introspect the database and build a compact schema summary in `nl2sql/schema.py:build_schema_summary`. It lists tables and their columns, prioritizing primary keys and name/id-like columns, and formats each table as `table(col1, col2, ...)`. This reduces hallucinated tables/columns and makes prompting consistent."

Code pointers:
- `nl2sql/schema.py:build_schema_summary`

Related literature:
- [Zhu et al., 2024](REFERENCES.md#ref-zhu2024-survey) (schema-grounded prompting)
- [Hong et al., 2025](REFERENCES.md#ref-hong2025-survey) (Text-to-SQL survey context)

---

### Q: "What does 'heuristic schema linking' mean in your project?"

Model answer (say this):
"It means I reduce the schema context using simple, transparent keyword-to-table rules rather than training a learned schema linker. In `nl2sql/agent_utils.py:build_schema_subset`, I pick a small set of tables based on NLQ keywords and append join hints. This is auditable and reduces prompt length, but it is brittle to paraphrases."

Code pointers:
- `nl2sql/agent_utils.py:build_schema_subset`

Related literature:
- [Zhu et al., 2024](REFERENCES.md#ref-zhu2024-survey) (schema linking as a major bottleneck)
- [Li et al., 2023](REFERENCES.md#ref-li2023-resdsql) (schema linking in Text-to-SQL models)

---

## Prompting, Generation, and Leakage Control

### Q: "How do you ensure improvements are not just prompt drift?"

Model answer (say this):
"I keep the baseline system prompt stable in `nl2sql/prompting.py:SYSTEM_INSTRUCTIONS` and reuse the same evaluator in `nl2sql/eval.py` across baseline and QLoRA. That way, changes in scores can be attributed more to the method than to untracked prompt changes."

Code pointers:
- `nl2sql/prompting.py:SYSTEM_INSTRUCTIONS`
- `nl2sql/eval.py:eval_run`

Related literature:
- [Mosbach et al., 2023](REFERENCES.md#ref-mosbach2023-icl) (fair comparison issues)

---

### Q: "How do you prevent few-shot exemplar leakage in evaluation?"

Model answer (say this):
"In `nl2sql/eval.py:eval_run`, when `avoid_exemplar_leakage=True`, I filter the exemplar pool to ensure the evaluated (NLQ, gold SQL) pair is not included as an exemplar. That prevents inflated performance due to copying the test item from context."

Code pointers:
- `nl2sql/eval.py:eval_run` (`avoid_exemplar_leakage` branch)

Related literature:
- [Mosbach et al., 2023](REFERENCES.md#ref-mosbach2023-icl)

---

### Q: "How do you extract SQL from the model output?"

Model answer (say this):
"I use a best-effort regex extraction to keep only the first SELECT block. The reusable version is in `nl2sql/llm.py:extract_first_select`, and both the postprocess layer and the agent utilities use the same idea to remove prompt echo and trailing text."

Code pointers:
- `nl2sql/llm.py:extract_first_select`
- `nl2sql/agent_utils.py:clean_candidate`
- `nl2sql/postprocess.py:first_select_only`

Related literature:
- [Yu et al., 2018](REFERENCES.md#ref-yu2018-spider) (strict evaluation encourages clean, single-statement outputs)

---

## Deterministic Postprocess and Constraints

### Q: "Is postprocessing cheating? Are you using gold SQL to fix outputs?"

Model answer (say this):
"No gold SQL is used during postprocessing. The postprocess layer is deterministic and only uses the NLQ and the model output. It targets generic failure modes like prompt echo, multi-statement outputs, spurious ORDER BY/LIMIT, and unwanted ID-like columns. It is closer to output validation / constrained generation than to leaking labels."

Code pointers:
- `nl2sql/postprocess.py:guarded_postprocess`

Related literature:
- [Scholak et al., 2021](REFERENCES.md#ref-scholak2021-picard) (constraint/validity motivation)

---

### Q: "What are the key postprocess rules you rely on and what failure did they address?"

Model answer (say this):
"The combined guard is `guarded_postprocess`. It keeps only the first SELECT to avoid run-on text, strips ORDER BY/LIMIT when the NLQ does not imply ranking, prunes ID-like columns when the NLQ did not request IDs/codes, and clamps some 'list all ...' questions to a minimal projection. These were added after observing repeated VA failures and noisy EM/EX behavior due to surface-form drift."

Code pointers:
- `nl2sql/postprocess.py:first_select_only`
- `nl2sql/postprocess.py:_strip_order_by_limit`
- `nl2sql/postprocess.py:prune_id_like_columns`
- `nl2sql/postprocess.py:enforce_minimal_projection`

Related literature:
- [Yu et al., 2018](REFERENCES.md#ref-yu2018-spider) (strict evaluation encourages stable outputs)
- [Scholak et al., 2021](REFERENCES.md#ref-scholak2021-picard) (validity constraints)

---

### Q: "What is a projection contract in your implementation?"

Model answer (say this):
"A projection contract constrains the SELECT list when the NLQ explicitly enumerates fields. In `nl2sql/agent_utils.py:enforce_projection_contract`, I detect explicit field mentions using a small synonym map and then drop extra SELECT items deterministically while preserving the NLQ order. It constrains output shape without injecting joins or predicates."

Code pointers:
- `nl2sql/agent_utils.py:enforce_projection_contract`

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (surface variability motivates focusing on semantics)

---

### Q: "Why do you penalize the number of selected columns?"

Model answer (say this):
"In early runs the model often produced executable SQL that selected extra columns 'just in case'. That harms interpretability and can inflate EM mismatches. A simple projection penalty via `count_select_columns` biases selection toward minimal projections once multiple candidates are executable. It is not a proof of correctness, just a tie-breaker aligned with the prompt rule 'select only needed columns'."

Code pointers:
- `nl2sql/agent_utils.py:count_select_columns`
- `nl2sql/agent.py:ReactConfig` (`column_penalty`)
- `nl2sql/agent.py:ReactSqlAgent.evaluate_candidate` (where the penalty is applied)

Related literature:
- [Yu et al., 2018](REFERENCES.md#ref-yu2018-spider) (strictness of evaluation encourages stable, minimal outputs)

---

## Generation Controls and Prompt Hygiene (Why Outputs Are Cleaner Than Raw LLM Text)

### Q: "How do you stop the model from generating a long explanation after the SQL?"

Model answer (say this):
"In the agent notebook I use a generation stopping criterion that stops at the first semicolon. That reduces a common VA failure where the model appends an explanation after a valid SELECT, making the full output non-executable."

Code pointers:
- `nl2sql/agent.py:_StopOnSemicolon` (used in `ReactSqlAgent.generate_candidates`)
- `nl2sql/agent_utils.py:clean_candidate_with_reason` (also cuts at first `;`)
- `nl2sql/llm.py:extract_first_select` (baseline extraction)

Related literature:
- [Yu et al., 2018](REFERENCES.md#ref-yu2018-spider) (single-statement outputs matter for strict evaluation)

---

### Q: "Why do you strip prompt echo and keep only the first SELECT?"

Model answer (say this):
"Prompt echo and multi-statement outputs were a repeated failure mode: the model sometimes repeats schema text or outputs multiple queries. I handle this in multiple layers: `extract_first_select` and `first_select_only` keep only the first SELECT block, and the agent helper layer strips common echo patterns. This improves VA and keeps traces readable."

Code pointers:
- `nl2sql/llm.py:extract_first_select`
- `nl2sql/postprocess.py:first_select_only`
- `nl2sql/agent_utils.py:clean_candidate_with_reason` (prompt-echo cutoff + single-SELECT enforcement)

---

### Q: "Why do you normalize spaced-out keywords like 'S E L E C T'?"

Model answer (say this):
"Some generations contain spaced-out tokens (e.g., 'S E L E C T') which are not valid SQL. The cleaner normalizes these back into proper keywords using regex. This is a narrow, deterministic fix for a known formatting pathology."

Code pointers:
- `nl2sql/agent_utils.py:_normalize_spaced_keywords`
- `nl2sql/agent_utils.py:clean_candidate_with_reason` (calls `_normalize_spaced_keywords`)

---

## Agent Loop (ReAct + Execution Feedback)

### Q: "Where is the ReAct loop implemented in your project?"

Model answer (say this):
"The canonical ReAct-style loop is implemented in code (`nl2sql/agent.py:ReactSqlAgent.react_sql`). The agentic notebook imports it, sets an explicit `ReactConfig` (bounds + toggles), and then runs a per-item evaluation loop that computes VA/EM/EX/TS. Moving the loop into a module makes it easier to defend because there is one source of truth shared between notebooks and scripts."

Code pointers:
- `nl2sql/agent.py:ReactSqlAgent.react_sql`
- `nl2sql/agent.py:ReactConfig`
- `notebooks/03_agentic_eval.ipynb` (cell `# 6) Agent implementation (imported)` config + instantiation)
- `nl2sql/agent_utils.py` (`intent_constraints`, `semantic_score`, `count_select_columns`)
- `nl2sql/query_runner.py:QueryRunner.run` (Act step)

Related literature:
- [Yao et al., 2023](REFERENCES.md#ref-yao2023-react)
- [Zhai et al., 2025](REFERENCES.md#ref-zhai2025-excot)

---

### Q: "What is the 'Act' tool and what is the observation in your ReAct loop?"

Model answer (say this):
"The Act tool is executing SQL against the database via `runner.run(sql)` (QueryRunner). Each candidate is logged with an explicit Action/Observation pair: observations include cleanup rejections, execution errors, or intent mismatches. Those short observation strings are injected into the next prompt so the ReAct loop is explicit, not implicit."

Code pointers:
- `nl2sql/query_runner.py:QueryRunner.run`
- `nl2sql/agent.py:ReactSqlAgent.react_sql` (how errors become observations and drive retries/repair)

Related literature:
- [Yao et al., 2023](REFERENCES.md#ref-yao2023-react) (act + observation loop)
- [Zhai et al., 2025](REFERENCES.md#ref-zhai2025-excot) (execution feedback for Text-to-SQL)

---

### Q: "How is the Action/Observation history actually represented in the prompt?"

Model answer (say this):
"The agent formats recent trace items into explicit `Action:` and `Observation:` lines using `ReactSqlAgent._format_history_item`. It keeps only the last few items to control prompt length. This makes the loop ReAct-like in structure rather than just re-trying silently."

Code pointers:
- `nl2sql/agent.py:ReactSqlAgent._format_history_item`
- `nl2sql/agent.py:ReactSqlAgent._build_react_prompt`

Related literature:
- [Yao et al., 2023](REFERENCES.md#ref-yao2023-react)

---

### Q: "What is `accept_score` and why did you add it?"

Model answer (say this):
"`accept_score` is an optional threshold in `ReactConfig`. If it’s set, the loop only returns a candidate when its score clears that threshold; otherwise it keeps iterating until it hits the step budget. This makes multi‑step refinement meaningful but still bounded and explainable. In the notebook I set it explicitly so the loop actually performs multi‑step refinement during evaluation."

Code pointers:
- `nl2sql/agent.py:ReactConfig` (`accept_score`)
- `nl2sql/agent.py:ReactSqlAgent.react_sql`

---

### Q: "How do you debug the full ReAct process end‑to‑end?"

Model answer (say this):
"I turn on `ReactConfig.verbose=True`. That prints prompt settings, candidate counts, postprocess changes, gate outcomes, scores, and repair attempts. It gives a full readable trace without changing the model logic."

Code pointers:
- `nl2sql/agent.py:ReactConfig` (`verbose`)
- `nl2sql/agent.py:ReactSqlAgent._debug`

---

### Q: "Did you add any exemplars to the ReAct prompt? Why?"

Model answer (say this):
"Yes, I inject a very small exemplar block into the ReAct and tabular prompts (via `REACT_EXEMPLARS` in the notebook). The goal is to anchor join patterns, which were the most common EX failures. It’s explicit and logged so the trade‑off (possible leakage if exemplars come from the test set) is transparent."

Code pointers:
- `nl2sql/agent.py:ReactSqlAgent._format_exemplars`
- `nl2sql/agent.py:_build_react_prompt` / `_build_tabular_prompt`
- `notebooks/03_agentic_eval.ipynb` (REACT_EXEMPLARS)

---

### Q: "How do you keep the agent loop from running forever or becoming un-auditable?"

Model answer (say this):
"The loop is explicitly bounded by a small maximum number of steps, and repair is also bounded to a small number of candidates. This is deliberate: it controls compute cost and keeps traces short enough to audit and discuss in the dissertation."

Code pointers:
- `nl2sql/agent.py:ReactConfig` (`max_steps`, `num_cands`, `enable_repair`, `repair_num_cands`)
- `notebooks/03_agentic_eval.ipynb` (cell `# 6) Agent implementation (imported)` sets the config explicitly)

---

### Q: "Why do you generate multiple candidates per step instead of one?"

Model answer (say this):
"Because Text-to-SQL generation is brittle: a single sample can fail due to one wrong column or join. Generating a small set of diverse candidates increases the chance that at least one is executable and semantically close. The system still remains explainable because candidates are postprocessed deterministically, executed, gated by intent constraints, and logged in the trace."

Code pointers:
- `nl2sql/agent.py:ReactSqlAgent.generate_candidates`
- `nl2sql/agent.py:ReactSqlAgent.evaluate_candidate` (gates + scoring)
- `nl2sql/agent_utils.py:semantic_score`, `count_select_columns`

Related literature:
- [Zhai et al., 2025](REFERENCES.md#ref-zhai2025-excot) (benefits of execution feedback and iteration)

---

### Q: "Is your agent deterministic? If not, how do you defend the evaluation?"

Model answer (say this):
"The baseline evaluation harness is deterministic by default (`nl2sql/llm.py:generate_sql_from_messages` sets `do_sample=False`). The agent loop may use sampling to create diverse candidates, so it is stochastic. To defend results, I log full traces and keep the loop bounded; for stronger rigor, I would report variance across multiple seeds/runs."

Code pointers:
- `nl2sql/llm.py:generate_sql_from_messages` (deterministic baseline)
- `nl2sql/agent.py:ReactConfig` (`do_sample`, `temperature`, `top_p`)
- `notebooks/03_agentic_eval.ipynb` (cell `# 6) Agent implementation (imported)` sets sampling explicitly)

Related literature:
- [Mosbach et al., 2023](REFERENCES.md#ref-mosbach2023-icl) (fair comparisons; controlling confounds)

---

## Scoring, Gating, and Repair (What You Added Beyond Prompting)

### Q: "What are the acceptance criteria for a candidate SQL in your agent loop?"

Model answer (say this):
"A candidate has to pass an execution gate and an intent gate before it is eligible to be the final answer. The execution gate is `QueryRunner.run(sql)` (it must execute successfully as a SELECT). The intent gate is `nl2sql/agent_utils.py:intent_constraints(nlq, sql)`, which rejects SQL that contradicts the NLQ intent (e.g., top-k requires ORDER BY and LIMIT). Only candidates that pass gates are scored and compared."

Code pointers:
- `nl2sql/query_runner.py:QueryRunner.run`
- `nl2sql/agent_utils.py:intent_constraints`
- `nl2sql/agent.py:ReactSqlAgent.evaluate_candidate`

Related literature:
- [Yao et al., 2023](REFERENCES.md#ref-yao2023-react) (tool gating via observations)
- [Scholak et al., 2021](REFERENCES.md#ref-scholak2021-picard) (validity/constraint motivation)

---

### Q: "How do you score candidates, exactly?"

Model answer (say this):
"Candidate scoring is explicit and auditable: `score = semantic_score(nlq, sql) - column_penalty * count_select_columns(sql) + extra_score_fn(nlq, sql)`. `semantic_score` is a lightweight lexical heuristic that includes a small bonus for explicit NLQ values (e.g., 'USA', 'San Francisco') and a penalty if explicitly requested fields are missing. `count_select_columns` penalizes wide projections. `column_penalty` is a config constant, and `extra_score_fn` is an optional hook that defaults to 0.0 so additional heuristics are clearly separated."

Code pointers:
- `nl2sql/agent.py:ReactConfig` (`column_penalty`)
- `nl2sql/agent.py:ReactSqlAgent.evaluate_candidate` (implements the formula)
- `nl2sql/agent_utils.py:semantic_score`
- `nl2sql/agent_utils.py:count_select_columns`

Related literature:
- [Ojuri et al., 2025](REFERENCES.md#ref-ojuri2025-agents) (agent + heuristic evaluation framing)

---

### Q: "You call it semantic_score. Is it actually semantic?"

Model answer (say this):
"No, it is not a formal semantic parser. It is a transparent lexical heuristic: it gives points when NLQ keywords align with SQL tokens and penalizes missing expected aggregates. I treat it as a lightweight reranker to choose among already executable candidates, not as a correctness proof."

Code pointers:
- `nl2sql/agent_utils.py:semantic_score` (token overlap + keyword hints)

---

### Q: "Why do you give extra points for literal values like 'USA' or 'San Francisco'?"

Model answer (say this):
"Those are strong cues that a candidate captured the intended filter. I extract likely literal values from the NLQ (quoted strings, capitalized multi‑word names, abbreviations) and add a small bonus if they appear in the SQL. It’s a lightweight heuristic to avoid returning queries that miss obvious filters."

Code pointers:
- `nl2sql/agent_utils.py:_extract_value_hints`
- `nl2sql/agent_utils.py:semantic_score`

---

### Q: "How do you handle NLQs that explicitly list fields (e.g., 'names, codes, and MSRPs')?"

Model answer (say this):
"I detect explicitly enumerated fields with a lightweight synonym map and check whether the SQL includes them. If fields are missing, the candidate gets a scoring penalty and the ReAct trace logs an observation like 'Missing requested fields: productCode' so the next step can correct it. This is still deterministic and doesn’t use gold SQL."

Code pointers:
- `nl2sql/agent_utils.py:missing_explicit_fields`
- `nl2sql/agent_utils.py:semantic_score`
- `nl2sql/agent.py:ReactSqlAgent.evaluate_candidate`

---

### Q: "What is the intent gate doing, concretely?"

Model answer (say this):
"`intent_constraints` classifies the NLQ into a coarse intent (lookup, aggregate, grouped aggregate, topk) and checks for required SQL structures (e.g., grouped aggregate requires both an aggregate function and GROUP BY; topk requires ORDER BY and LIMIT). This prevents a frequent failure mode where executable SQL answers a different question than asked."

Code pointers:
- `nl2sql/agent_utils.py:classify_intent`
- `nl2sql/agent_utils.py:intent_constraints`

Related literature:
- [Zhai et al., 2025](REFERENCES.md#ref-zhai2025-excot) (execution feedback helps fix reasoning errors)

---

### Q: "When do you trigger repair, and what exactly goes into the repair prompt?"

Model answer (say this):
"Repair is triggered only after an execution error. The repair prompt includes the schema context, the user question, the failing SQL, and the database error message. Repair generation is bounded (a small number of fix candidates) and every proposed fix must pass the same execution and intent gates before it can be accepted."

Code pointers:
- `nl2sql/agent.py:ReactSqlAgent.repair_sql`
- `nl2sql/query_runner.py:QueryRunner.run` (execution gate used before and after repair)

Related literature:
- [Zhai et al., 2025](REFERENCES.md#ref-zhai2025-excot) (execution feedback for repair)

---

## Evaluation (VA / EM / EX / TS)

### Q: "You say EX is execution accuracy. How exactly do you calculate EX in code?"

Model answer (say this):
"EX is computed in `nl2sql/eval.py:execution_accuracy`. I execute both `pred_sql` and `gold_sql` on the same SQLAlchemy `Engine` via `conn.execute(sqlalchemy.text(sql))`, fetch up to a row cap, and if both succeed I compare the returned row tuples as a multiset using `collections.Counter`. If either query fails to execute, EX is 0 and I keep the error string for analysis."

Code pointers:
- `nl2sql/eval.py` (`execute_fetch`, `execution_accuracy`)

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (motivates semantic evaluation beyond surface form)
- [Yu et al., 2018](REFERENCES.md#ref-yu2018-spider) (EM is standard but strict)

---

### Q: "Does your EX check column names? If not, why is that defensible?"

Model answer (say this):
"In this repo, EX compares rows only and intentionally ignores column names. That is a pragmatic design choice: earlier experiments showed EX was dominated by harmless projection/alias drift even when the underlying row sets matched. I treat that as evaluation noise and instead control projection via deterministic postprocessing and projection contracts, while TS provides a stronger semantic check across perturbed DBs."

Code pointers:
- `nl2sql/eval.py:execution_accuracy` (note about comparing rows only)
- `nl2sql/postprocess.py:guarded_postprocess`
- `nl2sql/agent_utils.py:enforce_projection_contract`

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (semantic equivalence vs surface variability)

---

### Q: "Why do you use Counter for EX rather than sorting rows and comparing?"

Model answer (say this):
"Counter gives a multiset comparison: it ignores row order but preserves duplicates. A `set()` would drop duplicates and can be wrong. Sorting is possible, but it requires a stable type-aware ordering across mixed row types; Counter avoids that complexity in a dissertation harness."

Code pointers:
- `nl2sql/eval.py:execution_accuracy`

---

### Q: "What happens if either query returns an enormous result set?"

Model answer (say this):
"Both EX and TS use row caps. In EX, `execute_fetch` uses `fetchmany(max_rows + 1)` and if the query exceeds the cap it is treated as a comparison failure. This avoids heavy memory use and long comparisons dominating evaluation."

Code pointers:
- `nl2sql/eval.py:execute_fetch` (`max_rows`, `Result set too large` branch)

---

### Q: "What is VA and how exactly is it computed?"

Model answer (say this):
"VA is executability of the predicted SQL only. I run `QueryRunner.run(pred_sql)` and set VA=1 if `QueryResult.success` is true; otherwise VA=0. VA is necessary (the SQL must run) but not sufficient (it can run and still be wrong)."

Code pointers:
- `nl2sql/query_runner.py:QueryRunner.run`
- `nl2sql/eval.py:eval_run` (uses `QueryRunner` for VA)

Related literature:
- [Ojuri et al., 2025](REFERENCES.md#ref-ojuri2025-agents) (VA/EX style evaluation framing)

---

### Q: "What is EM and why do you report it if it is known to be flawed?"

Model answer (say this):
"EM is normalized string equality between predicted and gold SQL. I keep it as a diagnostic of surface-form drift and projection variability. It is not treated as a semantic metric; EX and TS are the semantic checks in this dissertation."

Code pointers:
- `nl2sql/postprocess.py:normalize_sql`
- `nl2sql/eval.py:eval_run` (EM computed from `normalize_sql`)

Related literature:
- [Yu et al., 2018](REFERENCES.md#ref-yu2018-spider) (EM as a common benchmark metric)
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (EM limitations motivate semantic evaluation)

---

### Q: "What is TS, and how exactly do you compute it?"

Model answer (say this):
"TS is test-suite accuracy. In `nl2sql/eval.py:test_suite_accuracy_for_item`, I execute gold and predicted SQL across multiple perturbed DB replicas using an engine factory `make_engine_fn(db_name)`. The prediction only passes if it matches the gold result on all usable replicas. This reduces the chance of 'lucky execution' on a single DB."

Code pointers:
- `nl2sql/eval.py:test_suite_accuracy_for_item`
- `notebooks/03_agentic_eval.ipynb` (engine factory cell `# 1b) Engine factory for TS`)

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (distilled test suites for semantic evaluation)

---

### Q: "When do you compare TS results as ordered vs unordered?"

Model answer (say this):
"TS treats results as ordered only when the gold SQL contains ORDER BY. If ordered, it compares row lists directly; otherwise it sorts normalized rows and compares them. That matches the SQL semantics that row order is only meaningful when ORDER BY is present."

Code pointers:
- `nl2sql/eval.py:_has_order_by`, `_results_match_ts`

---

### Q: "What do you do if gold fails on a TS replica database?"

Model answer (say this):
"TS has a `strict_gold` option. With `strict_gold=True`, if gold fails on any TS replica I treat TS as failed rather than skipping that replica, because skipping can inflate TS. With `strict_gold=False`, I can ignore replicas where gold fails, but I require at least one usable replica overall."

Code pointers:
- `nl2sql/eval.py:test_suite_accuracy_for_item` (`strict_gold` logic)

---

## Dataset, Results, and Reporting

### Q: "What exactly is in your evaluation dataset, and where is it loaded?"

Model answer (say this):
"The evaluation set is a JSON list of items where each item includes `nlq` and a gold `sql`. It is stored at `data/classicmodels_test_200.json` and is loaded by notebooks and scripts (e.g., the baseline/QLoRA harness in `nl2sql/eval.py:eval_run` expects a list of dicts with those keys)."

Code pointers:
- `data/classicmodels_test_200.json`
- `nl2sql/eval.py:eval_run` (expects `item['nlq']` and `item['sql']`)

Related literature:
- [Li et al., 2023](REFERENCES.md#ref-li2023-bigbench) (benchmarking motivates standardized datasets)

---

### Q: "What do you actually save after an evaluation run?"

Model answer (say this):
"For baseline/QLoRA runs, `nl2sql/eval.py:eval_run` can save a JSON payload containing run metadata (timestamp, k, seed, limit) and per-item fields like `nlq`, `gold_sql`, `raw_sql`, `pred_sql`, and boolean VA/EM/EX plus error strings. Agentic runs also log a trace per item in the notebook."

Code pointers:
- `nl2sql/eval.py:EvalItem.to_jsonable`
- `nl2sql/eval.py:eval_run` (payload format)
- `results/README.md` (where outputs are written; gitignore note)

---

### Q: "Why does your agentic notebook save JSON with `default=str`?"

Model answer (say this):
"The agentic eval saves TS debug samples that can include `Decimal` values from MySQL. Python’s `json.dumps` can’t serialize Decimal by default, so I use `default=str` to preserve the debug output without crashing at the end of a run. The numeric metrics remain computed in memory as floats; only the saved debug fields become strings."

Code pointers:
- `notebooks/03_agentic_eval.ipynb` (evaluation save block with `default=str`)

---

### Q: "If EX and TS disagree, how do you debug it?"

Model answer (say this):
"I inspect per-item artifacts: the predicted SQL, gold SQL, execution errors, and the agent trace. For TS specifically, `test_suite_accuracy_for_item` returns debug info per replica DB (gold_ok, pred_ok, match, and sample rows). For broad patterns, `scripts/analyze_results.py` gives a lightweight taxonomy of failure types."

Code pointers:
- `nl2sql/eval.py:test_suite_accuracy_for_item` (returns `debug` with `per_db`)
- `scripts/analyze_results.py`

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (suite-based debugging motivation)

---

## QLoRA / PEFT

### Q: "What exactly is QLoRA in your project, and why use it?"

Model answer (say this):
"QLoRA is a parameter-efficient fine-tuning approach: instead of updating all model weights, I train small low-rank adapters while keeping the base model quantized. This reduces VRAM requirements and makes fine-tuning feasible in an honors dissertation setting, while still testing whether task adaptation helps beyond prompting."

Code pointers:
- `notebooks/05_qlora_train_eval.ipynb` (QLoRA training + eval)
- `scripts/run_full_pipeline.py` (`wrap_peft`, `run_qlora_eval`)

Related literature:
- [Ding et al., 2023](REFERENCES.md#ref-ding2023-peft) (PEFT overview)
- [Mosbach et al., 2023](REFERENCES.md#ref-mosbach2023-icl) (fair comparisons for ICL vs fine-tuning)

---

### Q: "How do you ensure the evaluation is comparable between baseline and QLoRA?"

Model answer (say this):
"Both methods are scored with the same evaluator (`nl2sql/eval.py:eval_run`) and use the same schema summary and prompt builder (`nl2sql/prompting.py`). The only difference is whether the model is the base model or wrapped with PEFT adapters."

Code pointers:
- `nl2sql/eval.py:eval_run`
- `scripts/run_full_pipeline.py:wrap_peft`

Related literature:
- [Mosbach et al., 2023](REFERENCES.md#ref-mosbach2023-icl)

---

## Methodology / Threats to Validity

### Q: "How do you justify that your agentic improvements are not just overfitting to one metric?"

Model answer (say this):
"I report multiple metrics: VA for executability, EM as a diagnostic surface metric, EX for semantic correctness on the base DB, and TS as a robustness check across perturbed DBs. If a change improves EM but harms EX/TS, I treat it as suspicious and document the trade-off in the decision log."

Code pointers:
- `nl2sql/eval.py` (VA/EM/EX)
- `nl2sql/eval.py:test_suite_accuracy_for_item` (TS)
- `5_ITERATIVE_REFINEMENTS.md` (decision records)

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (metric limitations motivate semantic checks)

---

### Q: "What are the biggest limitations of your evaluation as implemented?"

Model answer (say this):
"First, EX ignores column names and is order-insensitive, so it can overestimate correctness in some edge cases. Second, TS quality depends on the quality of the perturbed replica databases; if perturbations are weak or gold breaks, TS can be misleading. Third, the agent loop can be stochastic if sampling is enabled, so results can vary unless multiple runs or seeds are reported."

Code pointers:
- `nl2sql/eval.py:execution_accuracy` (rows-only EX)
- `nl2sql/eval.py:test_suite_accuracy_for_item` (TS strict_gold)
- `notebooks/03_agentic_eval.ipynb` (CFG sampling)

Related literature:
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts)

---

### Q: "If you had more time, what would you improve first?"

Model answer (say this):
"I would make the agent evaluation more statistically robust by running multiple seeds for the sampling-based loop and reporting variance. I would also strengthen constraints using a more principled SQL parser/constrained decoding approach rather than regex-only guards, and I would expand TS perturbations to better separate spurious queries from true semantic matches."

Related literature:
- [Scholak et al., 2021](REFERENCES.md#ref-scholak2021-picard) (principled constrained decoding)
- [Zhong et al., 2020](REFERENCES.md#ref-zhong2020-ts) (semantic evaluation via suites)
