# Project Context (NL->SQL Dissertation) - Single File For Viva Practice

This file is intentionally "too detailed": it is meant to be pasted into another LLM as context so it can drill you with examiner-style questions and you can practice answering defensively.

It is written to match the current code in this repo (not vague textbook definitions).

Last updated: 2026-02-05 (tool-driven ReAct loop with explicit actions + validation + schema linking)

---

## 0) One Paragraph Summary (If You Get Interrupted)

This project builds and evaluates a Natural Language -> SQL (MySQL) system over the ClassicModels database. It starts with a baseline prompt + few-shot in-context learning, adds QLoRA adapters for parameter-efficient fine-tuning, and then adds a tool‑driven ReAct loop where the LLM explicitly chooses tools (link schema, extract constraints, generate SQL, validate SQL, validate constraints, run SQL, repair SQL) and uses database execution as environment feedback. Deterministic guardrails are applied between generation and execution to improve executability and semantic correctness. Evaluation is reported using VA (valid SQL / executability), EM (normalized exact match), EX (execution accuracy by result-set comparison on the base DB), and TS (test-suite accuracy across multiple perturbed DB replicas).

---

## 0.1) Discovery Narrative (From LOGBOOK Dates)

This is the short, defensible story of *how* the method evolved. It is based on the dated entries in `LOGBOOK.md` and is meant to be said out loud.

**Narrative arc (one paragraph)**  
I began by scoping the problem and identifying a reproducibility gap (Sep–Oct 2025), then built a deterministic prompting baseline and learned that VA can improve without semantic gains. I introduced QLoRA to test whether lightweight fine‑tuning fixes semantic joins/aggregates (Jan 2026), but execution‑guided experiments showed that feedback alone does not fix semantic routing. That pushed me toward explicit guardrails (projection, intent, schema subset) and then toward a *true* tool‑driven ReAct loop with validation and auditable decision logs (Feb 2026), so each accept/reject can be justified.

**Timeline anchors**  
- **2025-09-29** — Scoping: identified the reproducibility gap and narrowed the scope to ClassicModels.  
- **2025-10-06 to 2025-10-13** — Literature consolidation: shifted evaluation emphasis from EM to EX/TS for semantic validity.  
- **2025-12-14** — Baseline prompting: VA improved, EX remained low; prompted the move to PEFT.  
- **2026-01-12** — QLoRA: EX improved but still relied on prompting for schema grounding.  
- **2026-01-23 to 2026-01-25** — Early ReAct/guardrails: VA high, EX low → “semantic bottleneck” diagnosis.  
- **2026-01-31** — Control layer: projection/intent/schema‑subset guardrails added to target EX failures.  
- **2026-02-01 to 2026-02-02** — TS harness and quick‑test toggles: enabled semantic robustness checks.  
- **2026-02-04** — Pivot to explicit ReAct: replaced candidate‑ranking with a tool‑driven Thought→Action→Observation loop.  
- **2026-02-05** — Tightened loop: added validation, schema linking, constraints, forced repair, and decision logs.

**What I learned at each stage**  
- Baselines expose *where* errors come from (shape, intent, schema), not just how many.  
- Execution feedback only helps if the loop is explicit and bounded; otherwise it drifts.  
- TS reveals semantic errors that EX can miss on a single DB snapshot.  

**What I chose *not* to do (and why)**  
- **No learned schema linker** — would add a second model and reduce interpretability; kept heuristic and auditable.  
- **No full distilled test‑suite construction** — TS is suite‑based (perturbed replicas) due to time/compute constraints.  
- **No unbounded reflection** — bounded steps keep behavior interpretable and evaluation costs stable.  
- **No heavy search / beam‑ranking pipeline** — candidate ranking obscured the ReAct action/observation contract.  

---

## 1) Repo Map (What Exists Where)

Top-level docs (canonical reading order):
- `1_LITERATURE.md`: what prior work says and why this problem is hard
- `2_METHODOLOGY.md`: experimental method + iteration loop (error-driven development)
- `3_AGENT_DESIGN.md`: the ReAct-style agent design and guards
- `4_EVALUATION.md`: definitions + implementation of VA/EM/EX/TS
- `5_ITERATIVE_REFINEMENTS.md`: decision records of changes made after observing failures
- `6_LIMITATIONS.md`: what is not solved / what is weak
- `LOGBOOK.md`: chronological notes (kept as a "lab notebook")

Code (reusable modules):
- `nl2sql/db.py`: Cloud SQL Connector -> SQLAlchemy engine creation + safe connection context manager
- `nl2sql/query_runner.py`: SELECT-only query executor (used as the agent's "Act" tool and as VA signal)
- `nl2sql/schema.py`: schema introspection + schema summary text used in prompts
- `nl2sql/prompting.py`: baseline system prompt + few-shot message builder
- `nl2sql/llm.py`: generation helper + extraction of first SELECT
- `nl2sql/postprocess.py`: deterministic SQL cleanups / guardrails
- `nl2sql/agent_utils.py`: lightweight heuristics for schema subset, intent constraints, semantic scoring, and cleanup (rejects keyword‑soup outputs)
- `nl2sql/agent_tools.py`: tool interface used by the explicit ReAct loop
- `nl2sql/prompts.py`: single system prompt for Thought/Action/Observation
- `nl2sql/agent.py`: legacy candidate-based loop (kept for comparison)
- `nl2sql/eval.py`: evaluation harness for VA/EM/EX and TS

Notebooks (Colab-first experiments):
- `notebooks/02_baseline_prompting_eval.ipynb`: baseline prompt eval (zero-shot vs few-shot)
- `notebooks/04_build_training_set.ipynb`: train-set validation / leakage checks
- `notebooks/05_qlora_train_eval.ipynb`: QLoRA fine-tune + eval
- `notebooks/03_agentic_eval.ipynb`: tool-driven ReAct loop + evaluation (main "demo" notebook)

Scripts (CLI / utilities):
- `scripts/run_full_pipeline.py`: CLI mirror of baseline + QLoRA + small reflection sanity check
- `scripts/analyze_results.py`: quick post-hoc error breakdown from a JSON run

Data + outputs:
- `data/classicmodels_test_200.json`: evaluation items (each has `nlq` + `sql`)
- `data/train/`: training data for QLoRA
- `results/`: outputs from notebooks (kept in git for reproducibility in this repo)

---

## 1.1) Key Function Index (Fast "Where Is That Implemented?" Map)

Database + execution:
- `nl2sql/db.py:create_engine_with_connector`: Cloud SQL Connector -> SQLAlchemy Engine
- `nl2sql/db.py:safe_connection`: context manager for `engine.connect()`
- `nl2sql/query_runner.py:QueryRunner.run`: executes SELECT-only SQL + returns structured metadata

Schema:
- `nl2sql/schema.py:list_tables`: `SHOW TABLES`
- `nl2sql/schema.py:get_table_columns`: query INFORMATION_SCHEMA.COLUMNS
- `nl2sql/schema.py:build_schema_summary`: schema text for prompts

Prompting + generation:
- `nl2sql/prompting.py:SYSTEM_INSTRUCTIONS`: stable system prompt used across experiments
- `nl2sql/prompting.py:make_few_shot_messages`: builds (system + schema + exemplars + NLQ) messages
- `nl2sql/llm.py:generate_sql_from_messages`: deterministic `model.generate(...)` wrapper
- `nl2sql/llm.py:extract_first_select`: best-effort extraction of first SELECT block

Deterministic postprocess:
- `nl2sql/postprocess.py:guarded_postprocess`: combined guardrail pipeline
- `nl2sql/postprocess.py:first_select_only`: keep first SELECT...;
- `nl2sql/postprocess.py:prune_id_like_columns`: drop ID-like columns unless NLQ requests them
- `nl2sql/postprocess.py:enforce_minimal_projection`: clamp list-all questions to 1 projection
- `nl2sql/postprocess.py:normalize_sql`: string normalization used for EM

Agent utilities (lightweight heuristics):
- `nl2sql/agent_utils.py:clean_candidate`: strict SELECT-only cleaner for raw model output
- `nl2sql/agent_utils.py:build_schema_subset`: heuristic schema linking via keyword->table hints
- `nl2sql/agent_utils.py:enforce_projection_contract`: drop extra SELECT fields when NLQ enumerates fields (uses synonyms + context-gated “codes” hint)
- `nl2sql/agent_utils.py:intent_constraints`: detect NLQ intent mismatches (hard reject or soft penalty depending on config)
- `nl2sql/agent_utils.py:missing_explicit_fields`: detect missing explicitly requested fields for validation
- `nl2sql/agent_utils.py:semantic_score` and `count_select_columns`: reranking heuristics (includes literal-value hint for filters + penalty for missing explicitly requested fields)

Agent tools + prompt:
- `nl2sql/agent_tools.py:get_schema`: structured schema (tables/columns/PK/FK)
- `nl2sql/agent_tools.py:link_schema`: heuristic schema linker (subset + join hints)
- `nl2sql/agent_tools.py:extract_constraints`: heuristic constraint extraction (aggregate/limit/order/distinct)
- `nl2sql/agent_tools.py:validate_sql`: schema + formatting validation before execution
- `nl2sql/agent_tools.py:validate_constraints`: constraint validation before execution
- `nl2sql/agent_tools.py:run_sql`: executes SQL (SELECT-only) and returns success/error + rows
- `nl2sql/agent_tools.py:generate_sql` / `repair_sql`: LLM-based SQL generation/repair tools
- `nl2sql/prompts.py:REACT_SYSTEM_PROMPT`: single system prompt for Thought/Action/Observation

Evaluation:
- `nl2sql/eval.py:eval_run`: baseline/QLoRA evaluation loop for VA/EM/EX
- `nl2sql/eval.py:execution_accuracy`: EX comparator (exec both, compare results)
- `nl2sql/eval.py:test_suite_accuracy_for_item`: TS comparator (exec across DB replicas)

Scripts:
- `scripts/run_full_pipeline.py`: CLI mirror (baseline, qlora, small reflection check)
- `scripts/analyze_results.py`: heuristic failure breakdown for a JSON run

Agent (tool-driven ReAct loop lives in the notebook, tools live in code):
- `nl2sql/agent_tools.py`: `get_schema`, `link_schema`, `extract_constraints`, `generate_sql`, `validate_sql`, `validate_constraints`, `run_sql`, `repair_sql`, `finish`
- `nl2sql/prompts.py`: `REACT_SYSTEM_PROMPT` (Thought/Action/Observation)
- `notebooks/03_agentic_eval.ipynb`: defines `react_sql` tool loop and logs full traces + decision logs + compliance summaries (actions/repairs/compliance)
- Verbose tracing: set the notebook flags to print full loop outputs for debugging.
- The loop is bounded by `REACT_MAX_STEPS`.

---

## 2) How To Reproduce (Conceptually)

You can describe the project as three progressively stronger "methods" run on the same dataset:

1. Baseline prompting
   - input: NLQ + schema summary + k exemplars
   - output: one SQL query
   - scoring: VA/EM/EX (and optionally TS)
   - notebook: `notebooks/02_baseline_prompting_eval.ipynb`
   - code: `nl2sql/prompting.py`, `nl2sql/llm.py`, `nl2sql/postprocess.py`, `nl2sql/eval.py`

2. QLoRA fine-tuned adapters + prompting
   - same as baseline, but model weights are adapted via LoRA/QLoRA
   - notebook: `notebooks/05_qlora_train_eval.ipynb`
   - code: still uses `nl2sql/eval.py` for scoring; adapters loaded via PEFT

3. Agentic ReAct loop (tool-driven execution feedback)
  - bounded loop: Thought → Action(tool) → Observation → repeat until `finish`
  - `run_sql` must succeed before `finish`; errors trigger `repair_sql`
  - notebook: `notebooks/03_agentic_eval.ipynb`
  - code: tools in `nl2sql/agent_tools.py`, prompt in `nl2sql/prompts.py`

Environment variables (do not hardcode values in write-up):
- `INSTANCE_CONNECTION_NAME` (GCP Cloud SQL instance connection name)
- `DB_USER`, `DB_PASS`, `DB_NAME` (MySQL credentials + base db name, typically `classicmodels`)
- `HF_TOKEN` (for gated model access)

---

## 3) Core Infrastructure: "How Do You Actually Execute SQL?"

### What actually happens in this repo (not a vague answer)

There are two closely related "execution" paths:

1) Execution gate / VA / observation feedback (predicted SQL only)
- Used by: ReAct loop + VA metric
- Implemented in: `nl2sql/query_runner.py` (`QueryRunner.run`)
- Runs: predicted SQL against the base DB engine
- Returns: a `QueryResult` object (success boolean, error string, columns, optional DataFrame preview)

2) Execution Accuracy EX (predicted SQL AND gold SQL)
- Used by: EX metric
- Implemented in: `nl2sql/eval.py` (`execution_accuracy` -> `execute_fetch`)
- Runs: both predicted SQL and gold SQL against the base DB engine
- Returns: boolean match + error strings (if either failed)

### Code path (base DB)

Engine creation:
- `nl2sql/db.py:create_engine_with_connector(...)`
  - uses the Cloud SQL Python Connector
  - defines a `getconn()` function that returns a DB-API connection
  - passes `creator=getconn` into `sqlalchemy.create_engine(...)`

Minimal code excerpt (Engine + Connector):
```python
# nl2sql/db.py
connector = Connector()
def getconn():
    return connector.connect(instance_connection_name, "pymysql", user=user, password=password, db=db_name)
engine = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn, future=True)
```

Safe connection wrapper:
- `nl2sql/db.py:safe_connection(engine)`
  - yields a SQLAlchemy `Connection`
  - ensures `conn.close()` in `finally`

Query execution:
- `nl2sql/query_runner.py:QueryRunner.run(sql, params=None, capture_df=True)`
  - calls `_safety_check(sql)` to block destructive tokens
  - executes `conn.execute(sqlalchemy.text(sql), params or {})`
  - calls `result.fetchall()` (full fetch) and `result.keys()` (column names)
  - optionally returns a preview `DataFrame` for debugging (not used for scoring)

Why SQLAlchemy `text(sql)` is used:
- consistent execution API
- reduces driver-specific differences vs raw cursor execution

Minimal code excerpt (QueryRunner):
```python
# nl2sql/query_runner.py
def run(self, sql: str, *, params=None, capture_df: bool = True) -> QueryResult:
    self._safety_check(sql)
    with safe_connection(self.engine) as conn:
        result = conn.execute(sqlalchemy.text(sql), params or {})
        rows = result.fetchall()
        cols = list(result.keys())
    # returns success/error/cols + optional preview DataFrame
```

### Safety story you can defend

This project runs model-generated SQL against a real database.
So the executor blocks destructive tokens (DDL/DML) using a simple, auditable blocklist.

Code:
- `nl2sql/query_runner.py:QueryRunner._safety_check`
- `nl2sql/eval.py:_safety_check` (separate safety check used by EX harness)

Limitations (be honest):
- Token blocklists are not perfect SQL security.
- This is a dissertation safety guard for a controlled setting, not a hardened production sandbox.

---

## 4) Evaluation Metrics (VA / EM / EX / TS) - What They Are AND How They Are Computed

This section is the most likely to be questioned, because it is easy to describe metrics vaguely but hard to explain the actual implementation.

### 4.1 VA (Validity / Executability)

Definition used here:
- VA = 1 if the predicted SQL executes successfully as a SELECT on the base DB.
- VA = 0 otherwise.

Implementation:
- `QueryRunner.run(pred_sql)` returns `QueryResult.success`.
- VA is `bool(meta.success)`.

Code pointers:
- `nl2sql/query_runner.py:QueryRunner.run`
- In the agent notebook: `notebooks/03_agentic_eval.ipynb` cell `# 9) Full ReAct-style evaluation ...`

Key design point:
- VA is necessary but not sufficient: VA=1 can still be semantically wrong.

### 4.2 EM (Exact Match)

Definition used here:
- EM is a diagnostic string metric.
- EM = 1 if normalized predicted SQL string equals normalized gold SQL string.

Normalization used here:
- lowercasing
- trim whitespace
- drop trailing semicolon

Implementation:
- `nl2sql/postprocess.py:normalize_sql` (used in `nl2sql/eval.py:eval_run`)
- In the agent notebook, EM normalization happens inline in the evaluation cell.

Key design point:
- EM is known to undercount semantically correct but differently formatted SQL.
- EM is kept because it diagnoses "surface-form drift" and can explain VA/EX disagreements.

### 4.3 EX (Execution Accuracy) - The One Supervisors Ask About

Definition used here (IMPORTANT: match the code, not an ideal metric):
- EX = 1 if executing the predicted SQL and gold SQL on the base DB yields the same result rows (as a multiset).
- EX = 0 otherwise.

What EX does NOT do in this repo:
- It does NOT require exact string match.
- It does NOT require matching column names.
- It does NOT require matching row order (unless the SQL itself enforces it; but the current EX comparator is order-insensitive).

Implementation in `nl2sql/eval.py`:
- `execution_accuracy(engine, pred_sql, gold_sql)` calls `execute_fetch` for each SQL.
- `execute_fetch`:
  - safety check (`_safety_check`)
  - `conn.execute(sqlalchemy.text(sql))`
  - `result.keys()` (columns) and `fetchmany(max_rows + 1)` (rows)
  - if > max_rows, treat as failure for comparison (prevents huge compare)
- `execution_accuracy`:
  - if gold fails: EX=0 (and return gold error)
  - if pred fails: EX=0 (and return pred error)
  - else compare `Counter(pred_rows) == Counter(gold_rows)`

Small code excerpt (key lines only):
```python
# nl2sql/eval.py
pred_ok, _, pred_rows, _ = execute_fetch(engine=engine, sql=pred_sql, max_rows=max_compare_rows)
gold_ok, _, gold_rows, _ = execute_fetch(engine=engine, sql=gold_sql, max_rows=max_compare_rows)
return Counter(pred_rows) == Counter(gold_rows), None, None
```

Why use a multiset (`Counter`)?
- ignores ordering differences when ORDER BY is absent
- preserves duplicates (a plain set comparison would be wrong)

Why ignore column names in EX here?
- empirical: earlier runs were dominated by harmless projection/alias drift
- goal: measure whether the query "returns the right rows", not whether it prints the same headings

Trade-off (say this out loud if asked):
- ignoring column names can overestimate correctness when returning correct rows but wrong selected fields.
- this is one reason TS exists (semantic robustness) and why postprocessing/intent constraints exist (to reduce wrong projection).

### 4.4 TS (Test-Suite Accuracy / Semantic Robustness)

Motivation:
- EX can be "lucky" on one DB instance.
- Zhong et al. (2020) propose evaluating semantic equivalence via test suites (multiple DBs) because EM/EX can be misleading.

Definition used here:
- TS = 1 if predicted SQL matches gold SQL on ALL usable TS databases (replicas with controlled perturbations).
- TS = 0 otherwise.

Implementation in `nl2sql/eval.py:test_suite_accuracy_for_item`:
- Inputs:
  - `make_engine_fn(db_name)` -> SQLAlchemy Engine for that DB
  - list of `suite_db_names` (e.g., `classicmodels_ts_01`, ...)
  - `gold_sql`, `pred_sql`
- For each TS DB:
  - run gold and pred using `_run_select_ts`
  - if gold fails:
    - strict_gold=True: TS=0 (do not inflate TS by ignoring broken gold)
    - strict_gold=False: skip that DB (but require at least 1 usable DB overall)
  - if pred fails: TS=0
  - compare results:
    - treat results as ordered only if gold has ORDER BY
    - otherwise sort rows and compare
  - also check column count equality (len(gold_cols) == len(pred_cols))

Key trade-offs you should be ready to defend:
- If TS DB generation is flawed, TS can undercount (gold breaks) or overcount (perturbations too weak).
- TS multiplies evaluation cost by N databases; you need row caps (`max_rows`) to keep it feasible.

---

## 5) The Agent (Tool‑Driven ReAct Loop) - What Was Added Beyond "Just Prompting"

### 5.1 What "ReAct" means here

ReAct (Yao et al., 2023) is the pattern "Reasoning + Acting":
- produce an intermediate trace (Thought / Action)
- take an action in an external environment (here: execute SQL)
- observe feedback (result preview or error)
- use the observation to improve the next attempt

In this project, the "Act" tool is `QueryRunner.run(sql)` (SELECT-only), exposed via `agent_tools.run_sql`.

### 5.2 Where the agent loop lives

Canonical implementation (single source of truth):
- `nl2sql/agent_tools.py` (tool interface)
- `nl2sql/prompts.py` (system prompt for Thought/Action/Observation)
- `notebooks/03_agentic_eval.ipynb` (defines the tool-driven `react_sql` loop)

Reusable shared helpers used by the notebook:
- `nl2sql/agent_utils.py`:
  - `build_schema_subset`
  - `enforce_projection_contract`
  - `intent_constraints`
  - `semantic_score`
  - `count_select_columns`
 - `nl2sql/postprocess.py`:
   - `guarded_postprocess`

### 5.3 The loop at a level you can explain in 60 seconds

1. Bootstrap the trace with the user question and a `get_schema` observation.
2. LLM outputs Thought + Action (tool call).
3. Python executes the tool and returns Observation.
4. Guardrails run between `generate_sql`/`repair_sql` and `run_sql`.
5. `run_sql` must succeed before `finish`; failures feed into `repair_sql`.
6. Loop is bounded by `REACT_MAX_STEPS`; fallback is used if it never finishes.

### 5.4 Why the agent is designed as "minimal interventions"

Academic defense:
- The agent does not change the base model weights (unless you are in the QLoRA experiment).
- Most improvements are deterministic constraints and bounded retries.
- This makes the system auditable: you can attribute gains to specific interventions.

Related work anchors:
- ReAct (Yao et al., 2023): interleaving reasoning and acting with tool feedback.
- ExCoT (Zhai et al., 2025): execution feedback for Text-to-SQL reasoning.
- PICARD (Scholak et al., 2021): constrained decoding / validity enforcement (this project approximates constraints with deterministic guards instead of incremental parsing).

---

## 6) Post-processing / Constraints (Why "Guards" Exist)

This project uses deterministic postprocessing to reduce common, repeatable failure modes.

Main entrypoint:
- `nl2sql/postprocess.py:guarded_postprocess(sql, nlq)`

Guards (what they do and why):

1) Keep only the first SELECT statement
- function: `first_select_only`
- why: models sometimes output multiple statements or add extra text; extra content often makes VA=0.

2) Strip ORDER BY / LIMIT when NLQ does not imply ranking
- function: `_strip_order_by_limit`
- why: spurious ORDER BY/LIMIT is common and can change semantics or break EM comparability.

3) Prune ID-like columns unless NLQ asks for them
- function: `prune_id_like_columns`
- why: evaluation gold often uses minimal projections; models often add ids/codes; this inflates EM failures and can mislead analysis.
- limitation: if the NLQ implicitly expects an id field, this can remove a legitimate column.

4) Enforce minimal projection for "list all ..." patterns
- function: `enforce_minimal_projection`
- why: a simple heuristic to stop the model from selecting multiple columns when the intent is "list entity names".

Important defense point:
- these are not "cheating": they are deterministic transformations motivated by observed error patterns.
- they aim to stabilize evaluation and reduce noise due to surface-form variability.

Related work anchor:
- PICARD-style constrained decoding is a principled way to enforce validity; this project uses simpler regex guards for auditability and because a full parser/constraint decoder was out of scope.

---

## 7) Heuristic Schema Linking (What It Means In This Repo)

Definition (in plain terms):
- "Heuristic schema linking" means using simple, transparent rules to guess which tables/columns are relevant to an NLQ, instead of training a learned linker.

Where it is implemented:
- `nl2sql/schema.py:build_schema_summary` builds the full schema text.
- `nl2sql/agent_utils.py:build_schema_subset` optionally reduces the schema text based on keyword-to-table hints.

What the heuristic is:
- a small keyword -> table map (`_TABLE_HINTS`)
- plus join hints text appended to the reduced schema

Why you did it (defensible):
- schema linking is widely reported as a major Text-to-SQL bottleneck (survey papers).
- a smaller schema reduces wrong-table selection and reduces prompt length.
- a heuristic is auditable: you can show exactly why it picked a table.

Limitations:
- brittle to paraphrases not in the hint map
- can miss required tables -> the model may fail even if it could have succeeded with the full schema

---

## 8) Prompting (Baseline + Agent Prompts)

Baseline prompt builder:
- `nl2sql/prompting.py:make_few_shot_messages(schema, exemplars, nlq)`

Key choices in `SYSTEM_INSTRUCTIONS` (and why):
- "Output ONLY SQL" reduces prompt-echo and non-SQL (improves VA).
- "Exactly ONE statement starting with SELECT" reduces multi-statement output (VA).
- "Minimal projection" reduces extra columns and stabilizes EM/analysis.
- "Use LIMIT/ORDER BY only when NLQ implies ranking" reduces spurious ranking.
- "Use only tables/columns in schema" reduces hallucinations (schema linking failure).
- "Routing hints" are a compromise: not full answers, but nudges to reduce common join mistakes.

Agent prompt (used by the tool‑driven ReAct loop):
- `nl2sql/prompts.py:REACT_SYSTEM_PROMPT` defines the Thought/Action/Observation format.
- The user prompt is the running trace (user question + Action/Observation history).

Why a single system prompt?
- reduces prompt‑engineering degrees of freedom and keeps the loop easier to audit.

---

## 9) QLoRA / PEFT (What You Actually Did, Not Just a Buzzword)

Goal:
- improve NL->SQL generation by adapting a base model using parameter-efficient fine-tuning.

Where it is run:
- `notebooks/05_qlora_train_eval.ipynb`
- CLI mirror: `scripts/run_full_pipeline.py` (loads adapters via PEFT and evaluates using the same harness)

Key idea (Ding et al., 2023; QLoRA literature):
- Instead of full fine-tuning all weights, add small low-rank adapter matrices (LoRA).
- Keep base model quantized (4-bit) and train adapters.

Why it matters in your dissertation:
- It's a controlled comparison with in-context learning (Mosbach et al., 2023 discuss fair comparisons).
- It tests whether "learning the schema/task" improves beyond prompt engineering.

Limitations to say explicitly:
- Adapter quality depends heavily on training set quality and leakage control.
- QLoRA does not magically solve execution errors; it can still hallucinate columns or mis-join.

---

## 10) Results Files (What Gets Saved And How To Explain It)

Baseline / QLoRA eval outputs from `nl2sql/eval.py:eval_run`:
- JSON payload includes:
  - timestamp, k, seed, limit, n
  - va_rate, em_rate, ex_rate
  - per-item fields: nlq, gold_sql, raw_sql, pred_sql, va, em, ex, error

Agentic eval outputs (notebook):
- Usually saves:
  - pred_sql
  - trace (history of candidates/errors/observations)
  - metrics per item (VA/EM/EX/TS)
  - JSON save uses `default=str` to serialize Decimal values in TS debug samples.

Analysis helper:
- `scripts/analyze_results.py` can categorize EX failures (projection mismatch, join mismatch, etc.) with simple heuristics.

---

## 11) Examiner-Style Question Bank (Hard Mode)

Use these to practice. The answers below are intentionally "how you should say it" (defensive, precise, and grounded in code).

### 11.1 Execution Accuracy (EX)

Q: "You keep saying EX compares results. What exactly do you execute and how do you compare?"
A: "In `nl2sql/eval.py:execution_accuracy`, I execute both the predicted SQL and the gold SQL against the same base SQLAlchemy engine. Each query is run through `execute_fetch`, which uses `conn.execute(sqlalchemy.text(sql))` and then fetches up to `max_compare_rows`. If either execution fails, EX is 0. If both succeed, I compare the returned rows as a multiset using `collections.Counter` to ignore row order but preserve duplicates. In this repo, EX intentionally compares rows only and does not require matching column names, because projection/alias drift was dominating EX failures in early experiments."

Q: "Why Counter? Why not a set or just sort and compare?"
A: "A set would drop duplicates and can be wrong when the correct result has repeated rows. Sorting and comparing is fine but requires defining a stable ordering across mixed types; Counter is simpler and preserves multiplicity. The trade-off is that Counter assumes order doesn't matter; that matches the 'no ORDER BY, order unspecified' SQL semantics."

Q: "So could EX be wrong if the predicted query returns the right rows but in the wrong columns?"
A: "Yes. Because this comparator ignores column names, it could overcount cases where the correct rows are returned but the projection is wrong. That is a known limitation and part of why I also report EM and why I add projection constraints and TS checks."

Q: "What happens if the gold SQL fails?"
A: "EX returns 0 for that item. In `execution_accuracy`, if the gold query doesn't execute, I treat the item as not comparable and return False plus the gold error. That keeps the metric conservative rather than inflating EX by skipping hard items."

Q: "Do you cap result sizes? What if a query returns 2 million rows?"
A: "Yes, `execute_fetch` uses `fetchmany(max_rows + 1)` and treats results larger than the cap as a comparison failure. This prevents expensive comparisons and memory blowups. It's a pragmatic constraint for a dissertation harness."

Q: "Does your EX metric handle ORDER BY?"
A: "In the current implementation, EX is order-insensitive because it uses Counter. If ORDER BY matters, that is not captured by EX here; instead TS has an 'ordered compare' mode keyed off whether gold has ORDER BY."

### 11.2 VA and the QueryRunner (DB Execution)

Q: "How do you actually connect to the database from the notebook?"
A: "I use the Cloud SQL Python Connector to open authenticated connections to the Cloud SQL MySQL instance. In `nl2sql/db.py:create_engine_with_connector`, I define a `getconn()` function that returns a connector-managed DB-API connection, and pass it into `sqlalchemy.create_engine` via the `creator=` hook. That lets all other code use the standard SQLAlchemy execution API without handling sockets or credentials directly."

Q: "What is the QueryRunner, and why not just call engine.execute everywhere?"
A: "The QueryRunner is a small wrapper used as the agent's Act tool and as the VA signal. It centralizes two things: a safety check that blocks destructive SQL tokens, and consistent capture of execution success/error/columns and a small preview DataFrame for debugging. That keeps both the notebook and the agent loop simpler and safer."

Q: "How do you ensure the model cannot destroy your database?"
A: "I enforce a read-only policy at the executor level. Both `QueryRunner._safety_check` and the EX harness `_safety_check` block DDL/DML keywords like DROP/DELETE/UPDATE/INSERT. It is not a perfect SQL sandbox, but in a controlled ClassicModels setting it is a pragmatic safety layer."

Q: "What exactly is returned to the agent as observation?"
A: "In the tool-driven loop (`notebooks/03_agentic_eval.ipynb` + `nl2sql/agent_tools.py`), each tool call returns an Observation: schema text from `get_schema`, row previews or errors from `run_sql`, and validation/guardrail errors when cleanup fails. Those observations are appended to the running trace and fed back to the LLM on the next step."

Q: "Do you commit to a single transaction? Any isolation concerns?"
A: "For this dissertation harness, each query is executed independently in a short-lived connection. It's read-only SELECT execution. Isolation is not a focus, and consistency is assumed because the underlying dataset is static during evaluation."

### 11.3 TS (Test Suite Accuracy)

Q: "Why do you need TS if you already have EX?"
A: "EX checks equivalence on a single database instance. Zhong et al. (2020) argue that a single execution can be misleading because spurious queries can coincidentally match the gold result on one database. TS reduces that risk by comparing gold vs predicted across multiple perturbed replicas, acting as a semantic robustness check."

Q: "How do you decide ordered vs unordered result comparison?"
A: "In `nl2sql/eval.py:test_suite_accuracy_for_item`, I treat results as ordered only when the gold SQL contains ORDER BY. Otherwise I sort normalized row tuples and compare them. That matches the idea that ORDER BY is the only time row order is semantically defined."

Q: "What if the gold query fails on a TS replica?"
A: "With `strict_gold=True` I set TS to 0 because that replica isn't a valid semantic test; skipping broken gold can inflate TS. With `strict_gold=False` I can ignore replicas where gold fails, but I require at least one usable DB overall."

Q: "How do you create TS databases?"
A: "Externally: clone the base ClassicModels DB into multiple databases and apply controlled perturbations (e.g., scaling money amounts, shifting dates) while preserving constraints. The notebook only assumes the list of TS DB names exists."

### 11.4 ReAct Loop Design

Q: "How is your ReAct loop different from simply sampling 10 candidates and picking one?"
A: "The key difference is observation feedback and bounded reflection. I don't just sample and rerank; I execute candidates and use execution errors as observations. On error, I build a reflection prompt that includes schema, the bad SQL, and the error string, then generate a bounded number of reflections. That is execution-guided, not just best-of-N sampling."

Q: "Where is the 'act' step in ReAct?"
A: "The act step is `runner.run(sql)` in `nl2sql/query_runner.py`. It's called in the agent loop to gate candidates and generate observations (success/error)."

Q: "What stops the agent from looping forever?"
A: "The loop is explicitly bounded by `CFG['max_steps']` in the notebook, and reflection is also bounded (e.g., generate 4 reflections). This is deliberate for auditability and compute control."

Q: "Why not use a real SQL parser / constrained decoder instead of regex guards?"
A: "A full constrained decoder like PICARD is more principled, but it increases implementation scope and complexity. For an honors dissertation, I chose deterministic guards and lightweight parsing to keep the system explainable and to make each intervention traceable."

### 11.5 Postprocessing / Projection Contracts

Q: "Isn't postprocessing 'cheating'?"
A: "It would be cheating if it injected task answers or used gold information. Here it is deterministic and only enforces generic constraints implied by the NLQ (like minimal projection) and safety (single SELECT, no DDL/DML). It's closer to constrained decoding / output validation than to label leakage."

Q: "How do you decide when to strip ORDER BY?"
A: "I check the NLQ for ranking cues (top/highest/lowest/most/least/order/sort). If none are present, I remove ORDER BY/LIMIT. This was motivated by spurious ORDER BY causing unnecessary VA/EM instability."

Q: "What is a projection contract in your implementation?"
A: "In `nl2sql/agent_utils.py:enforce_projection_contract`, if the NLQ explicitly enumerates fields, I deterministically drop extra SELECT items and preserve the NLQ order. It constrains output shape without adding predicates or joins."

### 11.6 Schema Summaries / Schema Linking

Q: "How do you build the schema text that goes into the prompt?"
A: "In `nl2sql/schema.py:build_schema_summary`, I list tables and columns from INFORMATION_SCHEMA, then prioritize primary keys and name-like columns. The output is a line per table: `table(col1, col2, ...)`."

Q: "What do you mean by heuristic schema linking?"
A: "I mean a small, hand-written keyword-to-table mapping that reduces the schema text for a given NLQ. It's implemented in `nl2sql/agent_utils.py:build_schema_subset`. It's transparent and easy to audit, but not as robust as a learned schema linker."

### 11.7 QLoRA vs In-Context Learning

Q: "How do you know improvements come from QLoRA and not from prompt changes?"
A: "I keep the system prompt stable in `nl2sql/prompting.py` and reuse the same evaluation harness in `nl2sql/eval.py`. That way, differences between baseline and QLoRA runs are less likely to be caused by prompt drift."

Q: "Is your comparison fair?"
A: "I attempt to control for the evaluator and prompt format, and I avoid exemplar leakage by ensuring the evaluated item is not sampled as an in-context exemplar. There are still threats to validity, which I describe in limitations."

---

## 12) Literature Anchors (How To Cite Without Overclaiming)

Use these as "just enough" academic justification in a viva. They are paraphrases, not direct quotes.

- Yu et al. (2018) Spider:
  - established EM as a standard Text-to-SQL metric and popularized complex cross-domain evaluation
  - important for explaining why EM exists but is brittle

- Zhong et al. (2020) Distilled Test Suites:
  - motivate semantic evaluation beyond EM and single-DB execution
  - directly supports TS as a robustness check

- Yao et al. (2023) ReAct:
  - interleaves reasoning and acting with tool feedback
  - supports using DB execution as an "Act" step and keeping traces

- Scholak et al. (2021) PICARD:
  - constrained decoding by incremental parsing to enforce validity
  - supports the general principle of output constraints; this project uses simpler deterministic guards

- Zhai et al. (2025) ExCoT:
  - execution feedback improves Text-to-SQL reasoning/reflection
  - supports the idea of error-guided retries / reflection prompts

- Ding et al. (2023) PEFT survey:
  - supports LoRA/QLoRA as a pragmatic fine-tuning approach when full fine-tuning is expensive

- Mosbach et al. (2023) few-shot fine-tuning vs ICL:
  - supports being careful about "fair comparison" framing between prompting and fine-tuning

---

## 13) What To Memorize (If You Want A 10/10 Answer Under Pressure)

EX in one sentence (implementation-true):
- "I compute EX by executing both predicted and gold SQL on the same engine and comparing the returned row tuples as a multiset using Counter, with a row cap and safety checks."

DB execution in one sentence:
- "I connect via Cloud SQL Connector, wrap it in a SQLAlchemy Engine using the creator hook, then run SELECT-only queries through a QueryRunner that returns success/errors and a preview."

ReAct in one sentence:
- "The agent runs a bounded Thought → Action → Observation loop where the LLM selects tools, Python executes them, and execution errors drive `repair_sql` before `finish`."

TS in one sentence:
- "TS runs gold and predicted SQL across multiple perturbed DB replicas and only passes if they match everywhere, making the evaluation more robust than single-DB execution."
