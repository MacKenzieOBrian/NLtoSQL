# Tool-Driven ReAct Loop: Technical Reference

**Version:** February 2026  
**Authoritative implementation:** `notebooks/03_agentic_eval.ipynb` (function `react_sql`) + `nl2sql/agent_tools.py`  
**Supporting modules:** `nl2sql/prompts.py`, `nl2sql/agent_utils.py`, `nl2sql/postprocess.py`, `nl2sql/query_runner.py`, `nl2sql/eval.py`  
**Literature foundation:** ReAct [16], ExCoT [2], Ojuri et al. [10], RESDSQL [17], PICARD [13], Xi et al. [21]  

> **Note:** Bracketed reference numbers (e.g., **[16]**) correspond to `REFERENCES.md`. This document describes the reference evaluation implementation, not the experimental `ReactSqlAgent` class in `nl2sql/agent.py`.

---

## 1. Overview

This document specifies the **reference ReAct implementation** used for evaluation in this dissertation. The loop implements a **Thought → Action → Observation** cycle in the ReAct style [16], with the following defining properties:

### 1.1 Core Design Features

1. **Explicit tool calls**  
   Actions are structured outputs (`Action: tool_name[json_args]`) emitted by the LLM and executed by Python.

2. **Enforced ordering (critical steps)**  
   The expected sequence is defined in the system prompt, and the notebook loop enforces the critical safety/correctness gates (validation, execution, finishing) with code-level overrides.

3. **Forced recovery on failure**  
   Validation and execution failures set a pending error and trigger forced repair, inspired by execution-feedback repair patterns (e.g., ExCoT) [2].

4. **Full traceability**  
   Every step is logged (trace + decision log) to support error attribution and auditability, consistent with agent traceability principles surveyed by Xi et al. [21].

### 1.2 Literature Positioning (Conceptual Mapping)

| System feature | Literature basis | What we implement (code-truth) |
| --- | --- | --- |
| Action/observation loop | ReAct [16] | Model emits `Action: ...[...]`; Python executes and appends `Observation: ...` |
| Execution-guided repair | ExCoT [2] | Execution/validation errors force `repair_sql` until resolved |
| Schema linking before generation | RESDSQL [17] | `link_schema` produces a focused schema text for generation |
| Structural constraint enforcement | PICARD [13] | Post-hoc `validate_constraints` + forced repair (not beam-time constrained decoding) |
| Logging and auditability | Xi et al. [21] | Full `trace` + compact `decision_log` + compliance checks |

---

## 2. System Prompt Contract

**Source:** `nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)

### 2.1 Required Output Format

The LLM is instructed to respond using:

```
Thought: <free-form reasoning>
Action: tool_name[{"arg": "value"}]
Observation: <tool output>
```

**Implementation note:** the loop parses and enforces only `Action`. `Thought` is included for interpretability and alignment with ReAct conventions [16], but is not validated.

### 2.2 Expected Tool Order (Policy)

The system prompt specifies the following expected sequence:

1. `get_schema` - retrieve full database schema  
2. `link_schema` - prune schema to relevant tables/columns (schema-linking before generation) [17]  
3. `extract_constraints` - infer structural requirements from the NLQ  
4. `generate_sql` - propose SQL candidate  
5. `validate_sql` - check formatting + schema references  
6. `validate_constraints` - verify structural intent (post-hoc constraint validation) [13]  
7. `run_sql` - execute query (execution feedback / oracle signal) [2]  
8. `finish` - terminate after successful execution  

Optional tool: `get_table_samples` (example rows for grounding).

### 2.3 Code-Level Enforcement (Gates and Overrides)

The notebook loop enforces critical rules in addition to the prompt policy:

| Gate / override | Condition | Effect (code) |
| --- | --- | --- |
| **Finish gate** | `last_run` missing or not successful | Block `finish` with an error observation |
| **Run gate** | `last_valid` or `last_constraints_ok` not `True` | Block `run_sql` with an error observation |
| **Forced repair** | `pending_repair_error` is set | Override requested action to `repair_sql` |
| **Forced constraint extraction** | `constraints` is `None` and action not in (`extract_constraints`, `repair_sql`) | Override action to `extract_constraints` |
| **Forced regenerate (bounded)** | Guardrails reject output and retry budget remains | Override action to `generate_sql` once (`MAX_CLEAN_REJECT_RETRIES=1`) |

**Practical nuance:** “enforced ordering” is strongest for the critical gates above. If the LLM proposes an out-of-order action, the loop either overrides it (forced repair / forced constraints / forced regen) or returns an error observation (run/finish gates) so the model can recover on the next step.

---

## 3. Tools (Action Space)

**Primary definitions:** `nl2sql/agent_tools.py`  
**Notebook dispatch:** `notebooks/03_agentic_eval.ipynb` (`TOOLS` mapping + action branches)

Each tool follows the pattern `(state, args) -> observation`. The notebook loop is **stateful**: many tools operate on shared state (`last_sql`, `constraints`, etc.) and may ignore user-provided args.

### 3.1 Tool Catalog (Code-Truth)

| Tool | Inputs (effective) | Output | Purpose |
| --- | --- | --- | --- |
| `get_schema()` | none | structured schema `dict` | Ground the agent in real tables/columns |
| `link_schema(nlq, schema_text, max_tables)` | NLQ + full schema text | `{schema_text, changed}` | Reduce schema scope before generation [17] |
| `extract_constraints(nlq)` | NLQ | `dict` | Infer structural requirements (agg/order/group/limit/distinct) |
| `generate_sql(nlq, schema_text, constraints)` | NLQ + focused schema text + constraints | raw SQL text | Generate candidate SQL |
| `validate_sql(sql, schema_text)` | SQL + schema text | `{valid, reason}` | Catch clean/schema-reference failures pre-execution |
| `validate_constraints(sql, constraints)` | SQL + constraints | `{valid, reason}` | Enforce NLQ-implied structure (post-hoc) [13] |
| `run_sql(sql)` | SQL | `{success, rows, ...}` or `{success: False, error}` | Execute SQL safely; return exec feedback [2] |
| `repair_sql(nlq, bad_sql, error, schema_text)` | NLQ + failed SQL + error + schema text | raw SQL text | Repair using error feedback [2] |
| `get_table_samples(table, n)` | table name | `list[dict]` | Optional grounding by example rows |
| `finish(answer, sql, provenance)` | last run rows (stringified) + final SQL + metadata | `dict` | Terminal action container |

**Helper (not a tool action):** `schema_to_text(schema)` formats schema into prompt text. It is used in bootstrap but is not part of the tool list in `REACT_SYSTEM_PROMPT`.

### 3.2 Constraint Dictionary (Exact Fields)

`extract_constraints(nlq)` returns:

- `agg`: `"COUNT" | "AVG" | "SUM" | "MAX" | "MIN" | None`
- `needs_group_by`: `bool`
- `needs_order_by`: `bool`
- `limit`: `int | None`
- `distinct`: `bool`

---

## 4. Guardrails (Deterministic Control Layer)

Guardrails run immediately after `generate_sql` and `repair_sql` via `_apply_guardrails(raw_sql, nlq, schema_text)` in the notebook.

### 4.1 Guardrail Pipeline

1. `clean_candidate_with_reason(raw_sql)` (from `nl2sql/agent_utils.py`)  
   Keeps a single executable `SELECT ... FROM ...;` and rejects prompt echo / multi-statement / junk.

2. `guarded_postprocess(sql, nlq)` (from `nl2sql/postprocess.py`)  
   Deterministic cleanup (e.g., strip ORDER/LIMIT when NLQ does not ask for ranking; normalize surface noise).

3. `enforce_projection_contract(sql, nlq)` (from `nl2sql/agent_utils.py`)  
   If NLQ enumerates fields, deterministically prunes extra SELECT items and preserves NLQ order (output-shape control).

4. `_canonicalize_table_casing(sql, schema_text_full)` (notebook helper)  
   Normalizes table casing for readability; schema checks are case-insensitive.

### 4.2 Guardrail Outcomes (Notebook Semantics)

| Outcome | How it is represented | Next control effect |
| --- | --- | --- |
| Accept | returns `(sql, None)` | sets `last_sql`, resets validation state |
| Reject | returns `(\"\", \"clean_reject:<reason>\")` | sets `pending_force_generate` and forces one re-generate (bounded) |

---

## 5. Loop Parameters (Notebook)

Configured in the evaluation cell in `notebooks/03_agentic_eval.ipynb`:

```python
REACT_MAX_STEPS = 8
REACT_MAX_NEW_TOKENS = 256
REACT_DO_SAMPLE = False
REACT_TEMPERATURE = 0.2
REACT_TOP_P = 0.9
USE_LINK_SCHEMA = True
MAX_CLEAN_REJECT_RETRIES = 1
```

**Design rationale (code-facing):**

- `REACT_MAX_STEPS` bounds iteration for cost and auditability.
- `REACT_DO_SAMPLE=False` makes the loop deterministic by default (temperature/top_p only matter when sampling is enabled).
- `USE_LINK_SCHEMA` enables schema pruning; setting it false reduces (but does not perfectly eliminate) pruning in practice.
- `MAX_CLEAN_REJECT_RETRIES=1` prevents infinite regen loops on junk outputs.

---

## 6. State Variables (Notebook Loop)

Maintained across steps (see `react_sql` in `notebooks/03_agentic_eval.ipynb`):

```python
history: list[str]                 # ReAct transcript shown to the model
trace: list[dict]                  # full step-by-step execution log
decision_log: list[dict]           # compact decision summary per step

last_sql: str | None               # most recent cleaned SQL candidate
last_error: str | None             # most recent failure reason
last_run: dict | None              # most recent run_sql observation

last_valid: bool | None            # validate_sql status for current last_sql
last_constraints_ok: bool | None   # validate_constraints status for current last_sql
constraints: dict | None           # extracted constraints for this NLQ

pending_repair_error: str | None   # triggers forced repair_sql
pending_force_generate: str | None # triggers forced generate_sql (bounded)
clean_reject_retries: int          # retry counter for guardrail rejections
```

---

## 7. Execution Flow (Reference Implementation)

### 7.1 Bootstrap (Before Step 0)

1. Load full schema:
   - `schema = get_schema()`
   - `schema_text_full = schema_to_text(schema)`
2. Initialize `history`:
   - `User question: <nlq>`
   - `Action: get_schema[{}]`
   - `Observation: <schema_text_full>`
3. Apply schema linking once:
   - `link_obs = link_schema(nlq, schema_text_full, max_tables=6 if USE_LINK_SCHEMA else 0)`
   - `schema_text_focus = link_obs["schema_text"] or schema_text_full`
   - Append `Action: link_schema[...]` + `Observation: <schema_text_focus>`
4. Initialize `constraints = None` (forced early if missing).

### 7.2 Main Loop (Steps 0..max_steps-1)

For each step:

1. Build prompt: `prompt = "\n".join(history)`
2. Call LLM: `llm_out = _call_react_llm(prompt)`
3. Parse action: `action, args = _parse_action(llm_out)` (last `Action:` wins)
4. Apply overrides:
   - pending repair forces `repair_sql`
   - pending regen forces `generate_sql`
   - missing constraints forces `extract_constraints` (unless repairing)
5. Dispatch tool branch; append `Observation: ...` to `history`; append to `trace` and `decision_log`
6. Enforce terminal semantics:
   - `finish` only permitted after successful `run_sql`

### 7.3 Tool Semantics (Notebook-Specific)

- `generate_sql` / `repair_sql`:
  - guardrails run immediately
  - on accept: sets `last_sql`, clears pending error flags, resets `last_valid`/`last_constraints_ok`
  - on reject: sets `pending_force_generate` (bounded)
- `validate_sql` / `validate_constraints`:
  - on failure: sets `pending_repair_error` and `last_error`
- `run_sql`:
  - blocked unless both validations passed
  - on exec failure: sets `pending_repair_error`
  - on exec success: applies `intent_constraints(nlq, last_sql)` (SQL-structure heuristic); intent mismatch is treated as a failure and triggers repair

### 7.4 Exit + Fallback

If `finish` is not reached within `REACT_MAX_STEPS`:

- If `schema_summary` is available: attempt deterministic `vanilla_candidate(...)` fallback (`nl2sql/agent_utils.py`).
- Else: return `last_sql` (or empty).

---

## 8. Logging and Compliance

### 8.1 Trace Log (`trace`)

Records:

- raw LLM output: `{"step": i, "llm": <text>}`
- tool execution: `{"step": i, "action": <tool>, "args": <dict>, "observation": <obs>}`
- forced action overrides: `{"step": i, "forced_action": <tool>, "requested_action": <tool>, "reason": <str>}`

### 8.2 Decision Log (`decision_log`)

Compact entries like:

```python
{"step": i, "decision": "validate_sql", "reason": "unknown_column:orders.foo", "status": "reject", "data": {...}}
```

### 8.3 Compliance Summary (`summarize_trace`)

Flags ordering violations such as:

- `generate_without_constraints`
- `run_without_validate`
- `run_without_validate_constraints`
- `finish_without_run`

---

## 9. Evaluation (Post-Loop)

After the loop returns final SQL, evaluation is performed separately via `nl2sql/eval.py`:

| Metric | Definition | Notes |
| --- | --- | --- |
| VA | executability (does it run?) | derived from execution outcome |
| EM | exact match | diagnostic only |
| EX | execution accuracy | compares result rows (row-equivalence) |
| TS | suite-based robustness | executes across perturbed replicas [18] |

TS is implemented as a **suite-based** robustness check across perturbed DB replicas, not full “distilled test suite” construction in the strongest Zhong et al. sense [18].

---

## 10. Relationship to `ReactSqlAgent` (Experimental Class)

`nl2sql/agent.py` (`ReactSqlAgent`) implements a separate candidate-driven loop (generate multiple candidates, score/rank, optional reflection). It is useful for ablations and legacy comparisons but is **not** the authoritative evaluation loop.

**Authoritative implementation:** the tool-driven loop defined in `notebooks/03_agentic_eval.ipynb` (`react_sql`) backed by `nl2sql/agent_tools.py`.
