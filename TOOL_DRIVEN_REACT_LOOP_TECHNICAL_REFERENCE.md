# Tool-Driven ReAct Loop: Technical Reference (Short)

**Version:** February 2026  
**Authoritative implementation:** `notebooks/03_agentic_eval.ipynb` (function `react_sql`) + `nl2sql/agent_tools.py`  
**Supporting modules:** `nl2sql/prompts.py`, `nl2sql/agent_utils.py`, `nl2sql/postprocess.py`, `nl2sql/query_runner.py`, `nl2sql/eval.py`

> Bracketed reference numbers (e.g., **[16]**) correspond to `REFERENCES.md`.

## Overview

The project's reference evaluation agent is a bounded **Thought -> Action -> Observation** loop in the ReAct style **[16]**:

- The LLM emits **structured tool calls** (`Action: tool_name[json_args]`).
- Python executes the tool and appends an **Observation** back into the transcript.
- Critical steps are **gated** (cannot execute without validation; cannot finish without a successful run).
- Failures trigger **forced recovery** (`repair_sql`) inspired by execution-feedback repair patterns (e.g., ExCoT **[2]** and execution-guided decoding **[25]**).
- Every step is logged (trace + decision log) to support auditability (traceability principles surveyed by Xi et al. **[21]**).

**Important:** this is **not** `ReactSqlAgent` in `nl2sql/agent.py`. That class is retained for comparison/ablations; the notebook `react_sql` loop is treated as the authoritative evaluation loop.

## Contract (Prompt + Parser)

**System prompt source:** `nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)

The model is instructed to emit:

```text
Thought: <free-form reasoning>
Action: tool_name[{"arg": "value"}]
```

The loop only parses/enforces `Action` (not `Thought`). The notebook parser uses a regex (`_ACTION_RE`) and selects the **last** `Action:` if the model emits more than one.

## Tools (Action Space)

**Primary definitions:** `nl2sql/agent_tools.py`  
**Notebook dispatcher:** `notebooks/03_agentic_eval.ipynb` (tool map + action branches)

Core actions:
- `get_schema`: retrieve full schema (tables/columns/PK/FK)
- `link_schema`: prune schema context before generation and rank columns within selected tables (RESDSQL-style separation of linking vs generation **[17]**; relation-aware schema linking in RAT-SQL **[22]**). Returns **`link_debug`** with selected tables, column scores, and value hints to make linking decisions auditable.
- `extract_constraints`: extract structural requirements (`agg`, `needs_group_by`, `needs_order_by`, `limit`, `distinct`) plus **value hints**, **value‑column hints**, **projection hints**, and **location cues/tables** for lightweight value linking (BRIDGE **[23]**)
- `generate_sql`: generate a raw SQL candidate (guardrails run immediately after)
- `validate_sql`: check "single executable SELECT" + schema references
- `validate_constraints`: post-hoc structural checks (PICARD-style *constraint idea* **[13]**, implemented as validation + repair); also enforces value-hint presence and location cues (including requiring a location table when needed) **[23]**
- `run_sql`: execute via `QueryRunner` (SELECT-only safety; execution feedback aligns with ExCoT **[2]** and execution-guided decoding **[25]**)
- `repair_sql`: revise SQL using the most recent error feedback
- `finish`: terminate (gated; see below)

## Required Order (Policy) + Enforcement (Code)

**Prompt policy order:** `get_schema` -> `link_schema` -> `extract_constraints` -> `generate_sql` -> `validate_sql` -> `validate_constraints` -> `run_sql` -> `finish`.

**Notebook enforcement (gates/overrides):**
- `run_sql` is blocked unless **both** `validate_sql` and `validate_constraints` have passed for the current `last_sql`.
- `finish` is blocked unless the most recent `run_sql` succeeded.
- If `constraints` are missing, the loop forces `extract_constraints` (unless repairing).
- If there is a pending validation/execution/intent failure, the loop forces `repair_sql`.
- If guardrails reject a candidate, the loop forces **one** bounded re-generate (`MAX_CLEAN_REJECT_RETRIES = 1`).

## Deterministic Guardrails (Between LLM and DB)

Guardrails are applied immediately after `generate_sql` and `repair_sql` in the notebook via `_apply_guardrails(...)`:

1. `clean_candidate_with_reason` (`nl2sql/agent_utils.py`)  
   Drops prompt echo / markdown / junk; keeps a single executable `SELECT`.
2. `guarded_postprocess` (`nl2sql/postprocess.py`)  
   Deterministic cleanup (e.g., strip spurious ORDER/LIMIT when NLQ does not request ranking).
3. `enforce_projection_contract` (`nl2sql/agent_utils.py`)  
   Output-shape control when the NLQ explicitly enumerates fields.
4. `_canonicalize_table_casing` (notebook helper)  
   Normalizes table casing for trace readability (schema checks are case-insensitive).

## State + Logging (Audit Trail)

The notebook loop maintains:
- `history`: transcript shown to the model (user + actions + observations)
- `trace`: raw model outputs + executed actions + observations (step-by-step)
- `decision_log`: compact per-step summary (what happened and why)

Key control state:
- `last_sql`, `last_valid`, `last_constraints_ok`, `last_run`, `last_error`
- `constraints`
- `pending_repair_error`, `pending_force_generate`, `clean_reject_retries`

**Post-execution intent check:** after a successful `run_sql`, the loop applies `intent_constraints(nlq, sql)` from `nl2sql/agent_utils.py`. This is a heuristic intent/shape check (SQL-structure based), and intent mismatch triggers repair.

## Evaluation (After the Loop Returns SQL)

Metrics are computed outside the loop in `nl2sql/eval.py`:
- **VA**: does the predicted SQL execute?
- **EX**: execution accuracy (row-equivalence vs gold)
- **TS**: suite-based robustness across perturbed replicas (Zhong et al. **[18]**; implemented here as replicas, not full distilled test suites)
- **EM**: exact match (diagnostic only)
