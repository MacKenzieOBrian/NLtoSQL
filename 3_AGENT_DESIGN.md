# Agent Design (Tool-Driven ReAct Loop)

This file is a short design overview. The authoritative evaluation loop is defined in `notebooks/03_agentic_eval.ipynb` (`react_sql`) and uses explicit tools in `nl2sql/agent_tools.py` under the system prompt contract in `nl2sql/prompts.py`.

For a code-truth specification (tool order, gates/overrides, state variables, and logging), see `TOOL_DRIVEN_REACT_LOOP_TECHNICAL_REFERENCE.md`.

---

## What the Agent Does

- **Input:** natural-language question (NLQ) + ClassicModels schema context.
- **Output:** exactly one executable MySQL `SELECT` statement.
- **Method:** bounded **Thought → Action → Observation** loop where the model emits tool calls and Python executes them.

---

## Tool Actions (What Exists and Why)

- `get_schema`: ground the model in real tables/columns.
- `link_schema`: reduce schema scope before generation to reduce wrong joins.
- `extract_constraints`: infer structural needs (aggregation, grouping, ordering, limit, distinct) from the NLQ.
- `generate_sql`: propose a SQL candidate using the focused schema + constraints.
- `validate_sql`: catch formatting/schema-reference issues before execution.
- `validate_constraints`: enforce NLQ-implied structure before execution.
- `run_sql`: execute safely and return the key observation (success/error + preview rows).
- `repair_sql`: fix failed SQL using the latest error feedback.
- `finish`: terminate only after a successful execution.
- `get_table_samples` (optional): provide example rows for ambiguous column usage.

---

## Control Layer (Deterministic Guardrails)

Guardrails run immediately after `generate_sql` and `repair_sql` in the notebook:

- `clean_candidate_with_reason` (single-statement SELECT-only cleaning)
- `guarded_postprocess` (deterministic cleanup)
- `enforce_projection_contract` (output-shape alignment when NLQ enumerates fields)
- table-casing normalization (readability; avoids confusing traces)

Safety is enforced at execution time via `QueryRunner` (SELECT-only + forbidden token blocklist).

---

## Loop Guarantees (Enforced in the Notebook)

- `run_sql` is blocked unless `validate_sql` and `validate_constraints` have both passed.
- `finish` is blocked unless the most recent `run_sql` succeeded.
- Validation/execution/intent failures force `repair_sql` on the next step.
- Guardrail rejection forces one re-generate (bounded retry).
- Missing constraints force `extract_constraints` (unless repairing).

This makes failures attributable to a specific step (validation vs constraints vs execution) rather than being hidden by post-hoc ranking.

---

## Where to Demo / Verify

- Notebook loop: `notebooks/03_agentic_eval.ipynb` (cell “Tool-driven ReAct loop (explicit Thought/Action/Observation)”)
- Tools: `nl2sql/agent_tools.py`
- Prompt contract: `nl2sql/prompts.py`
- Safety gate: `nl2sql/query_runner.py`
- Metrics: `nl2sql/eval.py`

---

## Legacy / Experimental Code

`nl2sql/agent.py` (`ReactSqlAgent`) is retained for ablations and comparison; it is not the reference evaluation loop.
