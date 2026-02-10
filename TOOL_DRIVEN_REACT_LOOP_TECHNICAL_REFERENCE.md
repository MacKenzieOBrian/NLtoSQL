# Tool-Driven ReAct Loop: Technical Reference

Version: February 2026  
Implementation source of truth: `nl2sql/react_pipeline.py` and `nl2sql/agent_tools.py`

## 1. Purpose in This Dissertation

This loop is execution infrastructure. It provides bounded tool use, validation gates, and error-attribution traces. It is not the primary mechanism for dissertation claims about semantic improvement.

Literature grounding:
- ReAct action-observation control: `REFERENCES.md#ref-yao2023-react`
- Execution-guided rejection/repair framing: `REFERENCES.md#ref-wang2018-eg-decoding`, `REFERENCES.md#ref-zhai2025-excot`

## 2. Default Configuration (Core)

`react_core` in `nl2sql/react_pipeline.py`:
- `use_schema_link=True`
- `use_constraint_policy=True`
- `use_repair_policy=True`
- `use_intent_gate=False`
- `stop_on_first_success=True`
- `max_repairs=1`

Interpretation:
- repair is triggered by validation/execution failures,
- no semantic wandering after first successful execution,
- complexity remains bounded and auditable.

## 3. Demonstration Ablations

Default ablation plan is intentionally minimal:
- `react_core`
- `react_no_repair`
- `react_extra_repair`

Purpose:
- isolate repair contribution,
- test whether extra repair depth adds value,
- avoid over-claiming from feature-heavy variants.

## 4. Tool Order and Gates

Required order:
1. `get_schema`
2. `link_schema` (if enabled)
3. `extract_constraints` (if enabled)
4. `generate_sql`
5. deterministic cleanup
6. `validate_sql`
7. `validate_constraints`
8. `run_sql`
9. `finish`

Gate policy:
- no execution before validation passes,
- no finish before successful run,
- repair loop only on failure paths.

## 5. Literature Link per Tool Stage

| Stage | Why needed | Literature anchor |
| --- | --- | --- |
| Schema + linking | reduce wrong-table/join errors | `ref-wang2020-ratsql`, `ref-li2023-resdsql` |
| Constraint extraction + validation | reduce structurally wrong but executable SQL | `ref-scholak2021-picard`, `ref-zhai2025-excot` |
| Execution + repair | convert runtime failures into corrective feedback | `ref-wang2018-eg-decoding`, `ref-yao2023-react` |

## 6. Reproducibility Metadata

Each evaluation report includes:
- UTC timestamp,
- seed (if provided),
- config snapshot,
- dataset signature hash,
- run metadata (optional).

This supports repeat-run auditing and exact rerun conditions.

## 7. Safety Model

- SELECT-only execution policy
- forbidden destructive token blocklist
- bounded query result previews

Safety implementation:
- `nl2sql/query_runner.py`
- `nl2sql/eval.py` safety checks

## 8. Failure Attribution

Per-item output contains:
- `trace` (step-level tool activity)
- validation and execution outcomes
- SQL strings after cleanup/repair

These traces are designed to answer where and why failures occur.

## 9. Relationship to Evaluation Metrics

Loop output is scored externally via shared evaluators:
- VA: executability
- EX: base-db semantic equivalence
- TS: suite-db semantic robustness
- EM: diagnostic exact match

Metric code:
- `nl2sql/eval.py`

## 10. Non-Goals

- not a free-form autonomous agent
- not an unrestricted planner
- not an SOTA optimization benchmark

Design choice: keep complexity low so behavior stays explainable and ablatable.
