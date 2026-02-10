# Agent Design (Execution Infrastructure)

This document describes the tool-driven ReAct component as infrastructure for robust evaluation, not as the main research contribution.

## Role in Dissertation

- Primary contribution: prompting and fine-tuning comparisons.
- Agent contribution: make execution failures explicit, enforce safe tool order, and improve validity stability.

## Core Design Rules

- Setup once per question: schema retrieval and optional schema linking.
- Generate one SQL candidate.
- Apply deterministic cleanup.
- Validate before execution.
- Repair only on validation/execution failure.
- Stop at first successful execution.

## Tool Surface

- `get_schema`
- `link_schema`
- `extract_constraints`
- `generate_sql`
- `validate_sql`
- `validate_constraints`
- `run_sql`
- `repair_sql`
- `finish`

Implementation source: `nl2sql/agent_tools.py`.

## Enforcement Layer

The pipeline blocks unsafe or out-of-order transitions:
- no execution before validation
- no finish before successful execution
- no write/delete SQL at runtime

Implementation source:
- `nl2sql/react_pipeline.py`
- `nl2sql/query_runner.py`

## Minimal by Default, Ablations Optional

Default configuration is now a minimal core loop (`react_core`).
Optional ablations (intent gating, extra repairs) are kept for explicit, named experiments only.

## Logging and Audit

Each run stores:
- tool trace
- per-item outcomes
- config snapshot
- run metadata and dataset signature

This keeps failure attribution inspectable and supports examiner-facing reproducibility.

## Where It Lives

- Core loop: `nl2sql/react_pipeline.py`
- Tool implementation: `nl2sql/agent_tools.py`
- Prompt contract: `nl2sql/prompts.py`
- Notebook orchestration: `notebooks/03_agentic_eval.ipynb`
