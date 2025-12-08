# ARCHITECTURE

## Overview
- Goal: ReAct-based text-to-SQL system targeting the classicmodels MySQL schema, using an open-source LLM (Meta-Llama-3-8B-Instruct) with QLoRA adapters.
- Agent loop: Thought → Action (SQL via tool) → Observation (QueryRunner output/error) → Refined Thought. Iterates until a stopping condition or maximum turns.
- Tools: `QueryRunner` exposes a guarded `run(sql, params)` interface; additional helper tools (schema cache loader, prompt builder) can be added as simple callables returning JSON/text.

## Components
- LLM Backbone: `meta-llama/Meta-Llama-3-8B-Instruct` loaded in 4-bit for QLoRA; tokenizer aligned to model.
- Prompting:
  - Few-shot baseline prompt (schema + table descriptions + NLQ + 2–4 exemplars) for pre-finetune evaluation.
  - ReAct prompt template includes: question, schema context, available tools, previous Thought/Action/Observation history, and safety note (read-only).
- QueryRunner (Act/Tool):
  - Safety: rejects destructive tokens (DROP/DELETE/TRUNCATE/ALTER/CREATE).
  - Metadata: `success`, `rowcount`, `exec_time_s`, `error`, `columns`, `result_preview` (truncated), timestamp.
  - History: append-only for later evaluation and TS scoring.
- Why it matters: QueryRunner is the bridge between the LLM’s SQL and the database. It enforces read-only safety, provides an audit trail (history/save_history), and produces the Observation (errors, row counts, previews) that the LLM uses to self-correct in ReAct. It is a thin, reproducible wrapper over SQLAlchemy + Cloud SQL connector, inspired by safe DB executor patterns—not a third-party agent lib.
- Observation Handling:
  - On error: include error message and prior SQL in the next prompt section for self-correction.
  - On success: include row count and preview to validate semantic correctness.

## Data Flow
1) Input: Natural language question.
2) Context assembly: schema cache + table descriptions + (optionally) few-shot exemplars.
3) LLM Thought: reason about intent and plan query.
4) Action: emit SQL string (read-only).
5) QueryRunner executes SQL → Observation (result/error).
6) LLM refines using Observation; repeat or stop.
7) Outputs: final SQL, execution metadata, reasoning trace.

## Context Assembly, Thought, Refinement
- Context assembly: gather schema/table blurbs (via list_tables/get_table_columns or cached JSON) and, for few-shot, 2–4 NLQ→SQL exemplars. Build a prompt that includes schema, brief table descriptions, the NLQ, and tool description (for ReAct).
- Thought: the model reasons over the assembled context to plan the query (tables, joins, filters).
- Action/Observation: emit SQL, run through QueryRunner; capture success/error, rowcount, columns, preview. Feed this back.
- Refinement: include errors or mismatched results in the next prompt turn so the LLM can adjust column names, joins, filters, or add/remove conditions. Iterate until stopping criteria are met.

## Training/Evaluation Hooks
- QLoRA SFT: train on curated NLQ-SQL pairs (schema-aware text fields) with 4-bit quantization and LoRA adapters; log VRAM/time.
- Metrics: VA (syntax), EX (execution vs gold), TS (distilled DB consistency). Distilled DBs are schema-identical variants with altered data.
- Tracing: save per-turn traces (Thought/Action/Observation) for interpretability and dissertation artifacts.
