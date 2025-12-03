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

## Training/Evaluation Hooks
- QLoRA SFT: train on curated NLQ-SQL pairs (schema-aware text fields) with 4-bit quantization and LoRA adapters; log VRAM/time.
- Metrics: VA (syntax), EX (execution vs gold), TS (distilled DB consistency). Distilled DBs are schema-identical variants with altered data.
- Tracing: save per-turn traces (Thought/Action/Observation) for interpretability and dissertation artifacts.
