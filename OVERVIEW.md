# Codebase Overview (Current State)

This file is a technical map of the repository as it exists right now: what each module does, how experiments are run, and exactly how evaluation scores are computed.

---

## Repo Layout

Top-level docs (dissertation):
- `1_LITERATURE.md` to `6_LIMITATIONS.md`: canonical chapter/log files (each decision uses the same 4-part format).
- `LOGBOOK.md`: chronological lab-notebook style notes (runs, dates, trace observations).
- `REFERENCES.md`: bibliography list.
- `9.AI_PROMPTS.md`: AI-tool usage disclosure.
- `DEMO_NOTEBOOK_03_AGENTIC.md`: cell-by-cell speaking guide for the agentic notebook.

Core code and experiment runners:
- `nl2sql/`: reusable Python package (prompting, schema extraction, execution, evaluation, agent helpers).
- `notebooks/`: Colab notebooks (baseline prompting, training-set validation, QLoRA training/eval, agentic eval).
- `scripts/`: CLI runners and utilities (pipeline runner, TS helpers, results analysis, notebook cleanup).
- `data/`: benchmark + training files.
- `results/`: JSON outputs, adapter checkpoints, agent traces.
- `requirements.txt`: pinned runtime dependencies (Colab-first).

---

## How Experiments Are Run (Entry Points)

Baseline prompting:
- Notebook: `notebooks/02_baseline_prompting_eval.ipynb`
- Calls: `nl2sql.eval.eval_run(...)` with `k=0` and `k=3`
- Writes JSON to: `results/baseline/...` (created on first run)

Training-set validation:
- Notebook: `notebooks/04_build_training_set.ipynb`
- Uses: `nl2sql.db.create_engine_with_connector(...)` to sanity-check DB access and data quality

QLoRA training and evaluation:
- Notebook: `notebooks/05_qlora_train_eval.ipynb`
- Trains adapters (TRL + PEFT), then evaluates using the same `nl2sql.eval.eval_run(...)` harness
- Writes JSON to: `results/qlora/...` (created on first run)
- Writes adapter artifacts to: typically under `results/adapters/...`

Agentic ReAct evaluation:
- Notebook: `notebooks/03_agentic_eval.ipynb`
- Implements the Stage-3 style ReAct loop in-notebook (helper layer + `react_sql`)
- Computes VA/EX/EM/TS and writes: `results/agent/results_react_200.json`

CLI alternative:
- `scripts/run_full_pipeline.py` mirrors the notebook flows for baseline + QLoRA eval and includes a small-slice ReAct sanity check.
- Important: the script ReAct loop is a simplified version; the main “Stage-3” loop you defend lives in `notebooks/03_agentic_eval.ipynb`.

---

## Data and Result Formats

Benchmark data:
- Test set: `data/classicmodels_test_200.json` (list of items with `nlq` and `sql`)
- Train set: `data/train/classicmodels_train_200.jsonl` (one JSON object per line)

Baseline/QLoRA eval outputs (from `nl2sql.eval.eval_run`):
- Saved JSON includes run metadata plus per-item results.
- Each item stores at least: NLQ, gold SQL, raw model SQL, postprocessed predicted SQL, VA/EM/EX, and errors.

Agentic eval output (from `notebooks/03_agentic_eval.ipynb` evaluation cell):
- Saved JSON includes: `va_rate`, `ex_rate`, `em_rate`, `ts_rate`
- Each item stores: NLQ, gold SQL, predicted SQL, VA/EM/EX/TS, TS debug info, and a structured trace (`trace`) from the ReAct loop.

---

## `nl2sql/` Package (Modules and Responsibilities)

### `nl2sql/db.py`
Purpose: create DB engines and safe connections.
- `create_engine_with_connector(...)`: Cloud SQL Connector + SQLAlchemy engine using a `creator` function.
- `safe_connection(engine)`: context manager for safe open/close of SQLAlchemy connections.

Used by: notebooks and scripts that need DB access.

### `nl2sql/schema.py`
Purpose: introspect schema and generate schema text for prompts.
- `build_schema_summary(engine, db_name, ...)`: produces `table(col1, col2, ...)` lines with PK/name-like columns prioritized.

Used by: baseline, QLoRA, and agent notebooks to ground the model in the actual DB schema.

### `nl2sql/prompting.py`
Purpose: consistent prompt format and few-shot message building.
- `SYSTEM_INSTRUCTIONS`: single authoritative system prompt.
- `make_few_shot_messages(schema, exemplars, nlq)`: builds schema + exemplars + NLQ into chat messages.

Used by: `nl2sql.eval.eval_run`.

### `nl2sql/llm.py`
Purpose: generation helpers.
- `extract_first_select(text)`: best-effort extraction of the first SELECT statement.
- `generate_sql_from_messages(model, tokenizer, messages, ...)`: deterministic generation (`do_sample=False`) for evaluation.

Used by: `nl2sql.eval.eval_run` and some notebook helpers.

### `nl2sql/postprocess.py`
Purpose: deterministic SQL cleanup to reduce evaluation noise.
- `normalize_sql(s)`: normalization for EM comparisons.
- `guarded_postprocess(sql, nlq)`: keep first SELECT, strip ORDER/LIMIT when not requested, drop ID-like columns unless requested, and apply minimal projection for “list all …” style NLQs.

Used by: baseline + QLoRA harness and parts of the agentic postprocess layer.

### `nl2sql/query_runner.py`
Purpose: safe SQL execution (the “Act” tool) and VA computation.
- `QueryRunner.run(sql, ...)`: SELECT-only guard, executes via SQLAlchemy, returns success/error and optional preview.
- Maintains a `history` list of `QueryResult` for debugging.

Used by: ReAct loop execution gate and evaluation VA computation.

### `nl2sql/eval.py`
Purpose: primary evaluation harness (VA/EM/EX) and TS harness.
- `eval_run(...)`: end-to-end evaluator for baseline and QLoRA experiments.
- `execution_accuracy(engine, pred_sql, gold_sql, ...)`: EX comparator (result equivalence).
- `test_suite_accuracy_for_item(...)`: TS comparator across multiple perturbed DBs (suite-based approximation).

Used by: notebooks and scripts for consistent scoring.

### `nl2sql/agent_utils.py`
Purpose: lightweight, explainable helpers used by agentic pipelines.
- Cleaning: `clean_candidate(raw)` (SELECT-only filter).
- Schema narrowing: `build_schema_subset(schema_summary, nlq)` (keyword-based).
- Output-shape constraints: `enforce_projection_contract(sql, nlq)` (only when NLQ explicitly enumerates fields).
- Intent gate: `intent_constraints(nlq, sql)` (aggregate/group/top-k structure checks).
- Reranking: `semantic_score(nlq, sql)`, `count_select_columns(sql)`.
- Repair scaffolding: `classify_error(err)`, `error_hint(kind, err)`.
- Fallback: `vanilla_candidate(...)` (deterministic few-shot baseline).

Used by: the agentic notebook and the CLI pipeline script.

---

## Evaluation (Deep Dive)

This repo reports four metrics: VA, EM, EX, TS.

### VA (Valid SQL / executability)
Definition:
- VA = 1 if the predicted SQL executes successfully; else 0.

Implementation:
- In the baseline/QLoRA harness: `nl2sql.eval.eval_run` executes `pred_sql` via `QueryRunner.run` and records `va = meta.success`.
- In the agentic evaluation notebook: VA is computed by `runner.run(pred_sql)` inside the per-item evaluation loop.

Key code:
- `nl2sql/query_runner.py` (`QueryRunner.run`)
- `nl2sql/eval.py` (`eval_run`)
- `notebooks/03_agentic_eval.ipynb` (cell `# 9) Full ReAct-style evaluation (VA/EX/EM/TS)`)

Why it exists:
- VA separates “could not run” failures from “ran but answered wrong” failures.

### EM (Exact Match, diagnostic only)
Definition:
- EM = 1 if normalized predicted SQL string equals normalized gold SQL string; else 0.

Implementation:
- Baseline/QLoRA harness: `em = normalize_sql(pred_sql) == normalize_sql(gold_sql)` (`nl2sql/postprocess.py`).
- Agentic notebook: `pred_clean = pred_sql.strip().rstrip(';').lower()` (and same for gold).

Key code:
- `nl2sql/postprocess.py` (`normalize_sql`)
- `nl2sql/eval.py` (`eval_run`)
- `notebooks/03_agentic_eval.ipynb` evaluation cell

Why it exists:
- EM is kept as a regression detector (postprocess changes can silently alter outputs). It is not treated as semantic correctness.

### EX (Execution Accuracy, base DB)
Definition:
- EX = 1 if predicted and gold SQL execute and return equivalent results on the base DB; else 0.

Implementation detail (what is compared):
- The current EX comparator compares **row multisets** using `Counter(pred_rows) == Counter(gold_rows)`.
- Column names are not required to match, but the returned row tuples must match as a multiset.

Key code:
- `nl2sql/eval.py` (`execution_accuracy`, `execute_fetch`)
- Agentic notebook calls `execution_accuracy(engine=engine, pred_sql=..., gold_sql=...)` only when VA=1.

Important guards:
- `execute_fetch` has its own destructive-token safety check.
- `max_compare_rows` prevents comparing arbitrarily large result sets.

Why it exists:
- SQL is non-unique; execution-based equivalence is a closer proxy for semantic correctness than EM.

### TS (Test-Suite Accuracy, perturbed DBs)
Definition:
- TS = 1 if predicted and gold queries match across all TS DB replicas in the suite; else 0.

Implementation detail (suite-based approximation):
- For each DB name in `suite_db_names`:
  1. Execute gold SQL and predicted SQL.
  2. If gold fails and `strict_gold=True`, mark TS as failed (suite DB not compatible with gold).
  3. Compare results:
     - If gold contains ORDER BY, comparison is ordered (exact list equality).
     - Otherwise comparison is unordered (rows sorted for a stable compare).
  4. Column-count mismatch is treated as failure.

Key code:
- `nl2sql/eval.py` (`test_suite_accuracy_for_item`, `_run_select_ts`, `_results_match_ts`)
- TS engines are created by a notebook engine-factory function (`make_engine_fn`) in `notebooks/03_agentic_eval.ipynb`.

Why it exists:
- Single-DB EX can be “lucky”. TS tests robustness by perturbing DB values.

---

## ReAct Loop (Deep Dive, Stage-3 Style)

The defended ReAct loop lives in `notebooks/03_agentic_eval.ipynb`. Conceptually:

1. Build prompts per step
- ReAct prompt: includes schema + history + last observation.
- Optional tabular prompt: encourages explicit join reasoning.

2. Generate candidate SQL strings
- Uses sampling when enabled (`do_sample=True`) to explore multiple candidates.
- Uses a stop-on-semicolon criterion to avoid run-on non-SQL text.

3. Clean and postprocess each candidate
- Cleaning: enforce “single SELECT with FROM”, strip prompt echo, reject junk.
- Deterministic postprocess: `guarded_postprocess`, canonicalize table casing, clamps.
- Optional projection contract: only applied when NLQ explicitly lists fields.

4. Execution gate (Act)
- Execute with `QueryRunner.run(sql)` against the base DB.
- Capture errors for debugging and for repair prompting.

5. Intent gate
- Apply `intent_constraints(nlq, sql)` to reject executable-but-wrong-type queries.

6. Scoring and selection
- Use `semantic_score` and `count_select_columns` (and any extra heuristic) to pick the best candidate.
- The loop returns as soon as a best candidate is found for a step.

7. Optional repair
- If there were execution failures, `repair_sql` prompts a fix using schema + bad_sql + error message.
- Repairs are re-executed and must pass the same gates.

8. Fallback
- If steps are exhausted, call `vanilla_candidate` (deterministic few-shot) to avoid empty outputs.

Key code locations:
- Helper layer (prompts, cleaning, clamps, repair): `notebooks/03_agentic_eval.ipynb` under `## 6. Helper Layer: ...`.
- Loop: `react_sql` and `evaluate_candidate` in `notebooks/03_agentic_eval.ipynb` (cell `# 7) Simple full ReAct loop (explainable + observation feedback)`).
- Execution tool: `nl2sql/query_runner.py` (`QueryRunner.run`).
- Intent + scoring helpers: `nl2sql/agent_utils.py`.

Traceability:
- The loop builds a structured `trace` list of dicts (phases like `clean_reject`, `exec_fail`, `intent_reject`, `accept`, `repair`, `final`). This is saved per item in the agent results JSON.

---

## Calculating and Reporting Results

### Baseline and QLoRA (library harness)
Where rates come from:
- `nl2sql.eval.eval_run` returns a list of `EvalItem` records.
- It computes rates as:
  - `va_rate = sum(va) / n`
  - `em_rate = sum(em) / n`
  - `ex_rate = sum(ex) / n`
- It prints a one-line summary and optionally writes the full JSON payload.

### Agentic notebook (custom evaluation loop)
Where rates come from:
- The notebook loops over `test_set` items and stores per-item dicts including `trace` and `ts_debug`.
- It computes:
  - `va_rate = sum(va) / n`
  - `em_rate = sum(em) / n`
  - `ex_rate = sum(ex) / n`
  - `ts_rate = sum(ts) / n`
- It writes a single JSON file: `results/agent/results_react_200.json`.

Output fields you can defend:
- VA/EX/EM/TS are explicitly stored per item, along with:
  - `pred_sql` and `gold_sql` (audit)
  - `trace` (why candidates were rejected/accepted)
  - `ts_debug` (per-replica outcomes)

---

## Scripts (What They Are For)

- `scripts/run_full_pipeline.py`: CLI to run baseline + QLoRA eval and a small ReAct sanity check (not identical to the full notebook ReAct loop).
- `scripts/analyze_results.py`: post-hoc analysis for EX failures (projection mismatch, intent mismatch, join mismatch, literal mismatch).
- `scripts/ts_eval.py`: older standalone TS utility (canonical TS now lives in `nl2sql/eval.py`).
- `scripts/strip_ipynb.py`: remove notebook outputs and excessive metadata.
- `scripts/fix_ipynb_widgets.py`: drop invalid widget metadata to fix GitHub rendering.

---

## Environment and Dependencies

Key runtime dependencies:
- Transformers/Accelerate/BitsAndBytes/PEFT/TRL for model loading and QLoRA training (`requirements.txt`).
- SQLAlchemy + PyMySQL + Cloud SQL connector for DB access.
- pandas/numpy for result handling and optional previews.

Expected environment variables (notebooks/scripts):
- `INSTANCE_CONNECTION_NAME`, `DB_USER`, `DB_PASS`, `DB_NAME`
- Optional auth: `GOOGLE_APPLICATION_CREDENTIALS` (service account JSON path) or ADC
- Optional model auth: `HF_TOKEN` / huggingface token where needed in notebooks

---

## Known Sharp Edges (Current-State Notes)

- `notebooks/03_agentic_eval.ipynb` references `score_sql(...)` in candidate scoring, but there is no `def score_sql` in the notebook or in `nl2sql/`. If run as-is, this will raise a `NameError` unless the notebook defines it in an earlier cell.
- `results/README.md` claims `results/` is gitignored, but `.gitignore` currently does not ignore it. Treat `results/README.md` as informational, not authoritative.
