# Methodology

This section explains how the experiments are designed and executed. The aim is controlled comparison across methods, not the absolute best possible score.

---

## Research Design

The methodology is error-driven and incremental:
1. Establish a deterministic prompting baseline.
2. Add PEFT (QLoRA) to test whether training improves SQL generation.
3. Add a bounded ReAct-style agent loop to use execution feedback without changing model weights.

This sequencing makes improvements attributable to specific changes.

---

## Dataset and Split

The ClassicModels database is used as the fixed schema. A small, clean split is used for feasibility:
- Training: `data/train/classicmodels_train_200.jsonl`
- Test: `data/classicmodels_test_200.json`

A fixed schema makes results interpretable and keeps comparisons fair across methods.

---

## Baseline Prompting (ICL)

Baseline evaluation uses a fixed system prompt and deterministic decoding. This creates a stable reference point for later changes.

Implementation notes:
- Prompt format: `nl2sql/prompting.py`
- Deterministic generation: `nl2sql/llm.py`
- Postprocess: `nl2sql/postprocess.py`
- Evaluation: `nl2sql/eval.py:eval_run`

---

## QLoRA Fine-Tuning

QLoRA adapters are trained to test whether task-specific data improves SQL generation. The base model is kept fixed and adapters are evaluated with the same harness as the baseline.

Implementation notes:
- Training + eval notebook: `notebooks/05_qlora_train_eval.ipynb`
- Evaluation harness: `nl2sql/eval.py:eval_run`

---

## Agentic ReAct Loop (Execution Feedback)

The agent adds control logic around execution feedback. It does not change model weights. The loop is bounded and traceable:
- Generate multiple candidates
- Clean + deterministic postprocess
- Execute (SELECT-only guard)
- Intent gate + scoring
- Optional bounded repair

Implementation notes:
- Loop: `nl2sql/agent.py:ReactSqlAgent.react_sql`
- Execution gate: `nl2sql/query_runner.py:QueryRunner.run`
- Evaluation: `notebooks/03_agentic_eval.ipynb`

---

## Evaluation Metrics

Four metrics are reported:
- VA: executability (SQL runs successfully)
- EM: exact match (diagnostic only)
- EX: execution accuracy on base DB (result equivalence)
- TS: test-suite accuracy across perturbed DB replicas

Implementation notes:
- VA: `nl2sql/query_runner.py`
- EM: `nl2sql/postprocess.py:normalize_sql`
- EX: `nl2sql/eval.py:execution_accuracy`
- TS: `nl2sql/eval.py:test_suite_accuracy_for_item`

---

## Reproducibility and Safety

- Dependencies are pinned in `requirements.txt`.
- Query execution is SELECT-only; destructive tokens are blocked.
- Result sets are capped during EX/TS to avoid runaway comparisons.

