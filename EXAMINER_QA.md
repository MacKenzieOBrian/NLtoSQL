# Examiner Q&A (Condensed)

This is a short viva cheat sheet written to match the **current implementation** in this repo.

Primary places to point:
- `notebooks/03_agentic_eval.ipynb` (evaluation loop + tool-driven ReAct loop)
- `LOGBOOK.md` (dated research journey)
- `2_METHODOLOGY.md`, `4_EVALUATION.md`, `6_LIMITATIONS.md` (write-up)
- `nl2sql/agent_tools.py`, `nl2sql/query_runner.py`, `nl2sql/eval.py` (code truth)

---

## Q: What is the core contribution?

A: A reproducible OSS NL->SQL evaluation pipeline on ClassicModels, plus an evidence-driven agentic loop that makes “what the model did” auditable: explicit tool actions, deterministic guardrails, execution feedback, and trace/decision logs.

---

## Q: Why prioritize EX/TS over exact-match (EM)?

A: EM is sensitive to formatting and semantically equivalent rewrites. EX checks whether the result rows match the gold query on the base DB. TS adds robustness by checking behavior across perturbed replicas (a lightweight approximation of Zhong et al.’s test-suite idea).

Code pointers:
- `nl2sql/eval.py` (`execution_accuracy`, `test_suite_accuracy_for_item`)
- `4_EVALUATION.md`

---

## Q: What does “tool-driven ReAct” mean in this project?

A: The model outputs `Action: tool_name[json]`. Python executes that tool and records an `Observation:`. Critical steps (validation, execution, finish) are gated, and failures force a repair step.

Code pointers:
- `notebooks/03_agentic_eval.ipynb` (`react_sql`)
- `nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)

---

## Q: What tools exist, and why?

A: Tools break the task into explicit, checkable steps:
- schema grounding: `get_schema`, `link_schema`, optional `get_table_samples`
- generation: `generate_sql`
- checks: `validate_sql`, `validate_constraints`, `intent_constraints`
- environment interaction: `run_sql`
- recovery: `repair_sql`
- termination: `finish`

Code pointers:
- `nl2sql/agent_tools.py`
- `nl2sql/agent_utils.py` (`intent_constraints`)

---

## Q: How do you stop the model from executing destructive SQL?

A: Execution is mediated by `QueryRunner` with a SELECT-only safety policy and a forbidden-token blocklist (no UPDATE/DELETE/DROP/etc.). Even if the model emits unsafe SQL, execution is blocked.

Code pointers:
- `nl2sql/query_runner.py` (`QueryRunner.run`)

---

## Q: Why include schema linking?

A: Full schemas are large and increase wrong-table / wrong-join errors. `link_schema` prunes the schema text to a relevant subset before generation (heuristic, logged), aligning with the RESDSQL idea of separating linking from generation.

Code pointers:
- `nl2sql/agent_tools.py` (`link_schema`)
- `nl2sql/agent_utils.py` (`build_schema_subset`)

---

## Q: What is “constraint extraction” and what does it buy you?

A: It is a deterministic pass over the NLQ to infer structural requirements (aggregation, GROUP BY, ORDER BY, LIMIT, DISTINCT). This reduces a common failure mode: SQL that runs but has the wrong structure.

Code pointers:
- `nl2sql/agent_tools.py` (`extract_constraints`, `validate_constraints`)

---

## Q: What happens when validation or execution fails?

A: The loop records the failure reason and forces a repair step. Repair uses the latest error feedback to revise SQL, then guardrails re-apply before re-validation.

Code pointers:
- `notebooks/03_agentic_eval.ipynb` (`pending_repair_error` + action override)
- `nl2sql/agent_tools.py` (`repair_sql`)

---

## Q: What is logged and why does it matter?

A: The loop logs a full `trace` (raw model output + action + observation per step) and a compact `decision_log` (what happened and why). This makes it possible to attribute failures to generation vs guardrails vs validation vs execution.

Code pointers:
- `notebooks/03_agentic_eval.ipynb` (trace + decision_log)

---

## Q: How does QLoRA relate to the agent loop?

A: QLoRA changes model weights to improve domain mapping (semantics). The agent loop does not change weights; it improves reliability via explicit tool checks and execution feedback. In practice they address different error sources.

Write-up pointers:
- `2_METHODOLOGY.md` (sequencing: prompting -> QLoRA -> agentic loop)
- `LOGBOOK.md`

---

## Q: What are the main limitations?

A: Heuristic schema linking and constraint extraction can miss subtle intent. EX is a single-DB oracle; TS is a suite-based replica approach rather than full distilled test-suite construction. The agent loop improves auditability but does not guarantee semantic correctness.

Write-up pointers:
- `6_LIMITATIONS.md`

---

## Q: How do you ensure reproducibility?

A: Deterministic decoding is used by default, evaluation is scripted/notebooked, dependencies are pinned, and outputs are saved in structured formats.

Repo pointers:
- `requirements.txt`
- `notebooks/03_agentic_eval.ipynb` (deterministic settings + result saving)
