# Methodology

This section explains how the experiments are designed and executed. The aim is controlled comparison across methods, not the absolute best possible score.

---

## Research Design

The methodology is error-driven and incremental (cf. ReAct and execution-feedback work):
1. Establish a deterministic prompting baseline.
2. Add PEFT (QLoRA) to test whether training improves SQL generation.
3. Add an agentic phase: first a candidate‑ranking utility study (Jan 2026) to identify EX‑relevant controls, then a bounded tool‑driven ReAct loop (Feb 2026) that formalizes those controls as explicit tools.

This sequencing makes improvements attributable to specific changes and follows a literature-backed progression from prompting → PEFT → agentic execution feedback.  
Refs: `REFERENCES.md#ref-brown2020-gpt3`, `REFERENCES.md#ref-ding2023-peft`, `REFERENCES.md#ref-goswami2024-peft`, `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-zhai2025-excot`, `REFERENCES.md#ref-ojuri2025-agents`.

**Research journey summary (from `LOGBOOK.md` dates)**  
- 2025-09-29: scoped the problem and identified a reproducibility gap.  
- 2025-10-06: shifted evaluation emphasis from EM to EX/TS for semantic validity.  
- 2026-01-31: candidate‑ranking utility study to see which guards improved EX and which were noise.  
- 2026-02-01: planned the explicit tool sequence (schema → generate → validate → run → reflect).  
- 2026-02-04 to 2026-02-05: rebuilt the loop as tool‑driven ReAct with validation, schema linking, constraint checks, and decision logging.  

**Interpretive narrative (what changed and why)**  
- Baselines improved VA but not EX, which signaled that syntax fixes alone do not solve semantic alignment.  
- QLoRA improved EX but still depended on strong schema grounding, so I added explicit guardrails.  
- The candidate‑ranking loop was used to test which utilities actually lifted EX; it improved validity but masked root causes by filtering errors away.  
- I then pivoted to a tool‑driven Thought→Action→Observation loop so errors became explicit observations and tool order could be enforced.  
- Carried forward: cleaning/normalization, schema‑aware validation, constraint checks, semantic scoring signals, and reflection logic.  
- Dropped or de‑emphasized: candidate‑ranking as the control structure, the tabular prompt variant, and hard intent rejection (softened to a penalty).  

**Explicit non‑decisions (scope control)**  
- No learned schema linker (kept heuristic for interpretability).  
- No unbounded reflection (bounded steps for auditability and cost).  
- No full distilled test‑suite construction (TS remains suite‑based due to time/compute constraints).

---

## Dataset and Split

The ClassicModels database is used as the fixed schema. A small, clean split is used for feasibility:
- Training: `data/train/classicmodels_train_200.jsonl`
- Test: `data/classicmodels_test_200.json`

A fixed schema makes results interpretable and keeps comparisons fair across methods.

---

## Baseline Prompting (ICL)

Baseline evaluation uses a fixed system prompt and deterministic decoding. This creates a stable reference point for later changes and matches ICL baselines in the literature.  
Refs: `REFERENCES.md#ref-brown2020-gpt3`, `REFERENCES.md#ref-mosbach2023-icl`.

Implementation notes:
- Prompt format: `nl2sql/prompting.py`
- Deterministic generation: `nl2sql/llm.py`
- Postprocess: `nl2sql/postprocess.py`
- Evaluation: `nl2sql/eval.py:eval_run`

---

## QLoRA Fine-Tuning

QLoRA adapters are trained to test whether task-specific data improves SQL generation. The base model is kept fixed and adapters are evaluated with the same harness as the baseline, aligning with PEFT/QLoRA practice.  
Refs: `REFERENCES.md#ref-ding2023-peft`, `REFERENCES.md#ref-goswami2024-peft`.

Implementation notes:
- Training + eval notebook: `notebooks/05_qlora_train_eval.ipynb`
- Evaluation harness: `nl2sql/eval.py:eval_run`

---

## Agentic ReAct Loop (Tool‑Driven Execution Feedback)

The agent uses an explicit Thought → Action → Observation loop with tools. It does not change model weights. The loop is bounded and traceable, mirroring ReAct and agent-mediated NL→SQL workflows:
- Bootstrap with `get_schema` then `link_schema` (schema observation + heuristic linker)
- LLM chooses actions (`extract_constraints`, `generate_sql`, `validate_sql`, `validate_constraints`, `run_sql`, `repair_sql`, `finish`)
- Python executes tools and returns observations
- Guardrails run between `generate_sql`/`repair_sql` and `validate_sql`
- `validate_sql` must pass before `validate_constraints`
- `validate_constraints` must pass before `run_sql`
- `run_sql` must succeed before `finish`
- Deterministic fallback if the loop fails to finish
Validation or execution failures force a `repair_sql` step. Constraint validation gates execution (e.g., missing COUNT/ORDER/LIMIT), and per‑query trace summaries log action sequences and compliance for auditability.
Refs: `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-zhai2025-excot`, `REFERENCES.md#ref-ojuri2025-agents`.

**Tightening for accuracy (constraint‑driven gating)**  
We progressively tightened acceptance criteria by enforcing explicit field/value constraints, schema‑aware validation (including join‑key checks), and execution‑gated acceptance. This mirrors execution‑guided decoding (rejecting candidates via execution feedback) and constrained decoding principles that restrict outputs to valid/consistent SQL, while relation‑aware linking helps avoid join/table errors.  
Refs: `REFERENCES.md#ref-wang2018-eg-decoding`, `REFERENCES.md#ref-scholak2021-picard`, `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`.

**Schema item ranking (table + column shortlist)**  
Before generation, the linker ranks both tables and columns to present a compact schema subset. This operationalizes relation‑aware schema linking and decoupled schema selection to reduce projection/join mistakes without retraining.  
Refs: `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`.

**Lightweight value linking (column hints)**  
NLQ literals are linked to likely columns using lexical cues (e.g., date patterns, location phrases, ID‑style phrases). These value‑column hints bias schema pruning toward correct filter columns without database lookups.  
Refs: `REFERENCES.md#ref-lin2020-bridge`, `REFERENCES.md#ref-wang2020-ratsql`.

**Conservative constrained decoding (PICARD‑lite)**  
Generation applies a minimal constrained decoding step that blocks obvious non‑SELECT outputs (DDL/DML/transaction keywords). This enforces early validity without over‑restricting the model.  
Refs: `REFERENCES.md#ref-scholak2021-picard`.

**Evolution from candidate‑ranking**  
- Candidate‑ranking utilities that improved EX were retained but converted into explicit tools or guardrails.  
- The ranking decision itself was removed; the loop now relies on ordered actions with observations and forced repair on failure.  
- This conversion made the pipeline auditable and aligned with ReAct’s explicit action/observation semantics.  

Implementation notes:
- Tool interface: `nl2sql/agent_tools.py`
- System prompt: `nl2sql/prompts.py`
- Notebook loop: `notebooks/03_agentic_eval.ipynb` (tool-driven `react_sql`)
- Execution gate: `nl2sql/query_runner.py:QueryRunner.run`
- Technical reference: `TOOL_DRIVEN_REACT_LOOP_TECHNICAL_REFERENCE.md`

---

## Evaluation Metrics

Four metrics are reported (execution-based evaluation prioritized over exact string match):
- VA: executability (SQL runs successfully)
- EM: exact match (diagnostic only)
- EX: execution accuracy on base DB (result equivalence)
- TS: test-suite accuracy across perturbed DB replicas
Refs: `REFERENCES.md#ref-zhong2020-ts`, `REFERENCES.md#ref-yu2018-spider`.

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
