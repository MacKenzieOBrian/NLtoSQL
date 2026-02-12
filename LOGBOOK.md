# Logbook (Research & Development Timeline)

---

## Phase 1 — Planning & Scoping (Sep–Oct 2025)

### 2025-09-29 — Literature Mapping & Scope Setting
- **Activities:** Supervisor meeting; initial outline; reading on ReAct, PEFT, and agentic NL->SQL.
- **Lit Context:** Proprietary agent stacks were strong but hard to reproduce locally.
- **Challenges:** Scope too broad and contribution boundary unclear.
- **Insights:** The reproducibility gap became the core research angle.
- **Next:** Expand SOTA coverage and connect it to measurable evaluation axes.

### 2025-10-06 — SOTA Consolidation
- **Activities:** Expanded literature chapter; drafted ethics; organized references and citation flow.
- **Lit Link:** Early framing that NL->SQL quality depends on schema grounding, semantics, and execution.
- **Insight:** Literature section should justify method choices, not just list models.
- **Next:** Tie SOTA claims directly to metric and experiment design.

### 2025-10-13 — Evaluation Perspective Added
- **Activities:** Added VA/EX/TS framing and rewrote SOTA as critical analysis.
- **Lit Link:** Spider/Test Suite work supports execution-first semantic evaluation.
- **Reflection:** Narrative shifted from descriptive to analytical.

### 2025-10-20 — Methodology Framing
- **Activities:** Added constraints (compute, time, reproducibility), SMART/MoSCoW planning.
- **Lit Link:** Execution-centric evaluation aligns with benchmark practice.
- **Insight:** Contribution should be controlled reproduction under realistic resource limits.
- **Next:** Build infrastructure for execution-grounded evaluation.

### 2025-10-27 — Infra Abstraction for Agents
- **Activities:** Moved from SQLite prototype to MySQL setup; introduced `QueryRunner` abstraction.
- **Lit Link:** ReAct-style methods require explicit tool execution interfaces.
- **Outcome:** Architecture became compatible with later agentic experimentation.

---

## Phase 2 — Infra & Data Prep (Nov–Dec 2025)

> **Research Question:**  
> _What baseline performance can OSS LLMs achieve on ClassicModels before any training?_

### 2025-11-03 — Execution Pipeline Setup
- **Activities:** Implemented Cloud SQL connection, safe execution wrapper, schema introspection.
- **Lit Link:** Execution evaluation is required for EX/TS and agentic feedback loops.
- **Reflection:** Evaluation infrastructure had to be stable before model adaptation work.

### 2025-12-14 — Few-Shot Baseline Introduced
- **Activities:** Added system+schema+exemplar prompting, deterministic decoding, and SELECT guardrails.
- **Lit Link:** ICL improves structure, but semantic robustness remains limited.
- **Observation:** Validity improved; semantic correctness lagged.
- **Learning:** Prompting mostly fixes format, not deep join/aggregation reasoning.
- **Next:** Test whether weight adaptation narrows semantic gap.

### 2025-12-23 — Full Baseline Results (200 items)
- **Results:**
  - k=0 -> VA 0.810 / EX 0.000
  - k=3 -> VA 0.865 / EX 0.250
- **Lit Alignment:** Matches reports that exemplars help schema patterning more than compositional semantics.
- **Learning:** Few-shot is necessary baseline evidence, but not sufficient for semantic accuracy.

---

## Phase 3 — QLoRA Fine-Tuning (Jan 2026)

> **Research Question:**  
> _Can lightweight PEFT (QLoRA) improve semantic mapping (joins/aggregates) under commodity constraints?_

### 2026-01-12 — QLoRA Run #2 (r=32, alpha=64, 3 epochs)
- **Results:**
  - k=0 -> EX 0.065
  - k=3 -> EX 0.380
- **Lit Link:** PEFT/QLoRA can improve domain adaptation while preserving feasible compute.
- **Reflection:** Prompting and QLoRA behaved as complementary mechanisms.
- **Next:** Introduce execution-guided loop to improve robustness and traceability.

---

## Phase 4 — ReAct Exploration & Refinement (Jan–Feb 2026)

> **Research Question:**  
> _How much can execution-guided tooling improve validity and auditability, and which semantic errors remain?_ 

### 2026-01-23 — Micro-Slice Stability Check
- **Activities:** Added projection guard and ORDER/LIMIT clamps; tested small slice.
- **Results:** Micro-slice reached VA/EX/EM near ceiling.
- **Lit Link:** Consistent with constrained-decoding intuition (PICARD-style).
- **Learning:** Structural controls can eliminate format errors but do not prove semantic generalization.

### 2026-01-25 — Full-Set Reality Check
- **Results:** VA remained high (~1.0), EX dropped (~0.05) on full set.
- **Failure Modes:** Join path, aggregation scope, and filter grounding errors.
- **Lit Link:** Execution feedback improves robustness more than semantic alignment.
- **Learning:** This is the semantic bottleneck regime.

### 2026-01-27 — Staged ReAct Decision Process
- **Activities:** Reorganized loop into staged ablations (minimal -> clamp -> rerank -> repair).
- **Motivation:** Isolate where valid candidates were being lost.
- **Outcome:** Pipeline became evidence-driven and easier to debug.

### 2026-01-29 — Trace Logging Upgrade
- **Activities:** Added structured traces across generation, cleaning, validation, execution, and repair.
- **Reason:** Needed attribution of failures to exact pipeline stage.
- **Outcome:** Improved explainability for viva and error analysis.

### 2026-01-30 — Stage-3 Full Run (200)
- **Results:** VA 0.805 / EX 0.130 / EM 0.105.
- **Interpretation:** Validity controls worked; semantic correctness still low.
- **Next:** Add TS and failure taxonomy for mechanism-level explanation.

### 2026-02-01 to 2026-02-02 — Rebuild Plan + Evaluation Cleanup
- **Activities:** Simplified notebook flow, clarified metric interpretation, tightened evaluation cell behavior.
- **Outcome:** Faster iteration loop and cleaner artifact generation.

### 2026-02-04 to 2026-02-05 — Tool-Driven Loop Implemented
- **Activities:** Implemented explicit tool order and failure-triggered repair path.
- **Changes:** Removed confusing prompt variants and softened over-strict intent blocking.
- **Outcome:** More stable and auditable execution path.

### 2026-02-06 — Action Parsing and Control-Flow Fixes
- **Activities:** Fixed multi-action parsing ambiguity; improved setup/tool boundaries; forced finish after success.
- **Learning:** Small parser/control defects had outsized impact on perceived agent quality.

### 2026-02-07 — Quick Check + Error Taxonomy Start
- **Results:**
  - pre-fix (10 items): VA 1.00 / EX 0.30 / EM 0.20 / TS 0.30
  - post-fix (10 items): VA 1.00 / EX 0.70 / EM 0.50 / TS 0.70
- **Observation:** Fallback misuse and projection errors dominated failures.
- **Next:** Add hard gates so fallback cannot overwrite better validated candidates.

### 2026-02-08 — Linking and Constraint Tightening
- **Activities:** Added schema ranking, projection hints, value-column hints, and conservative constrained decoding.
- **Lit Link:** RAT-SQL/RESDSQL/BRIDGE/PICARD-informed improvements.
- **Expected Effect:** Reduce join/value/projection mismatch errors.

### 2026-02-09 — Error-Driven Constraint/Guardrail Passes
- **Activities:** Iterated required tables, self-join constraints, join-path enforcement, and entity-listing guardrail relaxations.
- **Reason:** Addressed recurring EX failures in payment aggregates, manager self-joins, and listing projections.
- **Outcome:** Better structural alignment checks with clearer error-to-change mapping.

### 2026-02-10 — Refactor and Research-First Consolidation
- **Activities:**
  - Split policy/orchestration modules (`constraint_policy`, `repair_policy`, helpers).
  - Added statistical utilities (Wilson intervals, exact McNemar) and paired comparison outputs.
  - Reframed docs so prompting/QLoRA remain primary claims and ReAct remains infrastructure.
  - Logged QLoRA configuration rationale and aligned replication framing across docs/notebooks.
- **Outcome:** Method became cleaner to defend, with stronger statistical reporting.

### 2026-02-12 — ReAct Loop Hardening
- **Activities:**
  - Fixed self-join validation false-negative in `validation.py`.
  - Tightened `react_pipeline.py` so failed paths return `no_prediction` instead of known-failed SQL.
  - Added explicit stop reasons for failed exits in trace.
- **Method Alignment:** Updated methodology text to match implemented loop behavior.
- **Justification:** Execution-guided and constrained-acceptance principles favor rejecting failing programs before final output.

---

## Phase Summary — Current Position

- **Primary evidence track:** Controlled baseline vs QLoRA comparisons with paired statistics.
- **ReAct position:** Execution infrastructure for validity/traceability, not primary semantic claim.
- **Main bottleneck:** Semantic alignment (join path, aggregation scope, value grounding).
- **Immediate next step:** Re-run ReAct with hardened loop, regenerate `results/analysis/*`, then update dissertation claims from refreshed artifacts.
