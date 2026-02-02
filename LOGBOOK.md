# Logbook (Research & Development Timeline)

## At‑a‑Glance
- Four phases from scoping → baselines → QLoRA → agentic refinement
- Each phase anchored by a research question
- Entries emphasise **decisions, evidence, and evaluation outcomes** (not raw implementation logs)

## Index
1. Phase 1 — Planning & Scoping (Sep–Oct 2025)  
2. Phase 2 — Infra & Data Prep (Nov–Dec 2025)  
3. Phase 3 — QLoRA Fine‑Tuning (Jan 2026)  
4. Phase 4 — ReAct Exploration & Refinement (Jan 2026)  

This logbook documents the research trajectory across four phases, emphasising
(1) literature‑aligned decision making,
(2) hypothesis‑driven experimentation, and
(3) evaluation‑centred interpretation rather than implementation logs.

Each phase is anchored to a guiding research question, consistent with NL→SQL
methodologies in recent literature (Spider/BIRD/Ojuri).

---

## Phase 1 — Planning & Scoping (Sep–Oct 2025)

> **Research Question:**  
> _What components are required to reproduce an NL→SQL pipeline and how do prompting and schema grounding behave before any training?_

### 2025-09-29 — Literature Mapping & Scope Setting
- **Activities:** Supervisor meeting; initial outline; reading on ReAct, PEFT, agentic NL→SQL.
- **Lit Context:** Proprietary agents (GPT-4/Ojuri) strong but not reproducible; OSS gap identified.
- **Challenges:** Scope broad; unclear contribution boundary.
- **Insights:** “Reproducibility gap” emerged as a credible research angle.
- **Next:** Expand SOTA beyond seed paper; align contribution to evaluation axes.

### 2025-10-06 — SOTA Consolidation
- **Activities:** Expanded outline (+2k words); ethics draft; Zotero setup; literature broadened.
- **Lit Link:** Early recognition that NL→SQL = _schema + semantics + execution_.
- **Insight:** SOTA must justify methodology (e.g., metrics), not just catalogue models.
- **Next:** Tie SOTA to evaluation framework explicitly.

### 2025-10-13 — Evaluation Perspective Added
- **Activities:** Added execution metrics (VA/EX/TS) and citations; rewrote SOTA critically.
- **Lit Link:** Execution used as semantic oracle in NL→SQL benchmarks.
- **Reflection:** SOTA improved from descriptive → analytical.

### 2025-10-20 — Methodology Framing
- **Activities:** Added SMART/MoSCoW, compute constraints, reproducibility requirements.
- **Lit Link:** Execution-centric evaluation legitimised via Spider/TestSuite literature.
- **Insight:** Contribution formalised as controlled reproduction under resource constraints.
- **Next:** Begin DB/infra provisioning for execution-based eval.

### 2025-10-27 — Infra Abstraction for Agents
- **Activities:** Switched SQLite → MySQL Cloud SQL; added `QueryRunner` abstraction.
- **Lit Link:** ReAct treats DB as “Action” interface; tool execution is integral to evaluation.
- **Outcome:** Architecture compatible with future agentic experimentation.

---

## Phase 2 — Infra & Data Prep (Nov–Dec 2025)

> **Research Question:**  
> _What baseline performance can OSS LLMs achieve on ClassicModels before any training?_

### 2025-11-03 — Execution Pipeline Setup
- **Activities:** Cloud SQL connector; safe_connection; schema introspection; `QueryRunner`.
- **Lit Link:** Execution required for EX/TS metrics and ReAct/ExCoT-style agents.
- **Reflection:** Evaluation pipeline precedes model adaptation in NL→SQL research.

### 2025-12-14 — Few-Shot Baseline Introduced
- **Activities:** System + schema + exemplar prompting; deterministic decoding; SELECT guard.
- **Lit Link:** ICL boosts structure, weak on semantics (as reported in NL→SQL surveys).
- **Observation:** VA increased; EX remained low.
- **Interpretation:** Prompting solves syntax; semantic joins/aggregates remain unmet.
- **Next:** Test whether weight adaptation addresses semantic gap.

### 2025-12-23 — Full Baseline Results (200 items)
- **Results:**  
  - k=0 → VA 0.810 / EX 0.000  
  - k=3 → VA 0.865 / EX 0.250
- **Lit Alignment:** Matches literature: exemplars induce schema patterns but not numeric reasoning or compositional semantics.

---

## Phase 3 — QLoRA Fine-Tuning (Jan 2026)

> **Research Question:**  
> _Can lightweight PEFT (QLoRA) teach domain semantic mappings (joins/aggregates) under commodity GPUs?_

### 2026-01-12 — QLoRA Run #2 (r=32, α=64, 3 epochs)
- **Results:**  
  - k=0 → EX 0.065  
  - k=3 → EX 0.380
- **Lit Link:** PEFT helps semantic alignment; still sensitive to exemplars for schema anchoring.
- **Reflection:**  
  - Prompting = _structure_  
  - QLoRA = _semantic mapping_  
  - Both are complementary in NL→SQL.
- **Next:** Add execution-guided agent to address robustness.

---

## Phase 4 — ReAct Exploration & Refinement (Jan 2026)

> **Research Question:**  
> _Does execution feedback reduce semantic errors without additional training?_

### 2026-01-23 — Micro-Slice Stability Check
- **Activities:** Projection guard + ORDER/LIMIT clamp; slice reached VA/EX/EM = 1.0.
- **Lit Link:** Consistent with constrained decoding (PICARD-style) literature.
- **Insight:** Some EX failures are structural, not semantic.

### 2026-01-25 — Full-Set Reality Check
- **Results:** VA ~1.0, EX ~0.05
- **Failure Modes:** revenue/grouping, multi-hop joins, misaligned filters.
- **Lit Link:** Execution guidance insufficient for semantic leaps; matches survey findings.
- **Reflection:** We entered “semantic bottleneck” regime.

### 2026-01-26 — Literature-Aligned Diagnosis
- **Conclusion:**  
  - Execution = robustness  
  - QLoRA = semantics  
  - Prompting = grounding  
- **Lit Alignment:** Mirrors Spider/BIRD error taxonomies + ReAct failure analyses.
- **Next Steps (lit-driven):**  
  1) strengthen semantic curriculum (joins/aggregates)  
  2) add critic/reranker (ValueNet / Self-Refine)  
  3) schema linking enhancements (RAT-SQL-style)

### 2026-01-26 — Dev Note (fallback robustness)
- **Change:** Relaxed `clean_candidate` and made `vanilla_candidate` baseline-aligned (extract first SELECT + guarded_postprocess, minimal filtering).
- **Reason:** Strict filtering was suppressing valid baseline SQL, producing empty `pred_sql` even on simple queries.
- **Effect:** Fallback now returns a valid SELECT more reliably while keeping strict filters for agentic candidates.

### 2026-01-27 — Staged Decision Process (minimal → clamp → rerank → repair)
- **Decision:** Re-structured the notebook into staged ablations (STAGE 0–3) to restore validity before re‑introducing complexity.
- **Motivation:** Debug evidence showed valid SQL was being discarded by downstream filters; a minimal execution‑gated generator isolates the bottleneck.
- **Process (lit‑guided):** Start with execution‑guided decoding only (Zhong et al.), then add constraints (PICARD‑style), then reranking/critics, then repair.
- **Outcome:** Made the pipeline evidence‑driven rather than feature‑driven; each component is now re‑introduced only after validation.

### 2026-01-27 — Debug Note (spaced SQL tokens)
- **Issue:** Model output contained valid SQL but with letter‑spaced tokens (e.g., `S E L E C T ... F R O M ...`), causing `extract_first_select` and `clean_candidate` to reject it.
- **Fix:** Added normalization to collapse spaced keywords before filtering; updated notebook to wrap `clean_candidate` with the normalizer.
- **Effect:** Valid SQL now passes the filter even when spacing artifacts appear, restoring non‑empty predictions at STAGE 0.

### 2026-01-27 — Notebook Debug Guide (stage-gated ablation)
- **Change:** Added a “Staged Debugging Guide” markdown cell and stage‑gated `react_sql` in the notebook.
- **Rationale:** Aligns with ablation practice in NL→SQL and execution‑guided decoding—validate minimal execution‑gated behaviour first, then re‑introduce clamps, reranking, and repair only after stability.
- **Outcome:** Reduces accidental overwrites of the minimal agent and makes failures attributable to a specific stage.

### 2026-01-28 — Scaled Back for Baseline Validity
- **Decision:** Dialed the notebook back to the minimal execution‑gated ReAct stage to confirm the model can produce valid SQL before layering clamps, reranking, and repair.
- **Reason:** Recent failures showed that complex filters were masking whether the generator itself was working; restoring a minimal baseline isolates the true bottleneck.
- **Effect:** Establishes a reliable “known‑good” starting point for staged re‑introduction of features (literature‑aligned ablation).

### 2026-01-28 — ReAct Agent Staging (Stages 0 & 1)
- **Goal:** Bring up a minimal but working ReAct‑style NL→SQL agent in `notebooks/03_agentic_eval.ipynb`, then progressively add structure:  
  **Stage 0** = minimal execution‑gated generator; **Stage 1** = add projection/ORDER clamps.  
  The aim was a stable, debuggable baseline before sampling/reranking/repair.
- **Stage 0 wiring:** Implemented `_react_sql_minimal` (STAGE=0) vs `_react_sql_full` (STAGE≥1) with stage dispatch. Stage 0 builds a prompt, generates candidates, cleans them, runs via `QueryRunner`, then **falls back to `vanilla_candidate`** if all candidates fail.
- **Cleaning failures diagnosed:** Blank `PRED` traces were caused by **over‑strict cleaning**, not model failure. Two issues were observed:  
  1) `FROM` check failed after `extract_first_select + split(';',1)` returned only the prefix;  
  2) prompt‑echo tails (“output only sql / no explanation”) triggered `bad_phrase` even when valid SQL preceded them.
- **Fix implemented:** Added a shared `_clean_candidate_core` that:
  - normalises spaced‑out tokens (`S E L E C T` → `SELECT`),  
  - realigns to first `SELECT`,  
  - trims prompt‑echo tails,  
  - splits at the first `;`,  
  - **does not enforce a strict FROM check** (execution gating catches invalid SQL).  
  The relaxed cleaner is used for ReAct **and** monkey‑patched into `agent_utils.clean_candidate` so `vanilla_candidate` is consistent.
- **Stage 0 sanity checks:**  
  - “List all product lines.” → correct SQL returned.  
  - “Show product names, codes, and MSRPs.” → correct columns, order mismatch (EX ok, EM ≠).  
  - “USA customers” → extra column trimmed only after Stage 1 clamps.  
  - “SF office count” and “total per order” remained **semantic routing** errors (wrong table/aggregation).
- **Stage 1 clamps:** Enabled `strip_order_by_if_not_requested` and `trim_to_first_column` to improve minimal projection and remove ORDER/LIMIT artifacts. This improved EM/projection alignment but **did not fix semantic routing**, confirming clamps affect style not semantics.
- **Conclusion:** Stage 0 is now stable and debuggable; Stage 1 improves projection hygiene; remaining errors are semantic (joins/aggregation), requiring reranking/repair or better priors.

---

## Current State (Jan 2026)

- Prompting fixes syntax
- QLoRA fixes semantics
- Execution fixes robustness
- Critic/curriculum expected to fix compositional reasoning

The dissertation narrative can legitimately focus on whether **small open models + PEFT + execution** can approximate proprietary agent performance under reproducible resource constraints.

---


### 2026-01-29 — Stage 3 Outputs + Trace Logging Upgrade
- **Observation:** Stage 3 outputs are mostly valid SQL; remaining issues are projection bloat and unnecessary ORDER BY/GROUP BY that reduce EM (e.g., extra `productLine`, spurious `ORDER BY` on totals).
- **Change:** Added structured trace logging to the ReAct loop (raw candidate → cleaned SQL → post‑clamp SQL → execution error → repair attempt).
- **Repair Logging:** `repair_sql` now returns both the repaired SQL and a small metadata dict (status, raw_fix, exec_error), so traces show *why* a repair succeeded or failed.
- **Reason:** Traceability is needed to attribute errors to generation vs cleaning vs execution vs repair; aligns with agentic evaluation practice in ReAct/Reflexion‑style loops.

### 2026-01-29 — Biggest Win: Output‑Control + Semantic Acceptance Gate
- **Finding:** Two failure modes dominated:  
  1) **Prompt‑echo garbage** after a valid SQL fragment (causing syntax errors).  
  2) **Executable but irrelevant repairs** (e.g., `SELECT 1 FROM dual;`) that inflate VA but fail task intent.
- **Decision:** Treat output control and acceptance as *first‑class* components of the agent loop:  
  - **Stop‑on‑semicolon** generation to end decoding at the first `;` (prevents prompt‑echo tails).  
  - **Prompt‑echo stripping** before cleaning (generic regex, not NLQ‑specific).  
  - **Semantic acceptance gate** (use `semantic_score` as a *threshold*, not only a reranker) so executable but irrelevant SQL is rejected.
- **Rationale (literature‑backed):**  
  Constrained decoding reduces invalid continuations (PICARD/Scholak et al., 2021), while execution‑guided decoding alone can accept spurious SQL unless paired with a semantic filter (Zhong et al., 2017; ValueNet/DIN‑SQL reranking). ReAct‑style agent loops require *format control + acceptance criteria* to avoid “valid‑but‑wrong” completions (Yao et al., 2023).
- **Outcome:** Stabilizes Stage‑3 correctness by separating **VA (runs)** from **task success (semantics)**, improving traceability and narrative clarity for the dissertation.

### 2026-01-29 — Stage 3 Stabilisation (Intent Constraints + Canonicalisation)
- **Fixes applied:**  
  - **Intent constraints:** added grouped‑aggregate checks (GROUP BY + aggregate + key in SELECT) and measure checks (total/amount ⇒ SUM), preventing “exec‑ok but wrong metric” outputs.  
  - **Table‑casing canonicalisation:** rewrote `FROM/JOIN` table names to the schema’s canonical case to remove case‑sensitive “table doesn’t exist” failures.  
  - **Cleaner hardening:** blocked `FROM dual`, `GROUP BY NULL`, dangling clauses, and prompt‑echo remnants that survived trimming.  
  - **Repair filtering:** multi‑candidate repair + best‑SELECT extraction to reject keyword‑soup fixes.
- **Effect:** Stage 3 now accepts **semantically plausible** SQL rather than any executable SQL; trace logs cleanly show where failures originate (generation vs cleaning vs execution vs repair).

### 2026-01-29 — Stage‑Gated Justification (Ablation + ReAct Pattern)
- **Why STAGE 0–3:** stage‑gating is an **ablation ladder** that isolates the impact of clamps, reranking, and repair. This prevents “opaque agent” claims and supports attribution of VA/EX changes to specific mechanisms (aligned with execution‑guided decoding and ReAct ablations).
- **STAGE 0 vs STAGE ≥1:** Stage 0 is **minimal execution‑gated decoding** (generate → extract → execute), while Stage ≥1 operationalises a **ReAct‑style loop** (generate → execute → observe → refine) with multi‑candidate search, clamping, and repair.
- **Trace logging:** structured traces (raw → cleaned → post‑clamp → exec → repair) create an audit trail for failure‑mode analysis and reproducibility.
- **Fallback rationale:** deterministic few‑shot fallback preserves benchmark coverage and comparability across configurations.

### 2026-01-30 — Stage‑3 Full Run (200 queries)
- **Run:** Stage‑3 ReAct on 200 ClassicModels NLQs (`results_react_200`).
- **Metrics:** VA **0.805**, EX **0.13**, EM **0.105**.
- **Interpretation (VA):** strong syntactic control; filtering + clamps + execution gating are working as intended.
- **Interpretation (EX):** semantic alignment remains the bottleneck (aggregation scope, join selection, projection granularity).
- **Interpretation (EM):** low EM is expected under agentic post‑processing and query rewrites; EM is diagnostic, not primary.
- **Literature alignment:** matches reports that execution feedback stabilises validity but does not guarantee semantic correctness (Ojuri et al., 2025; ExCoT; execution‑guided decoding).
- **Next steps logged:** introduce **Test‑Suite Accuracy (TS)** and a structured **error taxonomy** (projection, aggregation scope, join selection) to explain EX failures.
- **Dissertation narrative hook:** this run is the first **full‑set, agentic** result that isolates the “execution‑valid vs semantically‑correct” gap. It provides a concrete anchor for claims about why execution guidance improves stability but requires stronger semantic grounding or critic signals for EX gains.

### 2026-01-30 — EX Stabilisation Plan (Projection + Intent + Schema Linking)
- **Adjustment 1 — Projection contract:** enforce NLQ‑requested columns and drop extras; targets EX loss from projection drift.
- **Adjustment 2 — Intent classifier:** constrain query type (lookup vs aggregate vs grouped vs top‑k) to stop “wrong‑question” outputs.
- **Adjustment 3 — Schema‑subset prompting:** light schema linking (keyword→table + join hints) to reduce wrong table selection.
- **Rationale:** these are output‑shape and selection controls (not answer injection), aligned with constrained decoding and schema‑linking guidance in NL→SQL surveys.

### 2026-01-31 — Implemented EX Stabilisation (Projection + Intent + Schema Subset)
- **Implemented:** projection contract, intent classifier constraints, and schema‑subset prompting in the ReAct helper layer and notebook pipeline.  
- **Why (dissertation framing):** targets the dominant EX failure modes (projection drift, wrong question type, wrong table selection) without changing model weights. These are *control‑layer* interventions aligned with execution‑guided decoding and schema‑linking recommendations.  
- **Notes:**  
  - Projection contract enforces *output shape* when NLQ explicitly names fields.  
  - Intent constraints prevent valid‑but‑wrong query types (e.g., aggregate vs list).  
  - Schema subset reduces prompt scope using keyword‑to‑table hints + join hints.  
- **Expected impact:** raises EX by correcting “almost‑right” outputs while preserving VA.  

### 2026-01-31 — Simplified ReAct Loop (No STAGE Branching)
- **Change:** removed STAGE gating and replaced with a single, explainable ReAct loop that always follows: generate → clean → postprocess → execute → intent‑gate → score → (repair) → fallback.  
- **Config:** one `CFG` dict controls sampling, candidate count, clamps, projection contract, and repair.  
- **Reason:** improves traceability and removes branch‑specific behavior so results are easier to interpret and reproduce.  

### 2026-01-31 — EX Protection Patch (Projection Order + Repair Intent Gate)
- **Change:** projection contract now preserves explicit NLQ field order; repair acceptance is gated by the same intent constraints used for primary candidates.  
- **Why:** EX was failing on “almost‑right” outputs due to column order mismatch and repair occasionally overwrote correct intent with executable but irrelevant SQL.  
- **Effect:** raises EX by aligning output shape with the NLQ and prevents semantic drift during repair.  

### 2026-02-01 — TS Harness + Quick‑Test Toggles
- **Change:** added Test‑Suite Accuracy (TS) evaluation harness and quick‑test toggles (limit, TS_N, max rows) to the agentic eval notebook.  
- **Why:** TS provides semantic‑equivalence evaluation across perturbed DBs, while quick‑test toggles make iterative debugging feasible without full‑run cost.  
- **Effect:** enables rapid validation of EX improvements and supports a rigorous, reproducible evaluation narrative.  

### 2026-02-02 — Simplified Cell 6 (Readable ReAct Utilities)
- **Change:** refactored Cell 6 into small, named helpers (normalize, trim prompt‑echo, clean candidate, post‑process, clamps, repair) with plain‑English comments.  
- **Why:** improves explainability for examiners and makes the control‑layer logic defendable without reading dense regex.  
- **Effect:** same behavior, clearer narrative and easier debugging.  

### 2026-02-02 — Notebook Cleanup (TS util + score helper)
- **Change:** moved TS harness into `scripts/ts_eval.py` and imported it in the notebook; added a single `score_sql()` helper cell to centralize candidate scoring.  
- **Why:** keeps the notebook as an orchestration document and reduces “wall‑of‑code” sections; easier to justify and audit.  
- **Effect:** same evaluation behavior, cleaner notebook structure, clearer explanation for examiners.  

---

## ReAct Pipeline Cheat Sheet (Quick Reference)

**Goal:** make the loop explainable, debuggable, and reproducible.

**Inputs**
- NLQ (question), schema summary, model, runner (DB executor)

**Core phases (in order)**
1) **Generate** — produce multiple SQL candidates (main + tabular prompt)  
2) **Clean** — enforce single `SELECT … FROM … ;`, remove prompt echo, reject dangling SQL  
3) **Post‑process** — projection guard, optional projection contract, canonical table casing, clamps  
4) **Execute** — run SQL; failure becomes observation  
5) **Intent‑gate + Score** — reject wrong query type; score by semantics + compactness  
6) **Repair (optional)** — one‑shot fix using DB error; re‑exec  
7) **Fallback** — deterministic few‑shot if all else fails

**Key functions**
- `generate_candidates(...)` → raw SQL strings  
- `clean_candidate(...)` → `SELECT ... FROM ...;` or reject  
- `projection_guard(...)` → minimal projection baseline  
- `enforce_projection_contract(...)` → drop extra SELECT columns when NLQ lists fields  
- `canonicalize_table_casing(...)` → fix `ORDERS` → `orders`  
- `apply_clamps(...)` → strip ORDER/LIMIT, trim columns, add missing GROUP BY when asked  
- `intent_constraints(...)` → reject wrong query type (list vs aggregate vs grouped vs top‑k)  
- `repair_sql(...)` → uses DB error message to attempt one fix  

**Success rule**
- A query is accepted only if it **executes** and **passes intent constraints**, then wins the **score**.

**Explainable story (1 line)**
Generate → clean → post‑process → execute → intent‑gate → score → repair/fallback.
