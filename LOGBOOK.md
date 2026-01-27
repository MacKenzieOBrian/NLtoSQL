# Logbook (Research & Development Timeline)

This logbook documents the research trajectory across four phases, emphasising
(1) literature-aligned decision making,
(2) hypothesis-driven experimentation, and
(3) evaluation-centred interpretation rather than implementation logs.

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

### 2026-01-31 — Dev Note (SELECT query echo)
- **Issue:** Model echoed instruction text (“SELECT query…”) which passed the old filter, yielding invalid-but-accepted SQL.
- **Fix:** Enforced `SELECT … FROM …` in `clean_candidate` and fallback; removed “SELECT query” phrasing from prompt.
- **Effect:** Rejects instruction echoes and restores valid SQL on simple queries.

---

## Current State (Jan 2026)

- Prompting fixes syntax
- QLoRA fixes semantics
- Execution fixes robustness
- Critic/curriculum expected to fix compositional reasoning

The dissertation narrative can legitimately focus on whether **small open models + PEFT + execution** can approximate proprietary agent performance under reproducible resource constraints.

---
