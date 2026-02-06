# Logbook (Research & Development Timeline)


---

## Phase 1 — Planning & Scoping (Sep–Oct 2025)


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
- **Learning:** Prompting alone fixes form, not meaning — the semantic gap is the real bottleneck.
- **Next:** Test whether weight adaptation addresses semantic gap.

### 2025-12-23 — Full Baseline Results (200 items)
- **Results:**  
  - k=0 → VA 0.810 / EX 0.000  
  - k=3 → VA 0.865 / EX 0.250
- **Lit Alignment:** Matches literature: exemplars induce schema patterns but not numeric reasoning or compositional semantics.
- **Learning:** Few-shot helps structure, but deeper reasoning still fails without adaptation.

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

### 2026-01-23 — Micro-Slice Stability Check
- **Activities:** Projection guard + ORDER/LIMIT clamp; slice reached VA/EX/EM = 1.0.
- **Lit Link:** Consistent with constrained decoding (PICARD-style) literature.
- **Insight:** Some EX failures are structural, not semantic.
- **Learning:** Tight guardrails can fix structural errors, but they don’t solve semantics.

### 2026-01-25 — Full-Set Reality Check
- **Results:** VA ~1.0, EX ~0.05
- **Failure Modes:** revenue/grouping, multi-hop joins, misaligned filters.
- **Lit Link:** Execution guidance insufficient for semantic leaps; matches survey findings.
- **Reflection:** We entered “semantic bottleneck” regime.
- **Learning:** Execution feedback stabilises validity, but meaning errors persist at scale.

### 2026-01-26 — Literature-Aligned Diagnosis
- **Conclusion:**  
  - Execution = robustness  
  - QLoRA = semantics  
  - Prompting = grounding  
- **Lit Alignment:** Mirrors Spider error analyses and ReAct failure analyses (Yu et al., 2018; Yao et al., 2023; Zhu et al., 2024).
- **Next Steps (lit-driven):**  
  1) strengthen semantic curriculum (joins/aggregates)  
  2) add critic/reranker (surveyed in Zhu et al., 2024; Gao et al., 2025)  
  3) schema linking enhancements (RESDSQL-style; Li, Zhang, Li, and Chen, 2023 (RESDSQL))

### 2026-01-27 — Staged Decision Process (minimal → clamp → rerank → repair)
- **Activities:** Re-structured the notebook into staged ablations (STAGE 0–3) to restore validity before re‑introducing complexity.
- **Motivation:** Debug evidence showed valid SQL was being discarded by downstream filters; a minimal execution‑gated generator isolates the bottleneck.
- **Lit Link:** Start with execution‑guided decoding only (Zhong, Yu, and Klein, 2020), then add constraints (PICARD‑style; Scholak, Schucher, and Bahdanau, 2021 (PICARD)), then reranking/critics (Zhu et al., 2024; Gao et al., 2025), then repair (Zhai et al., 2025).
- **Outcome:** Made the pipeline evidence‑driven rather than feature‑driven; each component is now re‑introduced only after validation.

### 2026-01-29 — Stage 3 Outputs + Trace Logging Upgrade
- **Observation:** Stage 3 outputs are mostly valid SQL; remaining issues are projection bloat and unnecessary ORDER BY/GROUP BY that reduce EM (e.g., extra `productLine`, spurious `ORDER BY` on totals).
- **Activities:** Added structured trace logging to the ReAct loop (raw candidate → cleaned SQL → post‑clamp SQL → execution error → repair attempt).
- **Reflection Logging:** `reflect_sql` now returns both the reflected SQL and a metadata dict (status, raw_fix, exec_error) so traces show why a reflection succeeded or failed.
- **Reason:** Traceability is needed to attribute errors to generation vs cleaning vs execution vs repair; aligns with agentic evaluation practice in ReAct/Reflexion‑style loops.
- **Code:** `notebooks/03_agentic_eval.ipynb`

### 2026-01-30 — Stage‑3 Full Run (200 queries)
- **Run:** Stage‑3 ReAct on 200 ClassicModels NLQs (`results_react_200`).
- **Metrics:** VA **0.805**, EX **0.13**, EM **0.105**.
- **Interpretation (VA):** strong syntactic control; filtering + clamps + execution gating are working as intended.
- **Interpretation (EX):** semantic alignment remains the bottleneck (aggregation scope, join selection, projection granularity).
- **Interpretation (EM):** low EM is expected under agentic post‑processing and query rewrites; EM is diagnostic, not primary.
- **Literature alignment:** matches reports that execution feedback stabilises validity but does not guarantee semantic correctness (Ojuri et al., 2025; ExCoT; execution‑guided decoding).
- **Next steps logged:** introduce **Test‑Suite Accuracy (TS)** and a structured **error taxonomy** (projection, aggregation scope, join selection) to explain EX failures.
- **Dissertation narrative hook:** this run is the first **full‑set, agentic** result that isolates the “execution‑valid vs semantically‑correct” gap. It provides a concrete anchor for claims about why execution guidance improves stability but requires stronger semantic grounding or critic signals for EX gains.
   
### 2026-01-31 — Candidate‑Ranking Utilities (EX‑Focused Testing)
- **Activities:** Tested which agent utilities materially improved EX within a candidate‑ranking loop (generate many → score/filter → execute best).  
- **Guards/Utilities:** cleaning + normalization, projection guards, ORDER/LIMIT clamps, intent constraints, schema‑aware validation, `semantic_score`, `missing_explicit_fields`, prefilter to `max_exec_cands`, reflection/repair as fallback.  
- **Observation:** These utilities boosted VA and made EX failure modes legible, but the loop remained post‑hoc ranking rather than tool‑grounded ReAct.  
- **Reason:** Needed to isolate which controls affected EX before re‑architecting the loop.  
- **Outcome:** Evidence base for which utilities should become first‑class tools in February.  
- **Carried forward (Feb):** cleaning/normalization, schema‑aware validation logic, projection/clamp guardrails, semantic scoring signals, and reflection logic.  
- **Not carried forward:** candidate‑ranking as the control structure; hard intent rejection (softened later); tabular prompt variant.  
- **Learning:** Utility gains were real, but the *loop structure* still hid error causes.

### Late January 2026 Summary (Jan 23–31)
- **What dominated:** rapid iteration on the agent loop, safety checks, and evaluation.  
- **Why so much iteration:** many small, testable changes were needed to see where the system failed.  
- **What I learned:** running the SQL helps prevent invalid output, but “meaning” errors (wrong joins/aggregations) still persist without stronger guidance.  
- **Outcome:** a stable, debuggable loop with clear stages and a full‑run anchor on 200 items.

### January 2026 Summary (Month‑Level)
- **Phase shift:** moved from QLoRA results into agent‑loop refinement with execution feedback.  
- **Key insight:** prompts fix syntax, QLoRA helps meaning, and execution checks stabilize validity — but none alone solves semantic alignment.  
- **Methodological takeaway:** explicit tool boundaries and clear checks are necessary to make the process explainable and defensible.  
- **Evidence:** the full‑set run (Jan 30) exposed a gap between “runs correctly” and “answers the question,” motivating tighter guidance.

### Transition Note (Late Jan → Early Feb)
- **Realization:** I was still picking the best from a batch of answers rather than guiding the model step‑by‑step with tools.  
- **Change (Early Feb):** I rebuilt the loop into a clear, tool‑driven process where each step is checked and logged.  
- **Learning:** A true ReAct loop requires explicit actions + observations, not just better ranking.

### 2026-02-01–02 — ReAct Rebuild Plan + Evaluation Cleanup
- **Activities:** Turned the old safety checks into a step‑by‑step plan; simplified the notebook; added quicker tests; clarified how we judge correctness so it focuses on meaning, not formatting.  
- **Reason:** Make mistakes visible, speed iteration, and keep evaluation focused on the answer rather than SQL style.  
- **Outcome:** Clear rebuild plan and a cleaner, faster evaluation setup.  
- **Learning:** Visibility and evaluation design were prerequisites for meaningful progress.

### 2026-02-04–05 — Tool‑Driven Loop Implemented
- **Activities:** Implemented the step‑by‑step loop with checks for schema, structure, and execution; forced a fix‑and‑retry step on failure; improved feedback and logging; enabled multi‑step refinement.  
- **Dropped/Changed:** Removed a confusing prompt format and softened overly strict intent checks.  
- **Effect:** A more stable, auditable loop with clearer reasons for success and failure.  
- **Learning:** Explicit tool order + forced repair yields a more explainable and reliable agent loop.

### 2026-02-06 — Quick Sanity Check + Action Parsing Fix
- **Activities:** Ran a small sanity set of ClassicModels questions through the step‑by‑step walkthrough to check the end‑to‑end flow (draft → checks → run → refine).  
- **Sanity set examples:** “List all product lines”, “Which customers are in the USA?”, “Total amount per order number”, “Count employees in the San Francisco office”.  
- **Observation:** The SQL answers were often correct, but the trace looked inconsistent (e.g., unexpected late schema calls and out‑of‑order tool attempts).  
- **Diagnosis:** The model sometimes emitted **multiple `Action:` blocks** in one response (especially when it “replayed” the transcript). A greedy action parser could capture from the first `[` to the last `]`, causing the loop to follow the **wrong** action.  
- **Change:** Updated action parsing to extract **all** `Action:` lines and follow the **last** one (the model’s final decision). This made the loop more stable and the walkthrough easier to explain.  
- **Also fixed:** Corrected a small regex escape in table-casing normalization so guardrail traces reflect the schema consistently.  
- **Learning:** Small “plumbing” details (parsing + logging) can dominate perceived agent quality; reliable tool boundaries are part of reproducibility.  
- **Next:** Re-run sanity checks after a kernel restart to confirm cleaner tool order and reduced trace noise.
