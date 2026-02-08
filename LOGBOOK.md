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
- **Additional fixes (same day):** Forced finish after a successful run to prevent post‑run tool noise; blocked setup tools inside the loop; made schema validation explicit (`schema_ok`/`schema_missing`); expanded COUNT cues in constraint extraction; re‑added the interactive walkthrough cell; added guardrail stage debug output; set sanity checks to `auto_order=True`; hardened constraint handling to ignore non‑dict inputs.  
- **Next:** Re-run sanity checks after a kernel restart to confirm cleaner tool order and reduced trace noise.

### 2026-02-07 — Quick Check Error Log (Tool-Driven ReAct)
- **Activities:** Reviewed 10-item quick check outputs in `results/agent/results_react_200 (4).json`; classified errors and failure causes.
- **Results (pre-fix):** VA 1.00, EX 0.30, EM 0.20, TS 0.30 (3/10 correct).
- **Results (post-fix, 10 items):** VA 1.00, EX 0.70, EM 0.50, TS 0.70.
- **Error taxonomy:** Off-topic fallback reuse (5/10; “San Francisco employee count” query used for unrelated NLQs); projection/field selection errors (2/10); EM-only mismatch from alias/ORDER BY differences (1/10; EX/TS still correct).
- **Diagnostics:** Fallback triggered 6/10 items; in 5 cases it replaced earlier candidates and yielded wrong semantics. Blocked steps totaled 15 across the set, suggesting control-flow interruptions precede fallback misuse.
- **Justification (lit):** Execution-guided decoding rejects faulty programs using execution feedback, aligning with gating fallback candidates through validation/execution before accepting them. [Robust Text-to-SQL Generation with Execution-Guided Decoding](https://www.microsoft.com/en-us/research/publication/robust-text-to-sql-generation-with-execution-guided-decoding/)
- **Next:** Prevent fallback from overwriting validated candidates; add a guard that retains the best schema-validated SQL when fallback fires.

### 2026-02-07 — Literature‑Backed Error‑Reduction Plan (v1)
- **Trigger:** 50‑item slice in `results/agent/results_react_200 (5).json` showed projection/filter mismatch as the dominant EX failure, followed by join/table mismatch, with smaller invalid‑SQL and fallback‑reuse tails.
- **Gate fallback + candidate acceptance with execution and value/field constraints** — execution‑guided decoding shows that rejecting faulty programs via execution feedback improves text‑to‑SQL accuracy; this directly supports hard‑gating fallback and enforcing explicit NLQ fields/values. [Robust Text-to-SQL Generation with Execution-Guided Decoding](https://www.microsoft.com/en-us/research/publication/robust-text-to-sql-generation-with-execution-guided-decoding/)
- **Relation‑aware schema linking to reduce join/table errors** — RAT‑SQL demonstrates that explicit relation encoding and schema linking materially improves cross‑schema generalization. [RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers](https://aclanthology.org/2020.acl-main.677/)
- **Decouple schema linking from skeleton generation** — RESDSQL shows that separating schema item selection from SQL skeleton parsing reduces schema confusion in complex queries. [RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL](https://arxiv.org/abs/2302.05965)
- **Constrained decoding to reduce invalid SQL** — PICARD constrains decoding via incremental parsing, improving validity on Spider/CoSQL; aligns with reducing VA failures before execution. [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding](https://aclanthology.org/2021.emnlp-main.779/)
- **Maintain tool‑grounded ReAct loop for interpretability** — ReAct formalizes interleaved reasoning/action/observation, providing a principled basis for the tool‑driven loop. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **Add reflection memory for repeated errors** — Reflexion shows verbal feedback memory improves agent decisions without retraining; supports caching recent error patterns. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- **Evaluation framing** — Spider formalizes cross‑domain text‑to‑SQL and motivates EM/EX as primary metrics for semantic correctness and generalization. [Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL](https://aclanthology.org/D18-1425/)

### 2026-02-07 — Phase 1 Implemented: Hard Gates for Explicit Fields + Value Hints
- **Change:** Added `explicit_fields` to constraint extraction and enforced it in `validate_constraints` (tool‑driven loop). This hard‑rejects candidates missing explicitly requested columns (e.g., “names, codes, and MSRPs”).  
- **Change:** Tightened value‑hint extraction by adding numeric literals and excluding common “instruction” words (e.g., Top/Most), then enforced missing value hints as a hard gate.  
- **Change (agent path):** Added explicit‑field and value‑hint hard gates in `ReactSqlAgent.evaluate_candidate` so fallback and candidate acceptance use the same constraints.  
- **Justification (lit):** Execution‑guided decoding supports rejecting candidates that fail semantic constraints before acceptance, and constrained decoding literature supports enforcing structure/constraints early to prevent invalid or mis‑shaped SQL. [Robust Text-to-SQL Generation with Execution-Guided Decoding](https://www.microsoft.com/en-us/research/publication/robust-text-to-sql-generation-with-execution-guided-decoding/) [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding](https://aclanthology.org/2021.emnlp-main.779/)

### 2026-02-07 — Phase 2 Implemented: Relation‑Aware Linking + Join‑Key Validation
- **Change:** Added relation‑aware boosts in schema linking so tables connected by known joins are more likely to be included in the pruned schema context.  
- **Change:** Added join‑key validation that rejects SQL missing expected key joins (e.g., `customers.customerNumber = orders.customerNumber`) when both tables appear.  
- **Justification (lit):** Relation‑aware schema encoding/linking improves join accuracy (RAT‑SQL), and decoupling schema linking from SQL skeleton parsing improves schema selection in complex queries (RESDSQL). Join‑key validation operationalizes these insights by enforcing expected relational structure. [RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers](https://aclanthology.org/2020.acl-main.677/) [RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL](https://arxiv.org/abs/2302.05965)

### 2026-02-07 — Phase 3 Implemented: Constrained Decoding + Reflection Memory
- **Change:** Added semicolon‑stopping criteria in LLM generation for tool calls to reduce run‑on text and enforce single‑statement decoding.  
- **Change:** Added lightweight reflection memory in the class‑based agent: recent validation errors are appended to observations and repair prompts to discourage repeated mistakes.  
- **Justification (lit):** Constrained decoding (PICARD) reduces invalid SQL by restricting generation to syntactically valid forms; reflection memory follows Reflexion‑style feedback to improve iterative repairs without retraining. [PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding](https://aclanthology.org/2021.emnlp-main.779/) [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

### 2026-02-08 — Schema Item Ranking + Projection Hint Expansion
- **Change:** Expanded explicit field synonyms (e.g., order date, payment date, check number, office code) to catch more projection cues in NLQs.  
- **Change:** Added soft projection hints (entity defaults + explicit fields) and applied column‑level ranking inside `link_schema` so only top‑K columns are surfaced per table, while forcing join keys to remain visible.  
- **Why it helps:** Most remaining EX errors were projection mismatches. Column ranking narrows the schema space before decoding, which is consistent with relation‑aware schema linking and decoupled schema selection that improve schema accuracy.  
- **Refs:** `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`.

### 2026-02-08 — Lightweight Value Linking (Column Hints)
- **Change:** Added value‑to‑column linking via NLQ context + value patterns (dates, location phrases, ID‑style phrases). The linker now produces **value‑column hints** used to rank columns and guide schema pruning.  
- **Why it helps:** The remaining EX errors included filter/value mismatches. Value‑column hints bias the model toward the correct WHERE columns without requiring database lookups.  
- **Refs:** `REFERENCES.md#ref-lin2020-bridge`, `REFERENCES.md#ref-wang2020-ratsql`.
