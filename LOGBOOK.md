# Logbook

## Phase 1 — Planning & Scoping (Sep–Oct 2025)

### 2025-09-29
- Activities: Met supervisor to align scope; drafted project outline; read core references on agentic NL-to-SQL; captured framework notes.
- Challenges: Scope felt broad; state-of-the-art section thin.
- Insights: Clarified how PEFT, QLoRA, ReAct fit the project; identified improvement areas in seed paper.
- Next Steps: Broaden sources beyond seed paper; justify chosen frameworks; share outline with supervisor.

### 2025-10-06
- Activities: Expanded outline (~2k words); started ethics form; set up OneDrive/Teams; pulled more literature.
- Challenges: Over-reliance on seed sources; structure of SOTA unclear.
- Insights: Supervisor recommended Zotero; explored alternative agentic LLM methods (ReAct, knowledge graphs).
- Next Steps: Diversify citations; log everything in Zotero; finish ethics submission.

### 2025-10-13
- Activities: Submitted ethics; added 500 words linking literature to implementation; adjusted tone after lecture review.
- Challenges: Worried SOTA is descriptive, not critical.
- Insights: Zotero export saves time; deeper reading links theory to planned pipeline.
- Next Steps: Add critique of trade-offs; map literature to implementation milestones; enforce IEEE style.

### 2025-10-20
- Activities: Refined outline per feedback; added OSS vs proprietary comparison; SMART objectives and MoSCoW; 13-week timeline; defined metrics (VA/EX/TS); noted Colab fallback.
- Challenges: Balancing feedback with clarity on evaluation rationale; security/compute feasibility.
- Insights: “Reproducibility gap” is core contribution; hybrid approach (prompt + ReAct + constraints) needed.
- Next Steps: Submit outline; start practical setup; read Ojuri et al. and ReAct papers in full.

### 2025-10-27
- Activities: Reviewed outline; identified colab-llm as reference; weekly checklist; minimal test notebook; pivoted SQLite→Cloud SQL; set secure connector (creator + SQLAlchemy).
- Challenges: Adapting colab-llm for fine-tune/ReAct; SQLite MySQL incompatibility; hardcoded secrets.
- Insights: Cloud SQL connector avoids IP allowlisting/SSL.
- Next Steps: Verify Cloud SQL data; load Llama-3-8B 4-bit; run baseline prompts; record hardware specs; finish Ojuri/ReAct reading.

## Phase 2 — Infra & Data Prep (Nov–Dec 2025)

### 2025-11-03
- Activities: Built Colab scaffold (connector, safe_connection, schema introspection, QueryRunner); smoke-tested customers table.
- Challenges: Credentials still hardcoded; time constrained by coursework.
- Insights: Connector simplifies secure access; QueryRunner maps to ReAct actions.
- Next Steps: Load/evaluate models; deepen ReAct/Ojuri reading; assemble dataset.

### 2025-11-10
- Activities: Regrouped; reviewed timeline; planned data prep.
- Challenges: Shifting from infra to conceptual (data/ReAct) needs stronger theory.
- Insights: Fine-tuning likely beats few-shot for domain; theory first.
- Next Steps: Study fine-tune/ReAct; plan data generation.

### 2025-11-17
- Activities: Completed QLoRA plumbing; curated ~100 NLQ-SQL pairs; structured dataset; aligned QueryRunner with ReAct.
- Challenges: bitsandbytes instability on Colab GPUs; large model load issues.
- Insights: QLoRA 4-bit is feasible but fragile; QueryRunner essential for Act step.
- Next Steps: Load dataset for SFT; expand training data; load target model.

### 2025-12-02
- Activities: Reviewed notebook scaffold (GCP auth, connector, QueryRunner, schema helpers, Llama-3-8B load); drafted repo structure docs (ARCHITECTURE, CONFIG, DATA, NOTES); generated 200 NL-SQL pairs (ClassicModels).
- Insights: Need few-shot baseline (as in Ojuri) before QLoRA; keep QueryRunner as ReAct tool; log traces.
- Next Steps: Implement few-shot baseline.

### 2025-12-04
- Activities: Ran `validate_test_set` on 200 queries — all pass.
- Challenges: Needed kernel restart/import order; VS Code env vars.
- Insights: Test set is clean end-to-end.
- Next Steps: Pin deps; run few-shot baseline; start QLoRA prep.

### 2025-12-05
- Activities: Pinned deps in requirements.txt; updated installs; re-verified validation.
- Challenges: VS Code cache confusion.
- Insights: Locked deps aid reproducibility.
- Next Steps: Run few-shot baseline; lock QLoRA hyperparams.

### 2025-12-06
- Activities: Synced Colab; fixed connector/crypto pins & NumPy; set ADC/quota; installs now use requirements.txt; CONFIG shows env examples.
- Challenges: Cloud SQL auth friction.
- Insights: Always `git pull` in Colab; set env vars explicitly.
- Next Steps: Run few-shot baseline on GPU; log VA/EX; prep QLoRA SFT.

### 2025-12-07
- Activities: Documented gated-model loading, NF4 setup, pin rationale; validation still 200/200.
- Challenges: HF gated access + Colab wheel drift.
- Insights: Deterministic decoding + 4-bit load needed for reproducible evals.
- Next Steps: Run few-shot baseline; capture VA/EX; begin QLoRA SFT prep.

### 2025-12-12
- Activities: Added `triton==2.2.0`; refreshed Colab workflow (clean reclone); confirmed 4-bit Llama-3-8B-Instruct loads on cuda:0 with deterministic defaults.
- Challenges: Sampling warnings when mixing do_sample=False with temp/top_p.
- Insights: For VA/EX, keep deterministic; sampling only for exploration.
- Next Steps: Run schema-grounded few-shot baseline; log hardware/commit/prompt.

### 2025-12-14
- Activities: Added inference-time few-shot prompt pipeline (system+schema+exemplars); deterministic decode; SQL post-process (first SELECT; minimal projection); schema ordering.
- Challenges: Zero-shot executable but EX low; occasional instruction echo.
- Insights: Few-shot + ordered schema + post-process improves correctness without fine-tuning.
- Next Steps: Run full 200 VA/EX baseline; log commit/prompt/hardware.

- Research Commentary (Few-Shot Baseline): Hypothesis—exemplars cut structural SQL errors (Spider/Ojuri pattern). Observation—VA up, EX still low: exemplars help form, not intent. Interpretation—prompt-only struggles on aggregation/conditional joins. Implication—fine-tuning should yield bigger EX gains than more prompting.
### 2025-12-23
- Activities: Batch eval loop for k=0/k=3 on subsets then full 200 in `02_baseline_prompting_eval.ipynb`; outputs under `results/baseline/`.
- Challenges: Slow generation; strict EX hurts scores.
- Insights: Few-shot improved VA/EX (k=0 VA 0.810/EX 0.000; k=3 VA 0.865/EX 0.250).

## Phase 3 — Baselines & QLoRA (Jan 2026)

### 2026-01-06
- Activities: Refactored baseline eval into `nl2sql/`; Colab runner notebook; exemplar-leakage guard.
- Challenges: Avoiding large outputs in Git; need explicit export/download.
- Insights: Separating runner vs harness reduces drift; enables fair comparisons.
- Next Steps: Re-run full baselines; start QLoRA fine-tuning with same harness.

### 2026-01-09 — QLoRA run 1
- Activities: QLoRA r=16, 1 epoch, 4-bit on 200 train; eval on 200 test.
- Results: k=0 VA 0.73 / EX 0.03; k=3 VA 0.86 / EX 0.305.
- Take: Adapters didn’t beat prompt few-shot; need more capacity/steps and cleaner exemplars.

### 2026-01-12 — QLoRA run 2 (larger)
- Activities: r=32, α=64, 3 epochs, warmup; re-eval.
- Results: k=0 VA 0.865 / EX 0.065 / EM 0.000; k=3 VA 0.875 / EX 0.380 / EM 0.305.
- Take: k=3 EX now beats prompt baseline; k=0 still weak → adapters rely on exemplars.

### 2026-01-14 — Agentic plan
- Activities: Sketched ReAct loop and `03_agentic_eval.ipynb` (Thought→Action→Observation→Refinement with QueryRunner); short deterministic prompts.
- Take: Agentic refinement is next lever for EX/TS, esp. at k=0. Plan to compare base vs QLoRA in loop.

- Research Commentary (QLoRA Fine-Tuning): Hypothesis—adapters improve NL→SQL semantics (Mosbach, QLoRA, NL→SQL surveys). Result—EX improved for k=3 but stayed low for k=0; adapters internalise structure yet still lean on ICL schema anchoring. Interpretation—fine-tuning gives structure; prompting gives context; they are complementary. Constraint—aggregation/grouping remain failure cases.
### 2026-01-18 — Dependency wobble fix
- Activities: Added clean setup cell (torch/cu121 + bnb/triton); kept adapter fallback; stopped ignoring `results/`.
- Challenges: Colab wheel drift.
- Insights: Fresh runtime + restart after setup; fallback to base if adapters absent.
- Next Steps: Upload/regenerate adapters; run agentic loop end-to-end.

## Phase 4 — ReAct Exploration & Stabilisation (Jan 2026)

### 2026-01-19 — Zero-VA/EX root cause
- Activities: Found ReAct failures due to non-SELECT junk; tightened instructions, hardened `extract_sql`, deterministic decode, added 5-item quick check.
- Insights: Adapters fine (QLoRA k=3 EX ~0.38); issue was prompt/loop, not data.
- Next Steps: Run small slice; if VA>0, expand to full set.

### 2026-01-20 — Column mismatch sanity check
- Activities: Quick check now VA true; EX fails on projection/order mismatches.
- Insight: Strict EX penalises extra/misordered columns; row sets correct.
- Takeaway: Align projections to gold; use row-set comparison for diagnostics.

### 2026-01-21 — Alignment with Ojuri et al.
- Activities: Stabilised 5-item slice (tight prompt, deterministic decode, adapter check); kept default `test_set=full_set[:5]`.
- Insight: Iterative refinement mirrors Ojuri’s agent; next gains from retries, projection guard, beam/rerank, grammar check, trace logging.
- Plan: Run full 200 after slice stable; add TS proxy + McNemar-style paired test if time.
- Research Commentary (ReAct Small Slice): Hypothesis—execution feedback would cut syntax/unknown-column errors and yield join repairs. Observation—VA gains via execution filtering; EX gains minimal because many wrong-but-executable queries persist. Interpretation—execution supervision alone is insufficient when the error surface is semantic. Next question—can cheap semantic signals substitute for full agentic reasoning?

### 2026-01-22 — Small-slice executes, EX needs projection fixes
- Activities: Stripped prompt tokens before decode; added error logging. Slice VA=1.0, EX=0.2 (extra/misordered cols, one hardcoded filter).
- Insights: Execution works; misses are projection/aggregation. Need prompt reminder and projection guard.
- Next Steps: Tighten prompt + exemplars; add projection guard; rerun slice then full set.

### 2026-01-23 — Projection guard applied
- Activities: Added projection guard (minimal SELECT) + tightened prompt; slice hit VA/EX/EM = 1.0.
- Insight: EX drift was projection/order; guard fixed it.
- Next Steps: Apply to full set; consider rerank/beam if EX drops.

### 2026-01-24 — Helper tidy
- Research Commentary (Full-Set Agentic Run): VA at ceiling (~1.0) but EX low (~0.05–0.10). Interpretation—execution correctness saturated; semantics are bottleneck. Literature (Spider/EG-SQL) shows dominant errors here are wrong joins, missing aggregation/grouping, and filter semantics—executable yet wrong. Conclusion—agent must encode semantics, not just syntax.
- Activities: Silenced HF warnings; set pad_token; reduced candidates for speed; enforced ORDER/LIMIT clamp & SELECT-only filter; ensured QueryRunner defined; progress print every 5; LIMIT for slices.
- Insight: Stable runs, no NameErrors, manageable latency.

### 2026-01-25 — First full-set ReAct with guards
- Activities: Full 200 run with strict prompt + projection guard + retry. VA=1.0, EX≈0.05, EM≈0.025 (`results/agent/results_react.json`).
- Findings: Wrong joins/aggregates, extra cols/ORDER, hallucinations; guard too narrow for full set.
- Next Steps: Stronger clamps, rerank, broader retries.

### 2026-01-26 — Consolidation (retry, repair, fallback)
- Activities: Added execution-guided retry for unknown columns; one-shot repair prompt; filtered non-SELECT/markdown junk; 3 candidates/step; deterministic decode main pass, sampling in repair; deterministic few-shot fallback when no SELECT; fixed quick-check to strip prompt; added semantic reranker + error taxonomy helpers in `nl2sql/agent_utils.py` (clean_candidate, tabular prompt variant, lexical semantic_score, repair hints).
- Research Commentary (agent_utils Layer): Motivation—tackle “executable but wrong” via intent-aware rerank + baseline fallback. Design—SELECT-only filter; tabular prompt for join reasoning; semantic rerank; MySQL error taxonomy + hints; deterministic non-agentic candidate as lower bound. Rationale—avoid full Thought/Action/Observation (model too small) but try to lift EX without retraining.

### Theoretical Trace (stage framing)
- Literature ladder: Prompting → recovers syntax/schema; Fine-tuning → recovers semantic mapping; Agentic refinement → repairs via tool feedback. Project now follows Prompt → QLoRA → Execution-Guided Agent.

### Reflection (current state)
- Prompting solves syntax, not semantics.
- QLoRA solves structure, not reasoning.
- Execution feedback solves stability, not intent.
- Aligns with NL→SQL literature; motivates future true ReAct (explicit Thought/Action/Observation, SCHEMA_LOOKUP/EXEC_SQL tools) and multi-step semantic reasoning.
- Insight: Stability up (no empty pred_sql, fewer junk candidates) but EX still low on full set. Expect gains from richer rerank/repair plus stronger base model (QLoRA). Baseline/QLoRA remain primary; ReAct still exploratory.
- Rationale (for dissertation): agent_utils was introduced to counter the “executable but wrong” pattern—enforcing single-SELECT output, seeding a deterministic baseline candidate so the agent never underperforms prompt-only, steering repairs with an error taxonomy, and reranking by intent cues (aggregates/joins/grouping) rather than projection size. This keeps improvements in the eval loop without touching training.

### Dissertation narrative (cross-cutting)
- Journey shows three clear stages to discuss: (1) Prompt-only baseline: executable but semantically weak (VA high, EX low). (2) QLoRA adapters: structural lift in NL→SQL mapping (EX improves, especially with k=3 exemplars). (3) Agentic refinement: execution-guided stability plus intent-aware reranking/repairs to recover EX without extra training. This progression is the core “investigator story” for the evaluation chapter.
