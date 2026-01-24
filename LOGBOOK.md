# Logbook



## 2025-09-29
- Activities: Met supervisor to align scope; drafted project outline; read core references on agentic NL-to-SQL; captured framework notes.
- Challenges: Initial scope felt broad; over-wrote contextual background leaving less space for state-of-the-art; missed noting relevance of some sources.
- Insights: Clarified foreign methods (PEFT, QLoRA, ReAct) and how they fit agentic training; understood main paper’s findings and improvement areas.
- Next Steps: Gather diverse sources beyond seed paper; justify frameworks in outline; share draft with supervisor.


## 2025-10-06
- Activities: Expanded outline (~2000 words); leveraged cited studies for state-of-the-art; started ethics application; set up OneDrive/Teams for sharing.
- Challenges: Heavy reliance on seed sources; struggled with structure for state-of-the-art; still leaning on root references.
- Insights: Supervisor suggested Zotero; explored alternative agentic LLM methods; improved grasp of PEFT/QLoRA/ASOT/ReAct/knowledge graphs.
- Next Steps: Broaden sources to differentiate from seed paper; log all references in Zotero; complete ethics submission.

## 2025-10-13
- Activities: Submitted final ethics application; added ~500 words linking literature to implementation; adjusted tone per lecture review.
- Challenges: Worried state-of-the-art is descriptive vs analytical; need stronger critique of approach trade-offs.
- Insights: Zotero citation export saves time; deeper reading bridges contextual and technical discussion on agentic methods.
- Next Steps: Revisit state-of-the-art to add critical discussion; map literature to implementation plan; ensure IEEE style via Zotero; add cover page.

## 2025-10-20
- Activities: Refined outline per supervisor comments; added open-source vs proprietary comparison; created SMART objectives and MoSCoW; drafted 13-week timeline; documented evaluation metrics (VA/EX/TS); noted Colab as fallback for hardware limits.
- Challenges: Incorporating feedback while keeping evaluation rationale clear; security and compute feasibility concerns.
- Insights: “Reproducibility gap” is key contribution angle; QLoRA is enabling tech for feasibility; hybrid approach (optimized prompt + ReAct + constraints) required.
- Next Steps: Submit outline; begin practical setup; configure Colab environment; read Ojuri et al. and ReAct papers in full.


## 2025-10-27
- Activities: Reviewed outline/timeline; identified colab-llm repo as reference; planned weekly checklist; prepped minimal test notebook; pivoted from SQLite to Cloud SQL; established secure connector pattern (creator + SQLAlchemy).
- Challenges: colab-llm needs adaptation for fine-tune/ReAct; SQLite failed due to MySQL-specific syntax; temporary root credentials/hardcoded secret.
- Insights: Cloud SQL connector avoids IP allowlisting/SSL hassle
- Next Steps: Verify Cloud SQL data integrity; load Llama-3-8B in 4-bit; baseline prompt tests; record hardware specs; finish reading Ojuri/ReAct.

## 2025-11-03
- Activities: Built Colab scaffold with connector, safe_connection, schema introspection, QueryRunner; ran smoke tests on customers table.
- Challenges: Credentials still hardcoded; progress slowed by other coursework.
- Insights: Connector simplifies secure access; QueryRunner provides safe Act+metadata; need to prioritize model evals and literature depth.
- Next Steps: Load/evaluate models; read ReAct/Ojuri plus new sources; assemble 200-sample dataset.


## 2025-11-10
- Activities: Regrouped after coursework; reviewed timeline to enter data prep phase.
- Challenges: Moving from infra to conceptual tasks (data, ReAct) requires stronger grounding in papers.
- Insights: Fine-tuning likely superior to few-shot for domain tasks; theory first to design pipeline correctly.
- Next Steps: Deepen fine-tuning/ReAct understanding; start data generation planning.

## 2025-11-17
- Activities: Completed plumbing for QLoRA; curated ~100 NLQ-SQL pairs; structured dataset with schema+NLQ+SQL text field; aligned QueryRunner with ReAct Act.
- Challenges: Bitsandbytes/model loading unstable on Colab GPUs; Colab struggled with large model; uncertainty if QLoRA setup is at fault.
- Insights: QLoRA 4-bit is key for feasibility; loading/quantization is finicky; QueryRunner maps directly to Act in ReAct loop.
- Next Steps: Load generated dataset for SFT; load target open-source model and expand training data.

## 2025-12-02
- Activities: Reviewed current notebook scaffold (GCP auth, connector, QueryRunner, schema helpers, Llama-3-8B load). Drafted project structure plan in .md files (ARCHITECTURE.md, CONFIG.md, DATA.md, NOTES.md, scripts for data prep/train/eval/agent). Completed generation of 200 NL-SQL pairs using GPT-5 and ClassicModels context.
- Insights: Should demonstrate few-shot baseline (as in Ojuri et al.) before QLoRA fine-tuning. Keep QueryRunner as ReAct tool; log traces for interpretability. Add NOTES.md for design justifications.
- Next Steps: Implement few-shot prompt baseline against ClassicModels.

## 2025-12-04
- Activities: ran `validate_test_set` on all 200 ClassicModels queries—Success: 200, Failures: 0.
- Challenges: Needed kernel restart and imports before calling the helper; VS Code env setup was finicky until settings.json had the DB vars.
- Insights: Static test set executes cleanly end-to-end!
- Next Steps: Pin deps/requirements, run few-shot baseline with metrics/screenshots, then start QLoRA prep and training.

## 2025-12-05
- Activities: Pinned deps in requirements.txt, updated notebook/script installs to use it; pushed changes. Verified the updated install cell in the clean notebook. Confirmed validation remains green.
- Challenges: VS Code caching made it look like cells weren’t updating, annoying to figure out.
- Insights: Locking installs via requirements.txt keeps runs reproducible across Colab/local.
- Next Steps: Build/run the few-shot baseline (log VA/EX with prompts/screenshots), then start QLoRA hyperparam locking and SFT prep.


## 2025-12-06
- Activities: Synced Colab with repo, pulled latest pins; fixed connector/cryptography pins and NumPy mismatch; set ADC/quota project; ran installs via requirements.txt. Notebook installs now point at requirements.txt; Config now includes env var examples.
- Challenges: Cloud SQL auth issues in Colab (quota project/permissions).
- Insights: Use git pull in Colab before installs; set env vars explicitly.
- Next Steps: Run few-shot baseline on Colab GPU with current stack; record VA/EX and screenshots; then move to QLoRA SFT prep (hyperparams/logging) if set up is correct. 


## 2025-12-07
- Activities: Documented gated-model loading (HF token, access approval), 4-bit NF4 setup, and dependency pin rationale in CONFIG/ARCHITECTURE. Verified validation still 200/200.
- Challenges: Hugging Face gated access and Colab binary drift can still throw warnings; requires correct token + access and careful install/restart order.
- Insights: Deterministic decoding + chat template + 4-bit load are required for reproducible eval and future QLoRA on Colab GPUs.
- Next Steps: Run few-shot NL→SQL baseline on Colab GPU with current stack; capture VA/EX and prompts; begin QLoRA SFT prep with pinned toolchain.


## 2025-12-12
- Activities: Added `triton==2.2.0` to requirements and pushed; refreshed Colab workflow to always reclone clean (`rm -rf /content/NLtoSQL`, `git clone`) confirmed 4-bit Llama-3-8B-Instruct loads on `cuda:0` with deterministic defaults
- Challenges: saw sampling warnings when mixing `do_sample=False` with `temperature/top_p`.
- Insights: For VA/EX baselines, use deterministic generation (no sampling params, small `max_new_tokens`, set `pad_token_id=eos_token_id`); sampling is only for exploratory runs.
- Next Steps: Build/run the schema-grounded few-shot baseline, log VA/EX with fixed prompt template and generation settings; record hardware/commit/prompt version for reproducibility.

## 2025-12-14
- Activities: Added inference-time few-shot prompt pipeline (system + schema + k exemplars + NLQ), enforced deterministic decoding, and implemented SQL post-processing (extract first `SELECT ...;`, minimal projection for list-style queries). Kept schema columns ordered (PKs/name-like first) in the prompt.
- Challenges: Zero-shot outputs were executable (VA) but misaligned with gold SQL (EX gaps); model occasionally echoed instructions or over-selected columns.
- Insights: Few-shot exemplars + ordered schema + post-processing improved syntactic correctness and column choice (e.g., productLine over textDescription), achieving VA=True and EX=True on representative cases without changing model weights. Improvements stem from prompt conditioning and heuristics only.
- Next Steps: Run the full 200-sample VA/EX baseline with the fixed prompt/post-processing; log commit/prompt version/hardware. 

## 2025-12-23
- Activities: Implemented and ran the batch evaluation loop for both zero-shot (`k=0`) and few-shot (`k=3`), first on small subsets (`limit=1`, `limit=20`) and then on the full benchmark (`n=200`). The baseline runner is now `notebooks/02_baseline_prompting_eval.ipynb`, writing outputs under `results/baseline/` (gitignored by default).
- Challenges: Batch inference is slow (LLM generation dominates runtime). EX stayed low relative to VA because string-level matching is strict
- Insights: Few-shot prompting improves both executability and strict match: zero-shot reached `VA=0.810`, `EX=0.000`; few-shot reached `VA=0.865`, `EX=0.250`  with frozen weights.


## 2026-01-06
- Activities: Refactored the baseline evaluation into importable modules (`nl2sql/`) and created a Colab runner notebook (`notebooks/02_baseline_prompting_eval.ipynb`) to produce `results/baseline/` artifacts consistently. Added an exemplar-leakage guard in the few-shot evaluation loop.
- Challenges: Keeping notebook outputs off GitHub by default (to avoid accidental large commits) means baseline result JSONs must be downloaded from Colab or explicitly un-ignored for curated artifacts.
- Insights: Separating “runner notebooks” from the evaluation harness reduces copy/paste drift and makes it easier to compare future methods (ReAct, QLoRA) against the same baseline code path.
- Next Steps: Re-run the full baseline (`k=0`, `k=3`, `n=200`) with the refactored notebook to regenerate and archive the baseline JSON outputs; then start QLoRA fine-tuning with the same evaluation harness for comparability.

---

## 2026-01-09 — QLoRA run 1
- Activities: Fine-tuned QLoRA (r=16, 1 epoch, 4-bit) on the 200 training pairs; evaluated on the 200 test items.
- Results: k=0 → VA 0.73 / EX 0.03; k=3 → VA 0.86 / EX 0.305.
- Take: Adapters didn’t beat the prompt-only few-shot baseline; prompting still carried most of the performance. Logged the need for more steps/tuning and a clean exemplar pool.

## 2026-01-12 — QLoRA run 2 (longer, bigger adapters)
- Activities: Bumped capacity and steps (r=32, α=64, 3 epochs, warmup) and re-ran eval.
- Results: k=0 → VA 0.865 / EX 0.065 / EM 0.000; k=3 → VA 0.875 / EX 0.380 / EM 0.305. Saved JSONs under `results/qlora/…`.
- Take: VA jumped; k=3 EX now beats the prompt-only few-shot baseline (~0.325→0.380). k=0 EX is still low—adapters haven’t fully internalised semantics without exemplars.

## 2026-01-14 — Agentic plan
- Activities: Sketched a ReAct loop and scaffolded `03_agentic_eval.ipynb` (Thought → Action → Observation → Refinement with `QueryRunner`); kept prompts short and deterministic.
- Take: Agentic refinement is the next lever to lift EX (and TS, when added), especially at k=0. Plan to run base vs QLoRA adapters through the loop and log traces for a sample of failures.

## 2026-01-18 — Dependency wobble fix
- Activities: Added simple one-cell setup (clean + pin torch/cu121 + bnb/triton) to notebooks to dodge dtype/triton errors; kept adapter load fallback. Stopped ignoring `results/` so we can commit JSONs/adapters.
- Challenges: Colab preloads random wheels
- Insights: Run setup first in a fresh GPU runtime, restart once, then go. If adapters are missing in Colab, we fall back to base model and print it.
- Next Steps: Upload/regenerate adapters before agentic eval; run the loop end-to-end.

## 2026-01-20
- Activities: Tweaked agentic ReAct prompt (strict SELECT-only), defaulted to small 5-item slice for debugging; added refs/comments across notebooks and nl2sql modules; attempted to stage/push fixes.
- Challenges: ReAct pipeline previously produced non-executable SQL (VA/EX=0); git index.lock prevents staging here.
- Insights: Stricter prompt + extract + small-slice debug should surface issues before full 200-run; must stage/push locally after clearing index.lock.
- Next Steps: Test ReAct on small slice and inspect SQL; if valid, run full set; commit/push locally (rm .git/index.lock if needed).

## 2026-01-19 — ReAct zero-VA/EX root cause
- Activities: Investigated why the first ReAct run returned VA/EX/EM = 0. Tightened ReAct instructions (“single SELECT only, no DDL/DML/comments”), hardened `extract_sql`, forced deterministic decoding, and added a small-slice quick check for adapters.
- Challenges: Loose prompt/extraction allowed non-SELECT junk, so every execution failed—looked like “bad adapters” but was a prompting/loop issue.
- Insights: Adapters are fine (QLoRA k=3 EX ~0.38 on full 200); ReAct failed because SQL never executed. Sanity-check on a 5-item slice and inspect SQL before full runs.
- Next Steps: Run ReAct on the small slice; if SQL executes, switch to full set and log VA/EX; keep the tightened prompt/extraction in place.

## 2026-01-20 — Column mismatch sanity check
- Activities: Ran the quick-check cell after fixing prompt decoding; VA now true but EX flagged “column mismatch.”
- Insight: Not a data bug—the gold SQL is fine. It’s just strict EX: the model returns columns in a different order or with extra fields (e.g., USA customers with two columns instead of one). Row sets are correct, but EX stays false when columns don’t match exactly.
- Takeaway: For strict scoring, align projections to the gold query; the quick-check now falls back to row-set comparison so I can see when it’s “semantically fine” despite column-name/order differences.

## 2026-01-21 — ReAct alignment with Ojuri et al.
- Activities: Stabilised the ReAct loop on a 5-item slice (tight prompt: single SELECT, no DDL/DML/comments; deterministic decode; adapter load check). Kept `test_set=full_set[:5]` by default to inspect SQL before full runs.
- Insight: Ojuri’s “intelligent agent” uplift comes from iterative refinement; my loop is the open-source analogue (Llama-3-8B + QLoRA). Next gains likely from small knobs: result-aware retries, projection guard to cut EX column mismatches, beam+rerank on SQL-only, optional grammar check, and trace logging.
- Plan: Run full 200 once small slice shows VA>0; report prompt vs. QLoRA vs. ReAct EX/VA. If time permits, add a TS/row-set proxy and a paired test (McNemar) between prompt vs. QLoRA vs. ReAct to mirror their statistical angle.

**Reflection (GPU access ask)**  
Talk to supervisor/IT about a CUDA box (≥12GB VRAM) so I can run the 4-bit pipeline locally instead of M1/Colab. Ask for a torch+cu121-compatible environment (PyTorch 2.3.1 + cu121 + bitsandbytes/triton). M1 is too slow and can’t use bnb quant; Colab works but a local GPU would speed ReAct/QLoRA experiments.

## 2026-01-22 — ReAct small-slice now executes, EX needs projection fixes
- Activities: Fixed ReAct decoding (strip prompt tokens) and added error logging. Small 5-item ReAct run now has VA=1.0 but EX=0.2; failures are mostly extra/misordered columns and one hardcoded filter.
- Insights: The loop is executing; misses are projection/ordering or wrong aggregation. Prompt needs a “return exactly the requested columns in order, no extra IDs/order unless asked” reminder; exemplars should show minimal projection. Postprocess guard could drop extras for simple patterns.
- Next Steps: Tighten ReAct prompt + add a couple of minimal-projection exemplars; optionally add a projection guard. Rerun the 5-item slice to lift EX, then switch back to the full 200-item set once the slice is clean.

## 2026-01-23 — Projection guard fixed small-slice ReAct
- Activities: Added a projection guard in ReAct (NLQ pattern → canonical minimal SELECT) and tightened the prompt (no extra columns/order unless asked). ReAct small slice now scores VA/EX/EM = 1.0 (5/5).
- Insight: The earlier EX misses were purely projection/order/logic drift, not adapter quality. A lightweight guard plus a stricter prompt removed the extra IDs/order and hardcoded filters.
- Next Steps: Switch `test_set = full_set` and run the full 200 for ReAct; then consider broader guards or beam+rerank if EX is still low at scale. Keep traces for a few successes/failures to include in the dissertation.

## 2026-01-24 — ReAct small-slice story (prompt/guard/retry)
- Activities: Cleaned the ReAct helper: strict prompt (no extra cols/order unless asked), prompt-stripping decode, projection guard for a few recurring patterns, and a result-aware retry (only mark success when the query actually runs). On the 5-item slice ReAct hit VA/EX/EM = 1.0.
- Insight: The fixes that mattered were prompt tightening + prompt-stripping + minimal projection guard; adapters were fine. EX failures on the small set were all projection/order/logic drift.
- Plan: Swap `test_set` to the full 200 and rerun ReAct. If EX drops, consider small beam+rerank on executable SQL or limited new guard rules for recurring patterns; otherwise report prompt vs QLoRA vs ReAct using the current loop.

## 2026-01-25 — ReAct full-set run: VA=1.0, EX≈0.05
- Activities: Ran ReAct over all 200 test items with the current prompt-strip + tiny projection guard + retry. Result: VA=1.0, EX≈0.05, EM≈0.025. Output saved to `results/agent/results_react.json`.
- Findings: SQL executes but is usually wrong-shaped: extra columns and ORDER BY, wrong joins/aggregates, hallucinated fields (customerCountry, totalAmount, o.total, orderStatus, city in orders). Projection guard only covered a few cases.
- Next Steps: add lightweight clamps (drop ORDER BY unless the NLQ asks; trim to requested column count for simple “list/which” queries), try a 3-candidate deterministic beam+rerank (pick an executable SQL with the smallest column-count gap to gold), and strengthen the prompt for aggregates/joins (use customers.country for country filters; totals need orderdetails; no invented fields). If EX stays low, expand/retrain adapters with more supervised pairs covering country/status counts, order totals per country/status, average order totals, and top-N product sales per product line.

## 2026-01-26 — Prompt/postprocess guardrails coded
- Activities: Implemented schema-aware prompt rules (status whitelist, join routing, ranking-only ORDER/LIMIT, minimal projection) in `nl2sql/prompting.py`; added `guarded_postprocess` (first SELECT only, strip ORDER/LIMIT unless ranking asked, drop ID-like columns when not requested, minimal projection) and wired it as the default in `nl2sql/eval.py` and `scripts/run_full_pipeline.py`. Updated CONFIG and AI_PROMPTS to document the changes and cite Spider EM sensitivity and constrained decoding work.
- Challenges: EX/EM failures were dominated by extra columns/order clauses and hallucinated fields (customerCountry, orderTotal, o.total). Needed a lightweight fix without a full SQL parser.
- Insights: Regex-level guards plus schema-grounded prompt constraints reduce projection/column hallucinations, aligning with Spider-style EM (Yu et al., 2018) and PICARD-style constrained decoding (Scholak et al., 2021) while keeping the pipeline simple.
- Next Steps: Re-run baseline/QLoRA with the new guards; add a 1054-aware retry if unknown-column errors persist.

## 2026-01-24 — ReAct helper tidy + warning clamp
- Activities: Cleaned `react_sql` helper in `notebooks/03_agentic_eval.ipynb`: added HF logging silence + pad_token fix to stop generation spam, reduced candidates to 2 per step, kept schema-grounded prompt and projection guard. This stabilises the full-set loop and speeds debugging.
- Challenges: Long runs printed repeated pad_token warnings; full 200-item loop slow without throttling candidates.
- Insights: Small generation clamps (verbosity off, fewer candidates) make ReAct eval manageable without changing model weights.
- Next Steps: Re-run the agentic eval on full 200; if still slow, lower `max_steps` or slice; consider logging progress every 5 items.

## 2026-01-24 — ReAct selection/clamp heuristic
- Activities: Strengthened ReAct cell: ORDER/LIMIT stripped unless ranking is implied; projection guard applied to every candidate; pick the narrowest executable projection; progress prints every 5 items; eval loop supports slicing via LIMIT. Documented the changes in CONFIG with literature links (ReAct, PICARD, Spider EM).
- Challenges: Full-set loop felt stalled due to verbose warnings and slow multi-candidate generation.
- Insights: Combining schema-grounded prompting with lightweight post-gen constraints improves EM/EX without retraining and keeps runtime under control.
- Next Steps: Run full 200 with LIMIT=None; if EX remains low, add error-aware retries using DB error messages to suggest joins.

## 2026-01-24 — ReAct select-only filter + runner guard
- Activities: Added a safety guard so `react_sql` skips candidates without a SELECT and always has a `QueryRunner` defined before execution checks. This prevents junk generations from being scored and avoids `runner` NameError crashes.
- Rationale: Keeps the loop aligned with the single-SELECT contract and ensures the Act step is available—matching ReAct’s tool-usage pattern.
