# Logbook

Purpose: maintain a chronological record of decisions, experiments, results, and next steps for dissertation traceability.

Suggested format per entry:
- Activities (what was done)
- Challenges (what blocked/failed)
- Insights (what was learned)
- Next Steps (what to do next, testable)

Older entries were backfilled from earlier notes during repo clean-up.

## 2025-09-29
- Activities: Met supervisor to align scope; drafted project outline; read core references on agentic NL-to-SQL; captured framework notes.
- Challenges: Initial scope felt broad; over-wrote contextual background leaving less space for state-of-the-art; missed noting relevance of some sources.
- Insights: Clarified foreign methods (PEFT, QLoRA, ReAct) and how they fit agentic training; understood main paper’s findings and improvement areas.
- Next Steps: Gather diverse sources beyond seed paper; justify frameworks in outline; share draft with supervisor.

**Reflection (2025-09-29)**  
This was the formal starting point: I used the supervisor meeting to stop the project drifting into a “broad LLM essay” and instead anchor it in a concrete, testable system. The reading on agentic NL→SQL, PEFT, QLoRA and ReAct wasn’t just background—at this stage it was about identifying which methods were feasible under dissertation constraints (time, compute, reproducibility). The key positive outcome was that I left this week with a clear understanding of how methods relate (prompting → agent loops → fine-tuning) and a plan to broaden beyond one seed paper so the literature review would be defensible and not derivative.

## 2025-10-06
- Activities: Expanded outline (~2000 words); leveraged cited studies for state-of-the-art; started ethics application; set up OneDrive/Teams for sharing.
- Challenges: Heavy reliance on seed sources; struggled with structure for state-of-the-art; still leaning on root references.
- Insights: Supervisor suggested Zotero; explored alternative agentic LLM methods; improved grasp of PEFT/QLoRA/ASOT/ReAct/knowledge graphs.
- Next Steps: Broaden sources to differentiate from seed paper; log all references in Zotero; complete ethics submission.

**Reflection (2025-10-06)**  
The big achievement this week was structuring the dissertation foundation: outline expansion, ethics prep, and establishing tooling for collaboration (OneDrive/Teams). This mattered because I was building a “research workflow” rather than just a code repo. The main challenge—over-reliance on a seed paper—was important to recognise early. The supervisor’s suggestion of Zotero was also a turning point: it made citation discipline realistic, which is crucial later when justifying design decisions (not just describing them). In hindsight, this week is where the project began transitioning from reading to accountable deliverables.

## 2025-10-13
- Activities: Submitted final ethics application; added ~500 words linking literature to implementation; adjusted tone per lecture review.
- Challenges: Worried state-of-the-art is descriptive vs analytical; need stronger critique of approach trade-offs.
- Insights: Zotero citation export saves time; deeper reading bridges contextual and technical discussion on agentic methods.
- Next Steps: Revisit state-of-the-art to add critical discussion; map literature to implementation plan; ensure IEEE style via Zotero; add cover page.

**Reflection (2025-10-13)**  
Submitting ethics forced me to formalise what data I would use and how I would evaluate, and it reduced uncertainty about what I’m “allowed” to do. The challenge—being too descriptive in state-of-the-art—was a signal that I needed to start writing like an evaluator: comparing trade-offs, not listing methods.

## 2025-10-20
- Activities: Refined outline per supervisor comments; added open-source vs proprietary comparison; created SMART objectives and MoSCoW; drafted 13-week timeline; documented evaluation metrics (VA/EX/TS); noted Colab as fallback for hardware limits.
- Challenges: Incorporating feedback while keeping evaluation rationale clear; security and compute feasibility concerns.
- Insights: “Reproducibility gap” is key contribution angle; QLoRA is enabling tech for feasibility; hybrid approach (optimized prompt + ReAct + constraints) required.
- Next Steps: Submit outline; begin practical setup; configure Colab environment; read Ojuri et al. and ReAct papers in full.

**Reflection (2025-10-20)**  
This week is where the dissertation became an engineering-research plan rather than a set of ideas. Producing SMART objectives, MoSCoW priorities, a timeline, and clear evaluation metrics (VA/EX/TS) did two things: (1) it made the scope defendable, and (2) it framed “reproducibility gap” as a contribution, which is a strong dissertation angle. The challenge was balancing evaluation motivation with feasibility and security constraints. The insight that a hybrid approach would be required (prompt optimisation + constraints + potentially ReAct + fine-tuning) helped prevent unrealistic expectations that one technique would solve everything. In retrospect, this week defined the project’s shape: controlled baselines first, complex additions second.

## 2025-10-27
- Activities: Reviewed outline/timeline; identified colab-llm repo as reference; planned weekly checklist; prepped minimal test notebook; pivoted from SQLite to Cloud SQL; established secure connector pattern (creator + SQLAlchemy).
- Challenges: colab-llm needs adaptation for fine-tune/ReAct; SQLite failed due to MySQL-specific syntax; temporary root credentials/hardcoded secret.
- Insights: Cloud SQL connector avoids IP allowlisting/SSL hassle; scope requires custom fine-tuning/agent/eval; documentation now reduces later friction.
- Next Steps: Verify Cloud SQL data integrity; load Llama-3-8B in 4-bit; baseline prompt tests; record hardware specs; finish reading Ojuri/ReAct.

**Reflection (2025-10-27)**  
This was a practical pivot week. The secure connector pattern (Cloud SQL connector + SQLAlchemy creator) was a foundational design decision because it allowed reproducible DB access across Colab/local. The challenge was that early credentials handling was messy (hardcoded secrets), which is a known anti-pattern, but the benefit was speed: it let me validate the architecture quickly before tightening security. The key insight was recognising that the “colab-llm” reference repo couldn’t be used as-is—this project needed custom evaluation, safety constraints, and eventually fine-tuning/agent integration.

## 2025-11-03
- Activities: Built Colab scaffold with connector, safe_connection, schema introspection, QueryRunner; ran smoke tests on customers table.
- Challenges: Credentials still hardcoded; progress slowed by other coursework.
- Insights: Connector simplifies secure access; QueryRunner provides safe Act+metadata; need to prioritize model evals and literature depth.
- Next Steps: Load/evaluate models; read ReAct/Ojuri plus new sources; assemble 200-sample dataset.

**Reflection (2025-11-03)**  
This is when the system became testable end-to-end. Building the connector, schema introspection, and QueryRunner meant I had the minimum viable experimental platform: I could execute SQL safely, log outcomes, and inspect schema programmatically. That’s a huge step because VA cannot be measured without reliable execution. The insight that QueryRunner provides the “Act” analogue for a future ReAct loop is key—it shows early architectural foresight rather than building throwaway code. Overall, this week turned the project into something I could actually run experiments on.

## 2025-11-10
- Activities: Regrouped after coursework; reviewed timeline to enter data prep phase.
- Challenges: Moving from infra to conceptual tasks (data, ReAct) requires stronger grounding in papers.
- Insights: Fine-tuning likely superior to few-shot for domain tasks; theory first to design pipeline correctly.
- Next Steps: Deepen fine-tuning/ReAct understanding; start data generation planning.

**Reflection (2025-11-10)**  
This week was a reset into the research plan after other coursework. The main challenge was moving from infrastructure into conceptual tasks like dataset design and agent planning, which require stronger grounding in papers to avoid building the wrong thing. The insight—that fine-tuning is likely superior to few-shot for domain-specific NL→SQL—was important, but I treated it as a hypothesis rather than a conclusion.

## 2025-11-17
- Activities: Completed plumbing for QLoRA; curated ~100 NLQ-SQL pairs; structured dataset with schema+NLQ+SQL text field; aligned QueryRunner with ReAct Act.
- Challenges: Bitsandbytes/model loading unstable on Colab GPUs; Colab struggled with large model; uncertainty if QLoRA setup is at fault.
- Insights: QLoRA 4-bit is key for feasibility; loading/quantization is finicky; QueryRunner maps directly to Act in ReAct loop.
- Next Steps: Load generated dataset for SFT; load target open-source model and expand training data.

**Reflection (2025-11-17)**  
Here I started preparing for QLoRA in earnest: dataset structuring and 4-bit feasibility work. The major challenge was the reality of Colab GPU fragility: bitsandbytes and quantisation can be unstable depending on runtime versions and binaries. The insight that QLoRA feasibility depends on careful quantisation setup is central because it justifies why dependency pinning becomes a research requirement, not just an engineering preference. Also, aligning QueryRunner with ReAct’s “Act” concept shows that the work wasn’t siloed—execution and evaluation were being designed with future agent experiments in mind.

## 2025-12-02
- Activities: Reviewed current notebook scaffold (GCP auth, connector, QueryRunner, schema helpers, Llama-3-8B load). Drafted project structure plan in .md files (ARCHITECTURE.md, CONFIG.md, DATA.md, NOTES.md, scripts for data prep/train/eval/agent). Completed generation of 200 NL-SQL pairs using GPT-5 and ClassicModels context.
- Insights: Should demonstrate few-shot baseline (as in Ojuri et al.) before QLoRA fine-tuning. Keep QueryRunner as ReAct tool; log traces for interpretability. Add NOTES.md for design justifications.
- Next Steps: Implement few-shot prompt baseline against ClassicModels.

**Reflection (2025-12-02)**  
This week consolidated the project into a documented structure and produced the primary evaluation dataset (200 NLQ–SQL pairs). The key methodological insight was recognising that a few-shot baseline should be demonstrated before fine-tuning. This matters academically because without a baseline, any fine-tuning uplift can’t be contextualised. I also began formalising the repository documentation (ARCHITECTURE/CONFIG/DATA/NOTES).

## 2025-12-04
- Activities: Set VS Code terminal env vars/ADC so the notebook connects locally; ran `validate_test_set` on all 200 ClassicModels queries—Success: 200, Failures: 0.
- Challenges: Needed kernel restart and imports before calling the helper; VS Code env setup was finicky until settings.json had the DB vars.
- Insights: Static test set executes cleanly end-to-end!
- Next Steps: Pin deps/requirements, run few-shot baseline with metrics/screenshots, then start QLoRA prep and training.

**Reflection (2025-12-04)**  
This was a crucial validation milestone: running `validate_test_set` across all 200 queries and getting 200/200 success confirmed that the benchmark SQL is executable and the DB schema matches expectations. That result protects the validity of later evaluation: if EX fails later, it’s not because the gold SQL is broken or the database is inconsistent.

## 2025-12-05
- Activities: Pinned deps in requirements.txt, updated notebook/script installs to use it; pushed changes. Verified the updated install cell in the clean notebook. Confirmed validation remains green.
- Challenges: VS Code caching made it look like cells weren’t updating, annoying to figure out.
- Insights: Locking installs via requirements.txt keeps runs reproducible across Colab/local.
- Next Steps: Build/run the few-shot baseline (log VA/EX with prompts/screenshots), then start QLoRA hyperparam locking and SFT prep.

**Reflection (2025-12-05)**  
Pinning dependencies in requirements.txt and verifying installs in a clean environment is a research-quality step, not just a dev step. It addresses Colab/local drift, which can silently change outputs, performance, and even model loading behaviour. The challenge with VS Code caching is an example of why experimental pipelines need version control discipline and clear “clean run” procedures. The insight is that reproducibility is an active job, especially across Colab sessions. This week increased confidence that results can be rerun.

## 2025-12-06
- Activities: Synced Colab with repo, pulled latest pins; fixed connector/cryptography pins and NumPy mismatch; set ADC/quota project; ran installs via requirements.txt. Notebook installs now point at requirements.txt; Config now includes env var examples.
- Challenges: Cloud SQL auth issues in Colab (quota project/permissions).
- Insights: Use git pull in Colab before installs; set env vars explicitly.
- Next Steps: Run few-shot baseline on Colab GPU with current stack; record VA/EX and screenshots; then move to QLoRA SFT prep (hyperparams/logging) if set up is correct. 

**Reflection (2025-12-06)**  
This week was about hardening Colab workflow reliability: pulling repo changes, resolving dependency mismatches (connector/cryptography/NumPy). These problems are common in cloud notebook research and directly threaten reproducibility if ignored. The insight—pull before install, set env vars explicitly—became part of operational hygiene. Methodologically, this week strengthened the “experimental apparatus”: if the apparatus is unstable, performance comparisons are meaningless.

## 2025-12-07
- Activities: Documented gated-model loading (HF token, access approval), 4-bit NF4 setup, and dependency pin rationale in CONFIG/ARCHITECTURE. Verified validation still 200/200.
- Challenges: Hugging Face gated access and Colab binary drift can still throw warnings; requires correct token + access and careful install/restart order.
- Insights: Deterministic decoding + chat template + 4-bit load are required for reproducible eval and future QLoRA on Colab GPUs.
- Next Steps: Run few-shot NL→SQL baseline on Colab GPU with current stack; capture VA/EX and prompts; begin QLoRA SFT prep with pinned toolchain.

**Reflection (2025-12-07)**  
Documenting gated-model loading and 4-bit NF4 setup was necessary because it is both a reproducibility and feasibility issue. Without explicit documentation, another person (or future me) would not be able to replicate the model-loading stage, especially with Hugging Face access controls. The insight that deterministic decoding + chat template + quantised load are needed for consistent evaluation became a key methodological principle: baseline numbers must reflect method choices, not randomness.

## 2025-12-12
- Activities: Added `triton==2.2.0` to requirements and pushed; refreshed Colab workflow to always reclone clean (`rm -rf /content/NLtoSQL`, `git clone`) confirmed 4-bit Llama-3-8B-Instruct loads on `cuda:0` with deterministic defaults
- Challenges: saw sampling warnings when mixing `do_sample=False` with `temperature/top_p`.
- Insights: For VA/EX baselines, use deterministic generation (no sampling params, small `max_new_tokens`, set `pad_token_id=eos_token_id`); sampling is only for exploratory runs.
- Next Steps: Build/run the schema-grounded few-shot baseline, log VA/EX with fixed prompt template and generation settings; record hardware/commit/prompt version for reproducibility.

**Reflection (2025-12-12)**  
This was a “make it reliably runnable” week. Adding `triton==2.2.0`, adopting a fresh reclone workflow, and verifying 4-bit load on GPU all contributed to stability. The main challenge—warnings when mixing deterministic decoding with sampling parameters—was more than cosmetic: it highlighted that baseline evaluation must be cleanly configured. The insight here is methodological: for VA/EX baselines, generation should be deterministic and sampling removed. This ensures that measured changes are attributable to prompt/model variations, not stochastic decoding behaviour.

## 2025-12-14
- Activities: Added inference-time few-shot prompt pipeline (system + schema + k exemplars + NLQ), enforced deterministic decoding, and implemented SQL post-processing (extract first `SELECT ...;`, minimal projection for list-style queries). Kept schema columns ordered (PKs/name-like first) in the prompt.
- Challenges: Zero-shot outputs were executable (VA) but misaligned with gold SQL (EX gaps); model occasionally echoed instructions or over-selected columns.
- Insights: Few-shot exemplars + ordered schema + post-processing improved syntactic correctness and column choice (e.g., productLine over textDescription), achieving VA=True and EX=True on representative cases without changing model weights. Improvements stem from prompt conditioning and heuristics only.
- Next Steps: Run the full 200-sample VA/EX baseline with the fixed prompt/post-processing; log commit/prompt version/hardware. 

**Reflection (2025-12-14)**  
This is the turning point where the project produces a credible few-shot baseline rather than just infrastructure. Implementing the inference-time prompt pipeline (system + schema + k exemplars + NLQ), enforcing deterministic decoding, and adding robust SQL post-processing addressed observed failure modes: echoed instructions, invalid SQL blobs, and over-selection of columns. The ordered schema representation (PK/name-like first) introduced a lightweight inductive bias that improved column choice without touching model weights. The key insight is that improvements (VA=True, EX=True on representative examples) came entirely from prompt conditioning and controlled heuristics, which makes the baseline academically defensible: no fine-tuning, no hidden training, just a transparent inference-time method. The next step (full 200-sample evaluation with fixed prompt/version/hardware logging) follows directly from good experimental practice.

## 2025-12-23
- Activities: Implemented and ran the batch evaluation loop for both zero-shot (`k=0`) and few-shot (`k=3`), first on small subsets (`limit=1`, `limit=20`) and then on the full benchmark (`n=200`). The baseline runner is now `notebooks/02_baseline_prompting_eval.ipynb`, writing outputs under `results/baseline/` (gitignored by default).
- Challenges: Batch inference is slow (LLM generation dominates runtime). EX stayed low relative to VA because string-level matching is strict: aliasing, equivalent joins/subqueries, and harmless extra columns can flip EX to False even when the query is semantically correct.
- Insights: Few-shot prompting improves both executability and strict match: zero-shot reached `VA=0.810`, `EX=0.000`; few-shot reached `VA=0.865`, `EX=0.250` (i.e., +5.5pp VA and +25pp EX) with frozen weights. This supports using VA as an “apparatus check” and treating EX as a conservative baseline, complemented by semantic/result-based evaluation (TS). [10], [18], [19]
- Next Steps: Add a result-equivalence check on a subset (execute both gold and predicted SQL and compare result sets) as a proxy for TS; refine the “minimal projection” heuristic so it only fires for genuinely simple list-intent questions (avoid harming prompts like “List all offices with city and country”); then rerun the 200-item evaluation and report deltas transparently.

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

## 2026-02-XX — Agentic plan
- Activities: Sketched a ReAct loop and scaffolded `03_agentic_eval.ipynb` (Thought → Action → Observation → Refinement with `QueryRunner`); kept prompts short and deterministic.
- Take: Agentic refinement is the next lever to lift EX (and TS, when added), especially at k=0. Plan to run base vs QLoRA adapters through the loop and log traces for a sample of failures.

## 2026-02-18 — Dependency wobble fix
- Activities: Added guard + pinned-setup cells to both `03_agentic_eval.ipynb` and `05_qlora_train_eval.ipynb` to catch the classic `numpy.dtype size changed` / missing `triton.ops` errors. Turned on a pre-setup check that bails if torch/numpy/bnb/triton are already imported. Stopped ignoring `results/` so run artifacts can be committed.
- Challenges: Colab keeps preloading mismatched wheels; reinstalling mid-session was causing binary ABI mismatches.
- Insights: Run the guard + setup first in a fresh GPU runtime, then restart—no more dtype errors. Adapter path confusion is now handled (falls back to base model if adapters aren’t present).
- Next Steps: Upload or regenerate adapters before agentic eval; run full agentic pass; keep the guard pattern in any new notebooks.
