# Logbook

> Format per day: Activities, Challenges, Insights, Next Steps. Backfilled earlier entries from provided oldlogbook.txt.

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
- Insights: Cloud SQL connector avoids IP allowlisting/SSL hassle; scope requires custom fine-tuning/agent/eval; documentation now reduces later friction.
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
- Next Steps: Load generated dataset for SFT; load target open-source model and finalize QLoRA config; expand training data.

## 2025-12-02
- Activities: Reviewed current notebook scaffold (GCP auth, connector, QueryRunner, schema helpers, Llama-3-8B load). Drafted project structure plan in .md files (ARCHITECTURE.md, CONFIG.md, DATA.md, NOTES.md, scripts for data prep/train/eval/agent). Completed generation of 200 NL-SQL pairs using GPT-5 and ClassicModels context.
- Insights: Should demonstrate few-shot baseline (as in Ojuri et al.) before QLoRA fine-tuning. Keep QueryRunner as ReAct tool; log traces for interpretability. Add NOTES.md for design justifications.
- Next Steps: Implement few-shot prompt baseline against ClassicModels. Begin QLoRA prep with pinned deps and resource logging.

## 2025-12-04
- Activities: Set VS Code terminal env vars/ADC so the notebook connects locally; ran `validate_test_set` on all 200 ClassicModels queries—Success: 200, Failures: 0.
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
- Activities: Added `triton==2.2.0` to requirements and pushed; refreshed Colab workflow to always reclone clean (`rm -rf /content/NLtoSQL`, `git clone`) and install pins, then restart runtime; confirmed 4-bit Llama-3-8B-Instruct loads on `cuda:0` with deterministic defaults; captured the fresh-clone cell for notebooks.
- Challenges: saw sampling warnings when mixing `do_sample=False` with `temperature/top_p`.
- Insights: For VA/EX baselines, use deterministic generation (no sampling params, small `max_new_tokens`, set `pad_token_id=eos_token_id`); sampling is only for exploratory runs.
- Next Steps: Build/run the schema-grounded few-shot baseline, log VA/EX with fixed prompt template and generation settings; record hardware/commit/prompt version for reproducibility.

## 2025-12-14
- Observation: Zero-shot outputs executed (VA) but often misaligned with gold SQL (EX gaps).
- Intervention: Introduced inference-time few-shot exemplars and stricter prompts; added SQL post-processing to extract the first `SELECT ... ;` and enforce minimal projection for list-style queries.
- Outcome: Higher syntactic correctness and better column selection (identifier-first ordering reduced confusion between productLine vs textDescription); representative cases reached VA=True and EX=True without changing model weights.
- Rationale: All improvements come from prompt conditioning, schema ordering, and post-processing; the model remains frozen to isolate methodological effects.
