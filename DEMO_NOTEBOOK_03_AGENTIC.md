# Demo Guide — notebooks/03_agentic_eval.ipynb

This guide mirrors the notebook cell-by-cell so you can present a top‑down story and then drop into implementation details. Each cell has:
- **Top-down**: what the cell achieves in the narrative.
- **Technical**: what it does concretely.
- **Show more code**: where the reusable implementation lives.
- **Paper refs**: cite only if asked (use `REFERENCES.md`).

---

**Cell 00 (Markdown) — Agentic Evaluation (ReAct-style)**
Top-down: Introduce the goal: evaluate an execution-guided, ReAct-inspired NL→SQL agent on ClassicModels.
Technical: Sets the stage for the notebook and defines scope.
Show more code: `nl2sql/agent_utils.py#L90`, `nl2sql/eval.py#L291`.
Paper refs: Yao et al., 2023 (ReAct); Ojuri et al., 2025.

**Cell 01 (Markdown) — Docs I leaned on**
Top-down: Justify toolchain choices for quantization + PEFT.
Technical: Points to HF quantization/PEFT docs that match the model load path.
Show more code: `nl2sql/llm.py#L31`.
Paper refs: Ding et al., 2023; Goswami et al., 2024.

**Cell 02 (Markdown) — Setup (run first, then restart)**
Top-down: Reproducible environment setup is required before any results.
Technical: Informs that pinned installs + restart prevent binary drift.
Show more code: `requirements.txt`, `5.CONFIG.md`.
Paper refs: Zhu et al., 2024; Hong et al., 2025 (repro guidance).

**Cell 03 (Markdown) — Setup docs**
Top-down: Notes why CUDA/BnB versions are fixed.
Technical: Aligns local setup with the quantization stack.
Show more code: `nl2sql/llm.py#L31`.
Paper refs: Ding et al., 2023 (PEFT).

**Cell 04 (Code) — Pinned installs (CUDA/BnB/HF)**
Top-down: Freeze the runtime to avoid subtle eval drift.
Technical: Uninstalls preloads, installs pinned numpy/pandas/torch/BnB/transformers.
Show more code: `requirements.txt` and `5.CONFIG.md`.
Paper refs: None.

**Cell 05 (Markdown) — Model load note**
Top-down: Explain that model is 4‑bit NF4, deterministic decoding.
Technical: Matches the load path in code and keeps eval deterministic.
Show more code: `nl2sql/llm.py#L31`.
Paper refs: Ding et al., 2023; Goswami et al., 2024.

**Cell 06 (Code) — Clone repo + install deps**
Top-down: Bootstraps the project in Colab.
Technical: Clones repo, installs `requirements.txt`, prints versions.
Show more code: `requirements.txt`, `scripts/run_full_pipeline.py`.
Paper refs: None.

**Cell 07 (Markdown) — Prompt/eval overview**
Top-down: Summarize the prompt→generate→postprocess→VA/EX loop.
Technical: Signals the harness in `nl2sql` is the source of truth.
Show more code: `nl2sql/eval.py#L291`, `nl2sql/postprocess.py#L125`.
Paper refs: Zhong, Yu, and Klein, 2020; Yu et al., 2018.

**Cell 08 (Markdown) — Notebook architecture note**
Top-down: Notebook orchestrates; logic lives in modules.
Technical: Explains why the notebook is thin and reproducible.
Show more code: `nl2sql/` package.
Paper refs: Zhu et al., 2024 (surveyed reproducibility norms).

**Cell 09 (Markdown) — Reference notes**
Top-down: Map this notebook to the literature narrative.
Technical: Lists the research anchors for later claims.
Show more code: `1.LITERATURE.md`.
Paper refs: See `REFERENCES.md`.

**Cell 10 (Markdown) — Optional: gcloud ADC**
Top-down: Provide an auth path without a JSON key.
Technical: Uses gcloud ADC flow.
Show more code: `nl2sql/db.py#L21`.
Paper refs: None.

**Cell 11 (Markdown) — ADC docs**
Top-down: External auth docs pointer.
Technical: No code.
Show more code: None.
Paper refs: None.

**Cell 12 (Code) — gcloud auth**
Top-down: Optional auth step if you choose ADC.
Technical: Installs auth libs, runs `gcloud auth application-default login`.
Show more code: `nl2sql/db.py#L21`.
Paper refs: None.

**Cell 13 (Markdown) — CUDA/BnB pin rationale**
Top-down: Pins avoid runtime mismatch.
Technical: Documentation context only.
Show more code: `requirements.txt`.
Paper refs: None.

**Cell 14 (Markdown) — Cloud SQL connector reference**
Top-down: Justify DB connector choice.
Technical: Doc pointers only.
Show more code: `nl2sql/db.py#L21`.
Paper refs: None.

**Cell 15 (Markdown) — Auth/DB docs**
Top-down: Explain DB access prerequisites.
Technical: Doc pointers only.
Show more code: `nl2sql/db.py#L21`.
Paper refs: None.

**Cell 16 (Code) — Environment + DB**
Top-down: Establish a live read-only DB connection.
Technical: Reads env vars, builds SQLAlchemy engine via Cloud SQL Connector, sanity query.
Show more code: `nl2sql/db.py#L21`.
Paper refs: None.

**Cell 17 (Code) — TS engine factory**
Top-down: Enable Test‑Suite Accuracy across multiple DB replicas.
Technical: `make_engine(db_name)` returns DB-specific engines.
Show more code: `nl2sql/eval.py#L201`.
Paper refs: Zhong, Yu, and Klein, 2020.

**Cell 18 (Markdown) — Schema helper note**
Top-down: Explain why schema summary is used.
Technical: Points to schema-grounded prompting.
Show more code: `nl2sql/schema.py#L44`.
Paper refs: Zhu et al., 2024; Yu et al., 2018.

**Cell 19 (Markdown) — Schema prompts docs**
Top-down: Literature pointer.
Technical: No code.
Show more code: `nl2sql/prompting.py#L14`.
Paper refs: Zhu et al., 2024.

**Cell 20 (Code) — Load schema + test set (slice)**
Top-down: Prepare schema summary and a small debug slice.
Technical: Builds schema text and loads `classicmodels_test_200.json`, sets `TABLES`.
Show more code: `nl2sql/schema.py#L44`, `data/classicmodels_test_200.json`.
Paper refs: Yu et al., 2018.

**Cell 21 (Markdown) — Model load ref**
Top-down: Document quantized model load.
Technical: Pointer only.
Show more code: `nl2sql/llm.py#L31`.
Paper refs: Ding et al., 2023.

**Cell 22 (Markdown) — Model load docs**
Top-down: External doc pointer.
Technical: No code.
Show more code: `nl2sql/llm.py#L31`.
Paper refs: None.

**Cell 23 (Code) — Load model (base or QLoRA)**
Top-down: Load base model or adapters, keep decoding deterministic.
Technical: Loads tokenizer, 4‑bit NF4 model, then optional PEFT adapters.
Show more code: `nl2sql/llm.py#L31`, `scripts/run_full_pipeline.py`.
Paper refs: Ding et al., 2023; Goswami et al., 2024.

**Cell 24 (Markdown) — Optional adapter sanity check**
Top-down: Verify adapters before full agent loop.
Technical: Introduces the quick check cell.
Show more code: `nl2sql/eval.py#L104`.
Paper refs: Zhong, Yu, and Klein, 2020.

**Cell 25 (Markdown) — Prompt/eval docs**
Top-down: ICL + execution metrics doc pointer.
Technical: No code.
Show more code: `nl2sql/prompting.py#L32`, `nl2sql/eval.py#L104`.
Paper refs: Brown et al., 2020; Zhong, Yu, and Klein, 2020.

**Cell 26 (Code) — Adapter sanity check (VA/EX)**
Top-down: Smoke test for the model/adapters on a few samples.
Technical: Builds prompts, generates SQL, postprocesses, runs VA/EX.
Show more code: `nl2sql/prompting.py#L32`, `nl2sql/postprocess.py#L125`, `nl2sql/query_runner.py#L60`, `nl2sql/eval.py#L104`.
Paper refs: Brown et al., 2020; Zhong, Yu, and Klein, 2020.

**Cell 27 (Markdown) — ReAct pattern note**
Top-down: Introduce the ReAct framing.
Technical: States the DB is the “Act” tool.
Show more code: `nl2sql/query_runner.py#L60`.
Paper refs: Yao et al., 2023.

**Cell 28 (Markdown) — ReAct docs**
Top-down: Literature pointer for ReAct.
Technical: No code.
Show more code: `nl2sql/agent_utils.py#L263`.
Paper refs: Yao et al., 2023.

**Cell 29 (Code) — Helper imports**
Top-down: Pull in the reusable agent utilities.
Technical: Imports cleaner, prompt variants, error taxonomy, semantic scoring.
Show more code: `nl2sql/agent_utils.py#L90`, `nl2sql/agent_utils.py#L263`, `nl2sql/agent_utils.py#L414`.
Paper refs: Zhai et al., 2025; Zhong, Yu, and Klein, 2020.

**Cell 30 (Markdown) — Agent status**
Top-down: Set expectations about current agent maturity.
Technical: Narrative cell.
Show more code: `notebooks/03_agentic_eval.ipynb`.
Paper refs: Ojuri et al., 2025.

**Cell 31 (Markdown) — Eval harness ref**
Top-down: Anchor VA/EX/EM to shared harness.
Technical: Points to `nl2sql/eval.py`.
Show more code: `nl2sql/eval.py#L291`.
Paper refs: Zhong, Yu, and Klein, 2020.

**Cell 32 (Markdown) — Prompt/eval docs**
Top-down: Literature pointer.
Technical: No code.
Show more code: `nl2sql/prompting.py#L32`.
Paper refs: Brown et al., 2020.

**Cell 33 (Markdown) — ReAct pipeline header**
Top-down: Section divider before full agent.
Technical: No code.
Show more code: `nl2sql/agent_utils.py#L263`.
Paper refs: Yao et al., 2023.

**Cell 34 (Markdown) — Reference map**
Top-down: Links concept ↔ implementation.
Technical: Narration only.
Show more code: `2.METHODOLOGY.md`.
Paper refs: See `REFERENCES.md`.

**Cell 35 (Code) — Reload schema + full test set + QueryRunner**
Top-down: Move from debug slice to full dataset for real eval.
Technical: Reloads schema summary, full test set, creates QueryRunner.
Show more code: `nl2sql/schema.py#L44`, `nl2sql/query_runner.py#L60`.
Paper refs: Yu et al., 2018.

**Cell 36 (Code) — Agent utilities import (again)**
Top-down: Ensure helpers are in scope before the big helper layer.
Technical: Same as Cell 29.
Show more code: `nl2sql/agent_utils.py#L90`.
Paper refs: Zhai et al., 2025.

**Cell 37 (Markdown) — Helper layer header**
Top-down: Introduce the staged control layer.
Technical: Narration only.
Show more code: `nl2sql/agent_utils.py#L263`.
Paper refs: Scholak et al., 2021; Zhong, Yu, and Klein, 2020.

**Cell 38 (Code) — Helper layer (CFG, cleaning, clamps, repair)**
Top-down: Implements the core control layer used by the agent.
Technical: Defines CFG, custom cleaner, prompt echo stripping, clamps, schema subset, generation, repair, and candidate postprocess.
Show more code: `nl2sql/agent_utils.py#L90`, `nl2sql/agent_utils.py#L193`, `nl2sql/agent_utils.py#L235`, `nl2sql/postprocess.py#L125`.
Paper refs: Scholak et al., 2021 (PICARD); Zhong, Yu, and Klein, 2020; Zhai et al., 2025.

**Cell 39 (Code) — ReAct loop (react_sql)**
Top-down: Full agent loop: generate → clean → execute → gate → score → repair → fallback.
Technical: Scoring uses semantic_score + column penalty; traces are logged for interpretability.
Show more code: `nl2sql/agent_utils.py#L414`, `nl2sql/query_runner.py#L60`, `nl2sql/eval.py#L104`.
Paper refs: Yao et al., 2023; Zhai et al., 2025.

**Cell 40 (Markdown) — EX troubleshooting checklist**
Top-down: Debugging guide for VA≠EX issues.
Technical: Narrative only.
Show more code: `3.DECISIONS.md`.
Paper refs: Zhong, Yu, and Klein, 2020.

**Cell 41 (Code) — Quick sanity check**
Top-down: Run a few examples to validate the loop before full eval.
Technical: Calls `react_sql`, prints prediction vs gold, shows trace length.
Show more code: `notebooks/03_agentic_eval.ipynb` (this cell).
Paper refs: None.

**Cell 42 (Markdown) — Stage 3 interpretation**
Top-down: Summarize what Stage 3 means in the ablation ladder.
Technical: Narrative only.
Show more code: `LOGBOOK.md`.
Paper refs: Zhong, Yu, and Klein, 2020; Yao et al., 2023.

**Cell 43 (Markdown) — Run order**
Top-down: Provide the recommended execution order.
Technical: Narrative only.
Show more code: `notebooks/03_agentic_eval.ipynb`.
Paper refs: None.

**Cell 44 (Code) — TS harness import**
Top-down: Bring in suite-based semantic evaluation.
Technical: Imports `test_suite_accuracy_for_item` from `nl2sql.eval`.
Show more code: `nl2sql/eval.py#L201`.
Paper refs: Zhong, Yu, and Klein, 2020.

**Cell 45 (Code) — Quick test toggles**
Top-down: Make it safe to debug TS/EX quickly.
Technical: Sets limit, TS_N, MAX_ROWS_TS.
Show more code: `nl2sql/eval.py#L201`.
Paper refs: None.

**Cell 46 (Code) — Full ReAct evaluation + save**
Top-down: Run full eval with VA/EX/EM/TS and save results.
Technical: For each NLQ: run agent, compute TS, compute VA/EX/EM, store trace, then aggregate rates.
Show more code: `nl2sql/eval.py#L291`, `nl2sql/eval.py#L201`, `nl2sql/query_runner.py#L60`.
Paper refs: Zhong, Yu, and Klein, 2020; Ojuri et al., 2025.

---

## Q&A Quick Map
- **“Where is the core agent logic?”** `nl2sql/agent_utils.py#L90` to `nl2sql/agent_utils.py#L414`.
- **“Where are VA/EX/TS computed?”** `nl2sql/query_runner.py#L60`, `nl2sql/eval.py#L104`, `nl2sql/eval.py#L201`.
- **“Where is schema grounding done?”** `nl2sql/schema.py#L44`, `nl2sql/prompting.py#L14`.
- **“Where do we load the model/adapters?”** `notebooks/03_agentic_eval.ipynb` Cell 23, `scripts/run_full_pipeline.py`.

