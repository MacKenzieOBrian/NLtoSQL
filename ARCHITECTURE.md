# System Architecture & Design Justification  
_Dissertation Project: NL-to-SQL Translation over the ClassicModels Database_  
---

## 1. Overview

This project implements a complete NL-to-SQL experimentation pipeline over the **ClassicModels** MySQL dataset.  
The system is structured around four components:

1. **Secure database access layer** (Cloud SQL Connector + SQLAlchemy).  
2. **QueryRunner execution engine** for controlled SQL evaluation.  
3. **Schema introspection & representation pipeline** for LLM prompting.  
4. **Model inference and evaluation pipeline**, including few-shot prompting and later QLoRA fine-tuning.

The design explicitly follows the methodological recommendations of **Ojuri et al. (2025)**, specifically:

- Evaluation via **VA** (Validity), **EX** (Exact Match), and later **TS** (True Semantic). [10], [18], [19]  
- Systematic comparison of **few-shot prompting vs parameter-efficient fine-tuning**.  
- Emphasis on **safe SQL execution**, **reproducibility**, and **transparent experimental logging**.


---

## 1.1 Execution Environment & Dependency Discipline

- **Environment**: Google Colab GPU runtime (T4/A100), Python 3.12, repo synced from https://github.com/MacKenzieOBrian/NLtoSQL.  
- **Dependencies**: Pinned via `requirements.txt` to stabilize binaries: `torch==2.2.2`, `transformers==4.37.2`, `bitsandbytes==0.42.0`, `triton==2.2.0`, `pandas==2.2.1`, and `numpy==1.26.4` (fixes NumPy C-extension incompatibility).  
- **Process**: Install, then **restart runtime** to clear conflicting preinstalled extensions; verify NumPy/Pandas/Torch alignment.  
- **Missing files**: requirements, JSON test set, and docs are synced from the GitHub project; notebook cells rely on these paths.

## 1.2 Design Principles & Research Questions

- **Reproducibility first**: Pinned requirements, logged notebook cells, and this document form the reproducibility spine so results can be rerun across Colab sessions.  
- **Safety and auditability**: All SQL is read-only, filtered, and logged with metadata; prompts and outputs are stored for traceability.  
- **Baseline before fine-tune**: Start with schema-grounded few-shot prompting to establish VA/EX baselines before adding QLoRA adapters, aligning with Ojuri et al. on measuring true uplift.  
- **Tight schema grounding**: Schema summaries stay in the prompt; prompt length vs. fidelity trade-offs are recorded to justify accuracy/latency decisions.  
- **Research questions**: (a) How well do schema-grounded few-shot prompts perform on ClassicModels? (b) What uplift comes from QLoRA fine-tuning vs prompting alone? (c) How does safety-checked execution affect VA/EX outcomes?

## 1.3 Scope, Assumptions, Risks

- **Scope**: Single-schema ClassicModels MySQL DB; focus on SELECT-only NL-to-SQL.  
- **Assumptions**: Live DB connectivity is available; HF access to the gated Llama-3-8B-Instruct is granted.  
- **Risks & mitigations**: HF gate → request access and cache token in env (fallback to an open model if blocked); Colab binary drift → pin deps and restart runtime; DB safety → QueryRunner blocks DDL/DML substrings and logs violations.

## 1.4 Operational Hygiene (Colab)

- Always start with a **fresh clone** in `/content` (`rm -rf /content/NLtoSQL && git clone ...`) to avoid stale notebooks/deps.  
- Record the commit hash (`git rev-parse --short HEAD`) before installs/runs for reproducibility.  
- Install from `requirements.txt`, restart the runtime, then rerun the notebook from the top to ensure a clean C-extension state.

## 1.5 Inference-Time Few-Shot Prompting (No Training) [6], [3]

- Few-shot prompting is applied **only at inference time**; the LLM (Meta Llama-3-8B-Instruct) is loaded once and reused.  
- Prompt structure: system instruction + textual schema + *k* NLQ→SQL exemplar pairs + one held-out NLQ.  
- No gradient updates, adapters, or fine-tuning are used; any uplift vs zero-shot is attributable solely to prompt conditioning and post-processing.

## 1.6 Schema Grounding Strategy [1], [8], [9]

- Schema is dynamically extracted from `INFORMATION_SCHEMA` and textualized as `table(col1, col2, ...)`.  
- Column order is heuristically prioritised: primary keys first, then identifier/name-like fields (name, id, line, code, number) to reduce column-selection ambiguity.  
- This is a lightweight inductive bias at the prompt level; model parameters remain unchanged.

## 1.7 Post-Processing and Safety [13], [2], [10]

- SQL extraction: regex/pattern matching to keep only the first `SELECT ... ;` before execution.  
- Read-only enforcement: block destructive tokens (DROP/DELETE/ALTER/CREATE/UPDATE/INSERT).  
- Minimal projection heuristic: for “List all …” intents, constrain projected columns to avoid over-selection.  
- These steps are standard NL→SQL safeguards to ensure executability and better alignment with reference SQL without altering the model.

---

## 2. Secure Database Access Layer

### 2.1 Motivation

A live database is required to compute the **VA metric**, which measures whether an LLM-generated SQL query successfully executes. A secure connection must satisfy:

- Role-appropriate access (read-only queries only).  
- Stability under colab/prod environments.  
- Protection against credentials exposure.

### 2.2 Chosen Approach: Cloud SQL Connector + SQLAlchemy

The system uses:

- **Google Cloud SQL Connector (Python)**  
- **SQLAlchemy with a custom `creator` function**

This pattern was chosen because:

- It is the **recommended secure method** for accessing Cloud SQL instances without exposing IPs or opening public access.
- SQLAlchemy offers a consistent interface for *safe*, *parameterized* execution and schema metadata retrieval.
- Colab compatibility is guaranteed through ephemeral OAuth or ADC.

### 2.3 Alternative Considered

| Option | Rejected Because |
|--------|------------------|
| Direct TCP connections | Requires IP allowlisting; less secure. |
| mysqlclient connector | Difficult to configure across Colab + local OS environments. |
| REST/Proxy API layers | Increased latency; breaks standard SQLAlchemy workflows. |

The final choice ensures maximum portability and reproducibility for dissertation experiments.

---

## 3. QueryRunner Execution Engine

### 3.1 Purpose

The **QueryRunner** module is central to this system. It provides:

- Controlled SQL execution  
- Safety filtering  
- Metadata logging  
- Structured evaluation results  

This directly supports NL-to-SQL evaluation, where generated SQL queries must be validated and compared.

### 3.2 Academic Justification

Ojuri et al. (2024) emphasise that NL-to-SQL evaluation requires:

- Verification that a query executes (**VA**)  
- Comparison between generated and gold SQL (**EX**)  
- Eventually assessing semantic equivalence (**TS**)

To reliably compute VA at scale, the system must:

- Prevent destructive queries  
- Capture errors  
- Record execution traces  
- Support large-scale evaluation loops  

This mirrors patterns used in:
- The **Spider**

### 3.3 Safety Design

The QueryRunner blocks destructive operations via substring scanning:
["drop ", "delete ", "truncate ", "alter ", "create ", "update ", "insert "]

This is intentionally simple but effective for supervised NL-to-SQL generation, where:
- The model should *never* produce DDL/DML  
- We must guarantee **read-only execution** for cloud DB security  
- Safety violations must be logged for analysis
- A lightweight guard avoids the latency and complexity of full SQL parsing while still preventing the common destructive cases seen in generated SQL.

### 3.4 Metadata Logged

QueryRunner records:

- SQL query text  
- Timestamp  
- Execution success/failure  
- Error messages  
- Column names  
- Result preview (optional)  

This ensures **transparent, auditable evaluation**, consistent with reproducibility standards in experimental NLP research.

### 3.5 Smoke Tests Completed

- Connectivity and schema enumeration verified against ClassicModels.  
- Sample `SELECT` queries and execution limits validated.  
- The 200-item test set (`data/classicmodels_test_200.json`) runs 200/200 successfully via `validate_test_set`, confirming DB reliability for VA/EX evaluation.

## 4. Schema Exploration & Representation

### 4.1 Rationale

Modern LLM-based NL-to-SQL models require **schema grounding** to perform well.  
Schema representation is referenced throughout LLM prompting literature, including:

- GRAPPA (Li et al., 2020)  
- PICARD (Scholak et al., 2021)  
- Instruction-tuned SQL models (e.g., SQL-LLaMA)

This project, however, uses the simpler but effective approach used in Ojuri et al. (2024):

> Provide a flattened schema description directly inside the LLM prompt.

### 4.2 Implementation

The schema helper builds compact summaries like:
customers(customerNumber int, customerName varchar, city varchar, ...)
orders(orderNumber int, orderDate date, status varchar, ...)

This balances:
- Sufficient schema grounding  
- Minimal prompt length inflation  
- Easy readability for human inspection  

The same schema representation is used for:
- Few-shot prompting  
- Baseline evaluation  
- Fine-tuning experiments  

## 5. LLM Model Loading Architecture

### 5.1 Model Choice: Llama-3-8B-Instruct

This project uses **Llama-3-8B-Instruct** because:

- It is a state-of-the-art open LLM with strong reasoning capabilities.  
- It supports **chat-style prompting**, required for few-shot demonstrations.  
- Its 8B size makes it viable for **4-bit quantized inference** on a T4/A100 GPU in Colab.  
- It aligns with the experimental setups seen in NL2SQL research using LLaMA-based backbones.

### 5.2 Technical Justification for 4-bit Quantization [12], [11], [4], [5]

Using **BitsAndBytes 4-bit NF4 quantization** enables:

- ~4× memory reduction  
- Feasible inference in constrained GPU environments  
 - Maintaining competitive accuracy while fitting academic GPU memory limits

This also matches planned **QLoRA fine-tuning**, which requires the model to be loaded in 4-bit.

### 5.3 HuggingFace Access Controls (Gated Model)

- Meta-Llama-3-8B-Instruct is gated: login + approved access on the model page are both required.  
- Use `notebook_login()` (or `HF_TOKEN`) to persist tokens in Colab.  
- Verified auth via `whoami()`; remaining 403s are authorization-related, not token errors.

### 5.4 4-bit Load & Sanity Check

- Load with `BitsAndBytesConfig` (NF4) and `device_map="auto"` to fit on the GPU.  
- Pad tokenizer if needed; set deterministic generation parameters for evaluation.  
- Smoke test prompt: `"Reply with only the word OK."` confirms model load + generation.

### 5.5 Deterministic Baseline Generation [10], [20], [18]

- For VA/EX baselines, use **deterministic decoding**: `do_sample=False`, unset `temperature/top_p`, modest `max_new_tokens` for SQL, and `pad_token_id=eos_token_id`.  
- Rationale: removes sampling noise so differences in VA/EX reflect prompt/model changes, not randomness; keeps outputs comparable across runs.  
- Sampling (temperature/top_p) is reserved for exploratory prompts, not baseline reporting.
- Terminology: “few-shot learning” here means **inference-time prompt conditioning only**; the underlying model weights are not trained or adapted.

## 6. End-to-End Smoke Tests

- DB: connectivity, schema enumeration, and sample `SELECT` queries pass; `validate_test_set` reports 200/200 success on `data/classicmodels_test_200.json`.  
- Runtime: dependency reinstall + runtime restart clears Colab C-extension conflicts (notably NumPy).  
- Model: 4-bit Llama-3 load with a minimal prompt (“Reply with only the word OK.”) confirms GPU mapping and tokenizer padding before running NL→SQL prompts.

## 7. Few-Shot Prompting Architecture

- Objective: provide a transparent, non-fine-tuned baseline before QLoRA.  
- Prompt shape: schema summary + table blurbs + 2–4 NLQ→SQL exemplars + the new NLQ.  
- Justification: Mirrors Ojuri et al.’s guidance to measure uplift; schema grounding reduces column-name hallucinations; exemplar count balances recall vs. prompt length.  
- Notes: Prompt templates and exemplars are versioned to keep evaluations comparable across runs.
- Baseline rigor: use deterministic decoding (`do_sample=False`, no `temperature/top_p`, bounded `max_new_tokens`, `pad_token_id=eos_token_id`) so VA/EX differences reflect prompt/model changes, not sampling noise; log prompt version + commit hash per run for dissertation traceability.

## 8. Evaluation Architecture (VA / EX, TS planned)

- **VA (Validity)**: Checks if generated SQL executes; driven by QueryRunner success/error metadata.  
- **EX (Exact Match)**: Normalized SQL string comparison against gold SQL for the 200-sample test set.  
- **TS (True Semantic)**: Planned; result-equivalence on distilled DB variants to catch semantically wrong-but-executable SQL.  
- **Experimental control**: Few-shot prompts and QLoRA adapters are pluggable; the evaluation harness stays fixed to isolate model/prompt effects.  
- **Traceability**: Thought/Action/Observation traces, prompts, SQL strings, and execution metadata are logged for later dissertation analysis.

## 9. Summary
This architecture is intentionally designed for:

- Academic reproducibility
- Secure live SQL execution
- LLM baselines + fine-tuning comparisons
- Alignment with NL-to-SQL evaluation methodology (Ojuri et al.)

## 10. Large Language Model Integration (current)
- Backbone: `meta-llama/Meta-Llama-3-8B-Instruct` (gated).
- Quantization: 4-bit (NF4) via bitsandbytes; enables Colab GPU feasibility and QLoRA compatibility.
- Auth: gated access via Hugging Face token (`notebook_login()` or env token; pass `token=True` on load).
- Inference: chat template (`apply_chat_template`), deterministic decoding (`do_sample=False`, no sampling parameters) for reproducible evaluation, `device_map="auto"` (GPU-backed).
- Why 4-bit: reduces memory footprint with minimal capability loss, aligning with parameter-efficient training practice and enabling 8B models on modest GPUs (T4).
- Notes: deterministic decoding and chat template use are required for consistent VA/EX evaluation and align with Ojuri et al.’s methodology. we want changes in metrics to reflect prompt/model choices, not randomness. Setting do_sample=False and removing temperature/top_p makes outputs repeatable run-to-run.
