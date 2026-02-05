# ReAct Loop Diagrams (Tool‑Driven NL→SQL)

This file collects the diagrams used to explain the tool‑driven ReAct loop, the experimental methodology, and the code flow. The diagrams map directly to the current implementation in `notebooks/03_agentic_eval.ipynb` and the tool interface in `nl2sql/agent_tools.py`.

---

**Conceptual ReAct Loop (Tool‑Driven, Validation + Execution Feedback)**

Note: validation/execution/constraint failures force `repair_sql` in the implementation, and trace summaries log action order and compliance.

```mermaid
flowchart TD
  A[User NLQ] --> B[Bootstrap trace]
  B --> B1[Action: get_schema]
  B1 --> LS[Action: link_schema]
  LS --> EC[Action: extract_constraints]
  EC --> B2[Observation: linked schema + constraints]
  B2 --> C[LLM Thought]

  C --> D{Action chosen}
  D -->|get_table_samples| S[get_table_samples]
  S --> S1[Observation: sample rows]
  S1 --> C

  D -->|get_schema| B1

  D -->|generate_sql| G[generate_sql tool]
  G --> H[Guardrails: clean + projection + casing]
  H --> V[validate_sql tool]
  V --> VQ{Valid?}
  VQ -->|no| R[repair_sql tool]
  R --> H

  VQ -->|yes| VC[validate_constraints tool]
  VC --> VCQ{Constraints OK?}
  VCQ -->|no| R
  VCQ -->|yes| X[run_sql tool]
  X --> XQ{Exec OK?}
  XQ -->|no| R
  XQ -->|yes| I{Intent OK?}
  I -->|no| R
  I -->|yes| F[finish tool]

  F --> Z[Return SQL + trace]

  C --> SB{Step budget left?}
  SB -->|no| FB[Fallback: vanilla candidate]
  FB --> Z
```

---

**Methodology Pipeline (Literature‑Aligned Evaluation)**

```mermaid
flowchart LR
  subgraph Data
    DB[ClassicModels DB]
    Train[Train JSONL]
    Test[Test JSON]
  end

  subgraph Methods
    BL[Baseline ICL Prompting]
    FT[QLoRA Fine‑Tuning]
    RA[Tool‑Driven ReAct Loop\n(Validate → Run → Repair)]
  end

  subgraph Evaluation
    VA[VA: Valid SQL]
    EM[EM: Exact Match]
    EX[EX: Execution Accuracy]
    TS[TS: Test‑Suite Accuracy]
  end

  Train --> FT
  DB --> BL
  DB --> RA
  DB --> VA
  DB --> EX
  DB --> TS
  Test --> VA
  Test --> EM
  Test --> EX
  Test --> TS

  BL --> VA
  BL --> EM
  BL --> EX
  BL --> TS

  FT --> VA
  FT --> EM
  FT --> EX
  FT --> TS

  RA --> VA
  RA --> EM
  RA --> EX
  RA --> TS

  VA --> COMP[Compare Methods]
  EM --> COMP
  EX --> COMP
  TS --> COMP
  COMP --> REPORT[Results + Analysis]
```

---

**Code Flow (Module‑Level Implementation)**

```mermaid
flowchart TD
  NB[notebooks/03_agentic_eval.ipynb\norchestrator] --> P[nl2sql/prompts.py\nREACT_SYSTEM_PROMPT]
  NB --> AU[nl2sql/agent_utils.py\nguardrails + intent]
  NB --> EV[nl2sql/eval.py\nVA/EX/EM/TS]
  NB --> QR[nl2sql/query_runner.py\nsafe SELECT‑only exec]
  NB --> AT[nl2sql/agent_tools.py\ntool interface]

  subgraph AgentTools
    GS[get_schema]
    LS[link_schema]
    EC[extract_constraints]
    GTS[get_table_samples]
    GEN[generate_sql]
    VAL[validate_sql]
    VC[validate_constraints]
    RUN[run_sql]
    REP[repair_sql]
    FIN[finish]
  end

  AT --> GS
  AT --> LS
  AT --> EC
  AT --> GTS
  AT --> GEN
  AT --> VAL
  AT --> VC
  AT --> RUN
  AT --> REP
  AT --> FIN

  GS --> SCH[nl2sql/schema.py\nschema introspection]
  GEN --> LLM[nl2sql/llm.py\nmodel inference]
  REP --> LLM

  LLM --> MODEL[HF Model + QLoRA adapter]
  RUN --> DB[Cloud SQL via SQLAlchemy]

  AU --> POST[nl2sql/postprocess.py\nclean + clamp]

  NB --> RESULTS[results/agent/results_react_200.json]
  EV --> RESULTS
```

---

**Sequence Diagram: Tool‑Driven ReAct Loop (Per Query)**

```mermaid
sequenceDiagram
  autonumber
  participant U as User/Query Source
  participant NB as Notebook react_sql()
  participant LLM as LLM (Llama‑3‑8B + QLoRA)
  participant AT as Agent Tools
  participant PP as Guardrails (clean/postprocess)
  participant VS as validate_sql
  participant VC as validate_constraints
  participant DB as QueryRunner.run (DB)
  
  U->>NB: NLQ
  NB->>AT: get_schema()
  AT->>AT: schema introspection
  AT-->>NB: schema_text
  NB->>AT: link_schema(nlq, schema_text)
  AT-->>NB: linked schema_text
  NB->>AT: extract_constraints(nlq)
  AT-->>NB: constraints
  NB->>LLM: System prompt + trace (User question + linked schema observation)
  LLM-->>NB: Thought + Action: generate_sql[...]
  NB->>AT: generate_sql(nlq, linked_schema_text, constraints)
  AT->>LLM: prompt messages (SYSTEM + schema + NLQ)
  LLM-->>AT: raw SQL
  AT-->>NB: raw SQL
  NB->>PP: clean_candidate + clamps + projection + casing
  PP-->>NB: cleaned SQL (or reject reason)
  NB->>VS: validate_sql(cleaned SQL, schema_text)
  VS-->>NB: {valid: true/false, reason}

  alt validation failed
    NB->>LLM: Observation: validation error
    LLM-->>NB: Thought + Action: repair_sql[error]
  NB->>AT: repair_sql(nlq, bad_sql, error, full_schema_text)
    AT->>LLM: repair prompt (schema + NLQ + error)
    LLM-->>AT: repaired SQL
    AT-->>NB: repaired SQL
    NB->>PP: guardrails
    PP-->>NB: cleaned SQL
    NB->>VS: validate_sql(...)
    VS-->>NB: {valid: true/false}
  end

  alt validation passed
    NB->>VC: validate_constraints(cleaned SQL, constraints)
    VC-->>NB: {valid: true/false, reason}
    alt constraints failed
      NB->>LLM: Observation: constraint error
      LLM-->>NB: Thought + Action: repair_sql[error]
    else constraints passed
      NB->>DB: run_sql(cleaned SQL)
      DB-->>NB: {success: true/false, rows/error}
    end
  end

  alt execution error
    NB->>LLM: Observation: run_sql error
    LLM-->>NB: Thought + Action: repair_sql[error]
    NB->>AT: repair_sql(...)
    AT->>LLM: repair prompt
    LLM-->>AT: repaired SQL
    AT-->>NB: repaired SQL
    NB->>PP: guardrails
    PP-->>NB: cleaned SQL
    NB->>VS: validate_sql(...)
    VS-->>NB: {valid: true/false}
    NB->>VC: validate_constraints(...)
    VC-->>NB: {valid: true/false}
    NB->>DB: run_sql(...)
    DB-->>NB: {success: true/false}
  end

  alt execution ok
    NB->>NB: intent_constraints(nlq, sql)
    NB->>LLM: Observation: intent OK
    LLM-->>NB: Thought + Action: finish[answer, sql]
    NB-->>U: SQL + trace
  end
```

---

**Sequence Diagram: Full Evaluation Run (VA/EX/EM/TS)**

```mermaid
sequenceDiagram
  autonumber
  participant NB as Notebook eval loop
  participant AG as react_sql()
  participant LLM as LLM (Llama‑3‑8B + QLoRA)
  participant DB as Base DB (QueryRunner)
  participant TS as TS DBs (replicas)
  participant EV as Eval functions

  loop For each test item
    NB->>AG: react_sql(nlq)
    AG->>LLM: Thought/Action/Observation loop
    LLM-->>AG: Final SQL + trace
    AG-->>NB: pred_sql, trace
    
    NB->>EV: normalize_sql(pred_sql, gold_sql)
    EV-->>NB: EM
    
    NB->>DB: run_sql(pred_sql)
    DB-->>NB: VA (success/error)
    
    alt VA = 1
      NB->>EV: execution_accuracy(pred_sql, gold_sql)
      EV-->>NB: EX (result equivalence)
      
      NB->>TS: run_sql(pred_sql) on each replica
      TS-->>NB: TS results per DB
      NB->>EV: test_suite_accuracy_for_item(...)
      EV-->>NB: TS
    else VA = 0
      NB->>NB: TS skipped (va=0)
    end

    NB->>NB: append JSON record (pred_sql, va, em, ex, ts, trace)
  end

  NB->>NB: aggregate rates (VA/EM/EX/TS)
  NB->>NB: save results JSON
```

---

**Sequence Diagram: Prompt Construction + Guardrails**

```mermaid
sequenceDiagram
  autonumber
  participant NB as Notebook react_sql()
  participant LLM as LLM
  participant PR as REACT_SYSTEM_PROMPT
  participant AT as Agent Tools
  participant PP as Guardrails
  participant VS as validate_sql
  participant VC as validate_constraints

  NB->>PR: Load system prompt
  NB->>AT: get_schema()
  AT-->>NB: schema_text
  NB->>AT: link_schema(nlq, schema_text)
  AT-->>NB: linked schema_text
  NB->>AT: extract_constraints(nlq)
  AT-->>NB: constraints
  NB->>LLM: System prompt + trace (User question + schema)
  LLM-->>NB: Action: generate_sql[constraints]
  NB->>AT: generate_sql(nlq, linked_schema_text, constraints)
  AT->>LLM: SYSTEM + schema + NLQ message stack
  LLM-->>AT: raw SQL
  AT-->>NB: raw SQL
  NB->>PP: clean_candidate_with_reason
  PP-->>NB: cleaned SQL or reject
  NB->>PP: guarded_postprocess + projection contract + casing
  PP-->>NB: final SQL for validation
  NB->>VS: validate_sql(final SQL, full_schema_text)
  VS-->>NB: {valid, reason}
  NB->>VC: validate_constraints(final SQL, constraints)
  VC-->>NB: {valid, reason}
```

---

**Code Pointers**
- `notebooks/03_agentic_eval.ipynb` (tool‑driven `react_sql` loop, trace logging, evaluation loop)
- `nl2sql/agent_tools.py` (`get_schema`, `link_schema`, `extract_constraints`, `get_table_samples`, `generate_sql`, `validate_sql`, `validate_constraints`, `run_sql`, `repair_sql`, `finish`)
- `nl2sql/prompts.py` (`REACT_SYSTEM_PROMPT`)
- `nl2sql/agent_utils.py` (guardrails, intent constraints, cleaners)
- `nl2sql/postprocess.py` (deterministic SQL clamps and normalization)
- `nl2sql/llm.py` (generation wrapper, SELECT extraction)
- `nl2sql/query_runner.py` (SELECT‑only execution gate)
- `nl2sql/eval.py` (VA/EX/EM/TS)
- `data/classicmodels_test_200.json` (evaluation items)

---

**Literature Anchors**
- ReAct loop: `REFERENCES.md#ref-yao2023-react`
- Agent‑mediated NL→SQL workflow: `REFERENCES.md#ref-ojuri2025-agents`
- Execution feedback: `REFERENCES.md#ref-zhai2025-excot`
- Execution‑based evaluation and TS: `REFERENCES.md#ref-zhong2020-ts`
- Benchmark context for EM limitations: `REFERENCES.md#ref-yu2018-spider`

---

**Decision Log (Demo‑Friendly Format)**

The notebook prints a compact, reasoned decision log per query, e.g.:

```
[step -1] get_schema — loaded schema (ok)
  data: {"tables": ["customers", "orders", ...]}
[step -1] link_schema — prune schema context (ok)
  data: {"schema_text": "...", "changed": true}
[step 0] extract_constraints — heuristic extraction (ok)
  data: {"agg": "COUNT", "limit": 10, ...}
[step 0] generate_sql — model generation (ok)
  data: {"raw_sql": "SELECT ..."}
[step 0] guardrails — cleaned (ok)
  data: {"cleaned_sql": "SELECT ..."}
[step 0] validate_sql — ok (ok)
[step 0] validate_constraints — ok (ok)
[step 0] run_sql — execute (ok)
  data: {"success": true, "rowcount": 10}
[step 0] intent_check — ok (ok)
[step 0] finish — completed (ok)
```

This makes each decision and its justification visible for demos and dissertation narratives.
