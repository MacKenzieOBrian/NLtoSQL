# ReAct Loop Diagram (Tool-Driven NL->SQL)

This file keeps a single high-level diagram for explaining the tool-driven ReAct loop used in evaluation.

- Authoritative loop: `notebooks/03_agentic_eval.ipynb` (`react_sql`)
- Tool interface: `nl2sql/agent_tools.py`

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
  G --> H[Guardrails: clean + postprocess + projection + casing]
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
