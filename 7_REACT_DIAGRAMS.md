# ReAct Loop Diagram (Tool-Driven NL->SQL)

This file keeps a single high-level diagram for explaining the tool-driven ReAct loop used in evaluation.

- Authoritative loop: `notebooks/03_agentic_eval.ipynb` (`react_sql`)
- Tool interface: `nl2sql/agent_tools.py`

```mermaid
flowchart TD
  A[User NLQ] --> B[Bootstrap trace]
  B --> B1[Setup: get_schema]
  B1 --> LS[Setup: link_schema]
  LS --> B2[Observation: focused schema + join hints]
  B2 --> C[LLM Thought]

  C --> D{Action chosen}
  D -->|get_schema / link_schema| BLK[Blocked: setup-only]
  BLK --> C

  D -->|non-repair + constraints missing| FC[Forced: extract_constraints]
  FC --> EC[extract_constraints tool]
  EC --> EC1[Observation: constraints]
  EC1 --> C

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
  I -->|yes| AF[Auto-finish]
  AF --> F[finish tool]

  F --> Z[Return SQL + trace]

  C --> SB{Step budget left?}
  SB -->|no| FB[Fallback: vanilla candidate]
  FB --> Z
```
