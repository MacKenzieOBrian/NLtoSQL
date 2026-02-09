# ReAct Loop Diagram (Tool-Driven NL->SQL)

This file keeps high-level diagrams for explaining the tool-driven ReAct loop and the
new guardrails/constraint checks added for EX accuracy.

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
  EC --> EC1[Observation: constraints\n(explicit fields, value columns,\nentity identifiers, required tables)]
  EC1 --> C

  D -->|generate_sql| G[generate_sql tool]
  G --> RR{Rerank candidates?}
  RR -->|yes| RR1[Pick best by validate_sql + validate_constraints]
  RR -->|no| H
  RR1 --> H
  H[Guardrails: clean + postprocess\n(strip ranking, preserve listing IDs,\nskip minimal projection when fields enumerated)]
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
  SB -->|no| FB[Fallback: vanilla candidate\n(validated + constrained)]
  FB --> Z
```

```mermaid
flowchart TD
  Q[NLQ] --> EC[extract_constraints]
  EC --> EF[explicit_fields]
  EC --> EP[explicit_projection\n(only when NLQ enumerates fields)]
  EC --> EH[entity_hints]
  EC --> EI[entity_identifiers\n(customerName/orderNumber/etc.)]
  EC --> VC[value_columns]
  EC --> RT[required_tables\n(order totals -> orderdetails;\npayments -> payments)]

  EF --> VCHECK[validate_constraints]
  EP --> VCHECK
  EH --> VCHECK
  EI --> VCHECK
  VC --> VCHECK
  RT --> VCHECK

  VCHECK --> OK{Pass?}
  OK -->|no| R[repair_sql]
  OK -->|yes| RUN[run_sql]
```
