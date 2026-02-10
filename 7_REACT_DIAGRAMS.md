# ReAct Infrastructure Diagrams

These diagrams intentionally show ReAct as execution infrastructure supporting the primary research comparisons.

## Core Execution Loop (Minimal Default)

```mermaid
flowchart TD
  A["NL question"] --> B["Setup: get_schema"]
  B --> C["Optional: link_schema"]
  C --> D["extract_constraints"]
  D --> E["generate_sql"]
  E --> F["deterministic cleanup"]
  F --> G["validate_sql"]
  G --> H{"valid?"}
  H -->|no| R1["repair_sql"]
  R1 --> F
  H -->|yes| I["validate_constraints"]
  I --> J{"constraints pass?"}
  J -->|no| R2["repair_sql"]
  R2 --> F
  J -->|yes| K["run_sql"]
  K --> L{"execution success?"}
  L -->|no| R3["repair_sql"]
  R3 --> F
  L -->|yes| M["finish (stop on first success)"]
```

## Dissertation Evaluation Structure

```mermaid
flowchart LR
  A["Base model"] --> B["k=0"]
  A --> C["k=3"]
  D["QLoRA model"] --> E["k=0"]
  D --> F["k=3"]

  B --> G["Shared evaluator (VA/EX/TS/EM)"]
  C --> G
  E --> G
  F --> G

  H["ReAct core loop"] --> I["Execution infrastructure"]
  I --> G

  G --> J["Overall metrics + CI"]
  G --> K["Paired deltas + McNemar"]
  G --> L["Failure taxonomy"]

  J --> M["Primary claims: prompting + QLoRA"]
  K --> M
  L --> M
  I --> N["Secondary claims: validity stabilization"]
```

## Reading Guide for Viva

- Left-to-right in the second diagram: controlled method comparisons.
- ReAct is shown as a separate lane to prevent overclaiming.
- Claims should come from paired deltas and error shifts, not from single-run percentages alone.
