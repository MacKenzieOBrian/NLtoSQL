# Replication Positioning (Concise)

## Position
This dissertation replicates Ojuri-style comparison logic on a fully open-source local stack. Success is directional replication, not proprietary score parity.

## What is replicated
- comparison structure (prompting vs fine-tuning vs execution support)
- EX/TS-first interpretation
- paired statistical reporting

## What is not replicated
- proprietary model stacks
- proprietary infra settings
- exact absolute scores from closed systems

## Claim boundaries
- ReAct is an infrastructure claim unless EX/TS gains are significant.
- EM is diagnostic.
- Strong claims require paired deltas + uncertainty.

## Evidence map
- Method deltas: `results/analysis/paired_deltas.csv`
- Overall metrics: `results/analysis/overall_metrics_wide.csv`
- Run coverage: `results/analysis/run_manifest.csv`
- Error mechanism: `results/analysis/failure_taxonomy.csv`

## Anchor reference
`REFERENCES.md#ref-ojuri2025-agents`
