# Run Snapshot: 2026-02-10 (Baseline + QLoRA)

This folder freezes the core artifacts for one comparable run cycle.

## Core Metrics

| Condition | n | VA | EM | EX | Commit |
| --- | ---: | ---: | ---: | ---: | --- |
| Base k=0 | 200 | 0.875 | 0.000 | 0.485 | d2eaa50 |
| Base k=3 | 200 | 0.895 | 0.355 | 0.610 | d2eaa50 |
| QLoRA k=0 | 200 | 0.865 | 0.030 | 0.455 | 501a24a |
| QLoRA k=3 | 200 | 0.870 | 0.350 | 0.550 | 501a24a |

## Pairwise Deltas (point estimates)

- Few-shot gain (Base EX): +12.5 pp
- Few-shot gain (QLoRA EX): +9.5 pp
- Fine-tune gain at k=0 (EX): -3.0 pp
- Fine-tune gain at k=3 (EX): -6.0 pp

## Important Controls

- Baseline run commit: `d2eaa50`
- QLoRA run commit: `501a24a`
- If commits differ, rerun one side at a matched commit before making strong causal claims.

## Included Artifacts

- `baseline/results_zero_shot_200.json`
- `baseline/results_few_shot_k3_200.json`
- `qlora/results_zero_shot_200.json`
- `qlora/results_few_shot_k3_200.json`
- `analysis/summary.md`
- `analysis/overall_metrics_wide.csv`
- `analysis/paired_deltas.csv`
- `adapters/adapter_config.json`
- `adapters/trainer_state.json`
- `adapters/adapter_model.sha256.txt`
