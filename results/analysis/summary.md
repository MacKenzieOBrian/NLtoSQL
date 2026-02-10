# Research Comparison Summary

## Loaded Runs
- Base | k=0: loaded from `results/results_zero_shot_200.json` (n=200.0)
- Base | k=3: loaded from `results/results_few_shot_k3_200.json` (n=200.0)
- QLoRA | k=0: missing (checked `results/qlora/results_zero_shot_200.json`)
- QLoRA | k=3: missing (checked `results/qlora/results_few_shot_k3_200.json`)
- ReAct infra: loaded from `results/agent/results_react_200 (9).json` (n=20.0)

## Overall Metrics
| run_label | n | va_pct | em_pct | ex_pct | ts_pct |
| --- | --- | --- | --- | --- | --- |
| Base \| k=0 | 200 | 81.00 | 0.00 | 7.50 |  |
| Base \| k=3 | 200 | 85.50 | 32.50 | 37.00 |  |
| ReAct infra | 20 | 95.00 | 35.00 | 55.00 | 57.89 |

## Paired Delta Highlights
| comparison_label | metric | n_overlap | left_rate | right_rate | delta_pct | mcnemar_p | sig_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Few-shot gain (Base: k=0 -> k=3) | va | 200 | 81.00 | 85.50 | +4.50 | 0.1221 | 0 |
| Few-shot gain (Base: k=0 -> k=3) | em | 200 | 0.00 | 32.50 | +32.50 | 0.0000 | 1 |
| Few-shot gain (Base: k=0 -> k=3) | ex | 200 | 7.50 | 37.00 | +29.50 | 0.0000 | 1 |
| Execution infra effect (Base k=3 -> ReAct) | va | 20 | 95.00 | 95.00 | +0.00 | 1.0000 | 0 |
| Execution infra effect (Base k=3 -> ReAct) | em | 20 | 10.00 | 35.00 | +25.00 | 0.1250 | 0 |
| Execution infra effect (Base k=3 -> ReAct) | ex | 20 | 20.00 | 55.00 | +35.00 | 0.0391 | 1 |

## Notes
- Agent run is treated as execution infrastructure, not the primary contribution.
- Primary claims should use base/QLoRA and k=0/k=3 controlled comparisons.