# Research Comparison Summary

## Loaded Runs
- Base | k=0: loaded from `results/baseline/baseline_k0.json` (n=200)
- Base | k=3: loaded from `results/baseline/baseline_k3.json` (n=200)
- QLoRA | k=0: loaded from `results/qlora/qlora_k0.json` (n=200)
- QLoRA | k=3: loaded from `results/qlora/qlora_k3.json` (n=200)
- ReAct infra: loaded from `results/agent/react_infra_n20_v9.json` (n=20)

## Overall Metrics
| run_label | n | va_pct | em_pct | ex_pct | ts_pct |
| --- | --- | --- | --- | --- | --- |
| Base \| k=0 | 200 | 87.50 | 0.00 | 48.50 |  |
| Base \| k=3 | 200 | 89.50 | 35.50 | 61.00 |  |
| QLoRA \| k=0 | 200 | 86.50 | 3.00 | 45.50 |  |
| QLoRA \| k=3 | 200 | 87.00 | 35.00 | 55.00 |  |
| ReAct infra | 20 | 95.00 | 35.00 | 55.00 | 57.89 |

## Paired Delta Highlights
| comparison_label | metric | n_overlap | left_rate | right_rate | delta_pct | mcnemar_p | sig_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Few-shot gain (Base: k=0 -> k=3) | va | 200 | 87.50 | 89.50 | +2.00 | 0.4807 | 0 |
| Few-shot gain (Base: k=0 -> k=3) | em | 200 | 0.00 | 35.50 | +35.50 | 0.0000 | 1 |
| Few-shot gain (Base: k=0 -> k=3) | ex | 200 | 48.50 | 61.00 | +12.50 | 0.0001 | 1 |
| Few-shot gain (QLoRA: k=0 -> k=3) | va | 200 | 86.50 | 87.00 | +0.50 | 1.0000 | 0 |
| Few-shot gain (QLoRA: k=0 -> k=3) | em | 200 | 3.00 | 35.00 | +32.00 | 0.0000 | 1 |
| Few-shot gain (QLoRA: k=0 -> k=3) | ex | 200 | 45.50 | 55.00 | +9.50 | 0.0003 | 1 |
| Fine-tune gain (k=0: Base -> QLoRA) | va | 200 | 87.50 | 86.50 | -1.00 | 0.7905 | 0 |
| Fine-tune gain (k=0: Base -> QLoRA) | em | 200 | 0.00 | 3.00 | +3.00 | 0.0312 | 1 |
| Fine-tune gain (k=0: Base -> QLoRA) | ex | 200 | 48.50 | 45.50 | -3.00 | 0.3269 | 0 |
| Fine-tune gain (k=3: Base -> QLoRA) | va | 200 | 89.50 | 87.00 | -2.50 | 0.2668 | 0 |
| Fine-tune gain (k=3: Base -> QLoRA) | em | 200 | 35.50 | 35.00 | -0.50 | 1.0000 | 0 |
| Fine-tune gain (k=3: Base -> QLoRA) | ex | 200 | 61.00 | 55.00 | -6.00 | 0.0501 | 0 |
| Execution infra effect (Base k=3 -> ReAct) | va | 20 | 95.00 | 95.00 | +0.00 | 1.0000 | 0 |
| Execution infra effect (Base k=3 -> ReAct) | em | 20 | 20.00 | 35.00 | +15.00 | 0.3750 | 0 |
| Execution infra effect (Base k=3 -> ReAct) | ex | 20 | 70.00 | 55.00 | -15.00 | 0.2500 | 0 |

## Notes
- Agent run is treated as execution infrastructure, not the primary contribution.
- Primary claims should use base/QLoRA and k=0/k=3 controlled comparisons.