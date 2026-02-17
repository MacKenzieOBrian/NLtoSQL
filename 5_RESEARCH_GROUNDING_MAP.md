# Research Grounding Map

## Claims, anchors, and current evidence

| Claim | Literature anchor | Evidence artifact | Current status (2026-02-15 snapshot) |
| --- | --- | --- | --- |
| Few-shot prompting improves semantic performance under fixed weights | `REFERENCES.md#ref-mosbach2023-icl` | `results/analysis/paired_deltas.csv` | Supported for both Llama and Qwen at `k=3` on EX with significant paired tests |
| QLoRA can improve adaptation under constrained compute | `REFERENCES.md#ref-dettmers2023-qlora`, `REFERENCES.md#ref-ding2023-peft` | `results/analysis/paired_deltas.csv`, `results/analysis/overall_metrics_wide.csv` | Mixed: current Llama QLoRA snapshot does not exceed base EX at `k=3`; reported as empirical outcome |
| EX/TS are the semantic basis of NL->SQL claims | `REFERENCES.md#ref-yu2018-spider`, `REFERENCES.md#ref-zhong2020-ts` | `results/analysis/overall_metrics_wide.csv`, `results/analysis/paired_deltas.csv` | Applied across all result interpretation |
| ReAct is infrastructure unless EX/TS gains are demonstrated | `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-wang2018-eg-decoding` | `results/agent/react_infra_n20_v9.json`, traces + paired deltas | Treated as secondary claim; no significant EX gain in current `n=20` overlap |
| Residual errors cluster in linking/composition | `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`, `REFERENCES.md#ref-lin2020-bridge` | `results/analysis/failure_taxonomy.csv` | Supported: projection, invalid SQL, join path, value linking remain dominant |

## High-value numbers to quote directly
- Base `k=0 -> k=3`: EX `48.5% -> 61.0%` (`+12.5pp`, McNemar `p=0.0001`).
- QLoRA `k=0 -> k=3`: EX `45.5% -> 55.0%` (`+9.5pp`, `p=0.0003`).
- Qwen baseline: EX `49.5% (k=0)`, `58.5% (k=3)`, `65.0% (k=5)`.
- ReAct infra vs base `k=3` on overlap `n=20`: EX delta `-15.0pp`, not significant.

## Writing rule for results paragraphs
Every claim paragraph should contain:
1. setup (model/method, `k`, seed policy, `n`),
2. metric delta (EX/TS first),
3. uncertainty/significance,
4. explanation via failure taxonomy,
5. explicit artifact path.
