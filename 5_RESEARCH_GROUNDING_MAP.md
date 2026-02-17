# Research Grounding Map (Concise)

| Claim | Literature anchor | Evidence artifact |
| --- | --- | --- |
| Prompting improves outcomes under fixed weights | `REFERENCES.md#ref-mosbach2023-icl` | `results/analysis/paired_deltas.csv` (k deltas) |
| QLoRA can improve adaptation under constrained compute | `REFERENCES.md#ref-dettmers2023-qlora`, `REFERENCES.md#ref-ding2023-peft` | base vs qlora paired deltas |
| Semantic claims should be EX/TS-first | `REFERENCES.md#ref-yu2018-spider`, `REFERENCES.md#ref-zhong2020-ts` | `overall_metrics_wide.csv`, `paired_deltas.csv` |
| ReAct is execution infrastructure unless EX/TS gain is shown | `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-wang2018-eg-decoding` | `results/agent/*.json`, model-driven Thought/Action/Observation traces |
| Residual errors concentrate in linking/composition | `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`, `REFERENCES.md#ref-lin2020-bridge` | `failure_taxonomy.csv` |

## Writing rule
Every result sentence should cite:
1. comparison setup,
2. metric delta,
3. uncertainty/significance,
4. artifact path.
