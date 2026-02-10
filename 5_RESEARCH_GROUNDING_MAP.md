# Research Grounding Map

This file maps dissertation claims to literature anchors, code artifacts, and expected evidence.

## Claim-to-Evidence Matrix

| Claim | Literature anchor | Code / artifact | Required evidence |
| --- | --- | --- | --- |
| Open-source replication can reproduce the *direction* of enterprise NL->SQL findings under constrained compute | `REFERENCES.md#ref-ojuri2025-agents` | `2_METHODOLOGY.md`, `4_EVALUATION.md`, `results/analysis/summary.md` | replication table with metric deltas vs source paper and explicit trend match/mismatch notes |
| Few-shot improves structure under fixed weights | `REFERENCES.md#ref-brown2020-gpt3`, `REFERENCES.md#ref-mosbach2023-icl` | `notebooks/02_baseline_prompting_eval.ipynb`, `results/baseline/*.json` | k=0 vs k=3 deltas, CI, paired significance |
| QLoRA improves semantic mapping under constraints | `REFERENCES.md#ref-ding2023-peft`, `REFERENCES.md#ref-goswami2024-peft` | `notebooks/05_qlora_train_eval.ipynb`, `results/qlora/*.json` | base vs qlora deltas at matched k, CI, paired significance |
| Execution guidance improves validity and observability | `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-wang2018-eg-decoding`, `REFERENCES.md#ref-zhai2025-excot` | `nl2sql/react_pipeline.py`, `notebooks/03_agentic_eval.ipynb` | `react_core` vs `react_no_repair` comparison, trace evidence |
| Schema/value linking remains a bottleneck | `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`, `REFERENCES.md#ref-lin2020-bridge` | `nl2sql/agent_schema_linking.py`, `results/analysis/failure_taxonomy.csv` | failure class concentration in join/value categories |
| EX/TS should drive semantic claims | `REFERENCES.md#ref-yu2018-spider`, `REFERENCES.md#ref-zhong2020-ts` | `nl2sql/eval.py`, `4_EVALUATION.md` | claims framed on EX/TS first, EM diagnostic only |

## Minimal Reporting Standard

For each method comparison:
- include point estimates and 95% Wilson intervals,
- include paired delta and exact McNemar p-value,
- include top failure categories that explain metric movement,
- include sample size and run configuration metadata.

## Approved Claim Language

Use this pattern in results text:

Method B improved EX by +X pp over Method A on the same 200 items (paired McNemar p=Y), with residual errors dominated by [categories].

## Disallowed Overclaims

- ReAct solves semantic alignment end-to-end.
- EM increase alone implies semantic improvement.
- Single-run percentage differences without uncertainty are conclusive.
- ClassicModels performance automatically generalizes to arbitrary enterprise schemas.
