# Replication Positioning (Ojuri + Open-Source)

This note defines exactly how to replicate Ojuri-style comparisons while emphasizing open-source reproducibility as the dissertation contribution.

## One-Sentence Positioning

This dissertation replicates the *comparison methodology* of Ojuri et al. (2025) (`REFERENCES.md#ref-ojuri2025-agents`) on a fully open-source, locally reproducible stack, and evaluates whether directional findings persist under constrained compute rather than claiming proprietary-model score parity.

## What Is Replicated vs What Is Not

Replicated:
- controlled contrast families: prompting (`k=0` vs `k>0`), fine-tuning (base vs QLoRA), and execution infrastructure checks,
- shared metric hierarchy centered on semantic behavior (EX/TS first, VA second, EM diagnostic),
- paired significance reporting and uncertainty reporting on identical items.

Not replicated:
- proprietary model classes and enterprise infrastructure from source-paper setups,
- exact absolute percentages from closed-model environments,
- unconstrained agent complexity as a primary contribution.

Boundary documents:
- `2_METHODOLOGY.md`
- `4_EVALUATION.md`
- `6_LIMITATIONS.md`

## Open-Source Emphasis (What You Must Keep Explicit)

- Open model family and tooling provenance: `REFERENCES.md#ref-dubey2024-llama3`, `REFERENCES.md#ref-wolf2020-transformers`.
- PEFT/QLoRA under constrained hardware: `REFERENCES.md#ref-hu2021-lora`, `REFERENCES.md#ref-dettmers2023-qlora`, `REFERENCES.md#ref-ding2023-peft`, `REFERENCES.md#ref-goswami2024-peft`.
- Public benchmark/evaluation semantics: `REFERENCES.md#ref-yu2018-spider`, `REFERENCES.md#ref-zhong2020-ts`.
- Agent infrastructure as bounded execution support: `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-wang2018-eg-decoding`, `REFERENCES.md#ref-zhai2025-excot`.
- Schema/value bottlenecks reported explicitly: `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`, `REFERENCES.md#ref-lin2020-bridge`.

Implementation evidence:
- `10_EXPERIMENT_EXECUTION_PLAN.md`
- `results/RUN_INDEX.md`
- `results/analysis/summary.md`
- `results/analysis/paired_deltas.csv`
- `results/analysis/failure_taxonomy.csv`
- `LOGBOOK.md`

## Conclusion Targets (What You Are Trying to Show)

1. Prompting effect is real under fixed weights.
- Expected evidence: paired EX/TS deltas for `k=0` vs `k=3` with Wilson intervals and McNemar tests.
- Grounding: `REFERENCES.md#ref-brown2020-gpt3`, `REFERENCES.md#ref-mosbach2023-icl`.

2. QLoRA effect is tested as semantic adaptation under constraints.
- Expected evidence: base vs QLoRA at matched `k`, with effect sizes and significance.
- Grounding: `REFERENCES.md#ref-dettmers2023-qlora`, `REFERENCES.md#ref-ding2023-peft`, `REFERENCES.md#ref-goswami2024-peft`.

3. ReAct is an infrastructure claim unless semantic deltas are significant.
- Expected evidence: VA/trace stability improvements plus explicit failure traces; avoid semantic overclaims when EX/TS deltas are weak.
- Grounding: `REFERENCES.md#ref-yao2023-react`, `REFERENCES.md#ref-wang2018-eg-decoding`, `REFERENCES.md#ref-zhai2025-excot`.

4. Residual failures should be explained by linking/composition categories.
- Expected evidence: concentration of failures in join path, aggregation scope, value linking, projection.
- Grounding: `REFERENCES.md#ref-wang2020-ratsql`, `REFERENCES.md#ref-li2023-resdsql`, `REFERENCES.md#ref-lin2020-bridge`.

5. Replication success is directional, not absolute.
- Expected evidence: trend agreement/disagreement table against Ojuri-style comparison axes.
- Grounding: `REFERENCES.md#ref-ojuri2025-agents`, plus survey context in `REFERENCES.md#ref-zhu2024-survey`, `REFERENCES.md#ref-hong2025-survey`, `REFERENCES.md#ref-gao2025-llm-sql`.

## Required Statistical Language

- Rate + 95% Wilson interval for each run (`REFERENCES.md#ref-wilson1927`).
- Paired delta + exact McNemar p-value on shared items (`REFERENCES.md#ref-mcnemar1947`).
- Interpret significance with NLP testing caution (`REFERENCES.md#ref-dror2018-significance`).

Code anchors:
- `nl2sql/research_stats.py`
- `scripts/generate_research_comparison.py`

## Dissertation-Ready Claim Templates

Template A (trend replication):
- Under the same evaluation harness, the open-source stack reproduced the *direction* of [contrast] reported in Ojuri-style comparisons, with [metric] delta of [X] pp (paired McNemar p=[Y]).

Template B (open-source contribution):
- The contribution is reproducible methodological evidence: all runs are local, versioned, and re-runnable from notebook configuration through JSON artifacts and paired statistical summaries.

Template C (scope control):
- These findings support comparative behavior under constrained open-source conditions and do not claim equivalence to proprietary-model absolute performance.

## Viva-Safe Short Answer

I replicate Ojuri at the level of experimental logic and evaluation discipline, then show what changes when the same contrasts are run on an open-source local stack with full artifact transparency.
