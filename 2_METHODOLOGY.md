# Methodology

The methodology is designed to isolate causal effects, not maximize a single headline score.

## Research Angle and Replication Scope

This dissertation follows the methodology pattern in Ojuri et al. (2025) (`REFERENCES.md#ref-ojuri2025-agents`) but re-implements it with an open-source toolchain and a locally runnable LLM workflow.

Replication target:
- match the experimental structure (few-shot vs fine-tuning comparisons on the same ClassicModels-style task),
- match the evaluation style (VA/EX/TS/EM on the same held-out items),
- test whether the same directional conclusions hold under constrained, reproducible open-source conditions.

This is a replication-oriented comparison study, not a claim of exact model parity with proprietary systems (for example GPT-4 class models in the source paper).

For an examiner-facing replication narrative and claim templates, see:
- `11_REPLICATION_POSITIONING.md`

## Research Questions

- RQ1: How much does few-shot prompting (`k=0` vs `k=3`) improve VA/EX/TS/EM under fixed model weights?
- RQ2: How much does QLoRA improve VA/EX/TS/EM over the same prompting settings?
- RQ3: How much does execution-guided ReAct improve validity and robustness, and which semantic errors remain?

## Experimental Principle

Use one change at a time.

- Fixed test set: `data/classicmodels_test_200.json`
- Shared evaluation harness: `nl2sql/eval.py`
- Shared safety gate: `nl2sql/query_runner.py`
- Matched prompt schema context across methods
- Run-level JSON outputs archived for each condition

## Systems Compared

Primary research systems:
- Base model, `k=0`
- Base model, `k=3`
- QLoRA model, `k=0`
- QLoRA model, `k=3`

Execution infrastructure system:
- ReAct core loop (tool-driven), reported as robustness/validity support rather than primary contribution.

## ReAct Core Loop (Infrastructure)

Default loop used for comparison:
1. Setup once: `get_schema` -> `link_schema` (optional in ablation)
2. Per question: `extract_constraints` -> `generate_sql`
3. Deterministic cleanup: candidate cleaning + guarded postprocess
4. Validation gate: `validate_sql` -> `validate_constraints`
5. Execute: `run_sql`
6. Repair only if validation/execution fails
7. Stop immediately on first successful execution
8. If no successful execution is reached within repair budget, emit `no_prediction` (do not return known-failed SQL)

Implementation source:
- `nl2sql/react_pipeline.py`
- `nl2sql/agent_tools.py`

## Literature-Grounded Design Decisions

| Design decision | Literature basis | Implementation |
| --- | --- | --- |
| EX/TS prioritized over EM for semantic interpretation | Spider + test-suite semantics (`ref-yu2018-spider`, `ref-zhong2020-ts`) | `nl2sql/eval.py`, `4_EVALUATION.md` |
| Prompting and fine-tuning compared under shared conditions | Fair ICL vs FT framing (`ref-mosbach2023-icl`) | `notebooks/02_baseline_prompting_eval.ipynb`, `notebooks/05_qlora_train_eval.ipynb` |
| QLoRA used to test adaptation under compute limits | LoRA/QLoRA + PEFT evidence (`ref-hu2021-lora`, `ref-dettmers2023-qlora`, `ref-ding2023-peft`, `ref-goswami2024-peft`) | `results/adapters/qlora_classicmodels/`, `notebooks/05_qlora_train_eval.ipynb` |
| ReAct kept bounded and tool-explicit | ReAct + execution-guided work (`ref-yao2023-react`, `ref-wang2018-eg-decoding`, `ref-zhai2025-excot`) | `nl2sql/react_pipeline.py`, `notebooks/03_agentic_eval.ipynb` |
| Schema/value linking treated as explicit bottleneck | RAT-SQL, RESDSQL, BRIDGE (`ref-wang2020-ratsql`, `ref-li2023-resdsql`, `ref-lin2020-bridge`) | `nl2sql/agent_schema_linking.py`, `nl2sql/constraint_policy.py` |

## Reproducibility Protocol

- Dependencies pinned: `requirements.txt`
- Deterministic decode defaults in generation path
- Randomness sources logged when used (e.g., exemplar sampling seed)
- Run metadata saved with each JSON artifact
- Dataset fingerprinting and config snapshots stored in ReAct reports
- QLoRA training-argument rationale and run-log template documented in `8_QLORA_CONFIGURATION.md`

## Repeated-Run and Defensibility Protocol

For every primary comparison:
- report point estimate and 95% Wilson interval for VA/EX/TS/EM,
- use paired comparisons on the same examples,
- report exact McNemar p-values for binary paired outcomes,
- distinguish statistical significance from practical effect size.

Statistical basis:
- Wilson interval: `REFERENCES.md#ref-wilson1927`
- McNemar paired binary test: `REFERENCES.md#ref-mcnemar1947`
- NLP significance-testing practice guidance: `REFERENCES.md#ref-dror2018-significance`

Implemented in:
- `nl2sql/research_stats.py`
- `scripts/generate_research_comparison.py`
- `results/analysis/paired_deltas.csv`

Execution sequence and exact notebook settings are operationalized in:
- `10_EXPERIMENT_EXECUTION_PLAN.md`

## Error Analysis Protocol

Each failed item is assigned to one dominant category:
- `invalid_sql`
- `join_path`
- `aggregation`
- `value_linking`
- `projection`
- `ordering_limit`
- `other_semantic`

Purpose: connect metric movement to error-type movement, not only to final percentages.

## Evaluation Metrics

- VA: executable SQL rate
- EX: execution result equivalence on base DB
- TS: execution equivalence across perturbed suite DBs
- EM: normalized exact SQL string match (diagnostic)

Rationale:
- primary metrics: EX and TS (semantic behavior),
- secondary metric: VA (validity),
- diagnostic metric: EM (surface-form agreement).

## Threats to Validity and Controls

Internal validity controls:
- same test items across methods,
- same evaluator and DB execution policy,
- same postprocess class across base/QLoRA runs.

Remaining threats:
- single schema/domain (ClassicModels),
- heuristic linker/constraints instead of learned semantic parser,
- TS approximation via replicas rather than distilled suite generation.

## What This Method Does Not Claim

- It does not claim a state-of-the-art universal Text-to-SQL agent.
- It does not claim ReAct solves semantic alignment.
- It does not claim transfer to arbitrary enterprise schemas without new evidence.

The methodological claim is narrower: under fixed constraints, controlled prompting + QLoRA comparisons produce interpretable gains, while execution-guided infrastructure improves validity and traceability.
