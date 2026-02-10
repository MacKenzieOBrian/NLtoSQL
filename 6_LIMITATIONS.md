# Limitations

This section defines what cannot be concluded from the current evidence.

## External Validity

- Single schema/domain (ClassicModels).
- Results may not transfer to unseen enterprise schemas without additional data.
- SQL dialect assumptions are MySQL-oriented.

Consequence: claims are restricted to this benchmark setting.

## Evaluation Limits

- EX is evaluated on a single base DB state.
- TS uses perturbed replicas, not fully distilled test suites.
- EM underestimates semantic equivalence by construction.

Consequence: semantic confidence is improved, not absolute.

## Modeling Limits

- Heuristic schema linking (not learned linker).
- Heuristic constraint extraction (not semantic parser).
- ReAct repair is bounded and may stop before full semantic correction.

Consequence: the system remains vulnerable to complex join/aggregation/value-grounding errors.

## Methodological Limits

- Compute budget limits breadth of hyperparameter sweeps.
- Some comparisons currently have small-n slices (e.g., interim agent runs).
- Statistical significance on small paired sets should be treated cautiously.

Consequence: emphasize effect sizes and confidence intervals, not only p-values.

## Infrastructure vs Research Boundary

- ReAct loop is designed for robustness and observability.
- It is not claimed as the primary source of semantic gains.

Consequence: dissertation claims must center on controlled prompting/fine-tune comparisons.

## Planned Mitigations (Future Work)

- Add additional schemas for external-validity checks.
- Replace heuristic linker with a learned or hybrid linker ablation.
- Expand repeated-run protocol and report pooled uncertainty.
- Introduce richer semantic error labels for join-chain and aggregation-scope subtypes.
