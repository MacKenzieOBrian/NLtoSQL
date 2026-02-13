# Replication Positioning

## One-line position

This dissertation replicates Ojuri-style comparison logic on a fully open-source, reproducible local stack, aiming for directional replication rather than proprietary score parity.

## What is replicated

- comparison structure: prompting vs fine-tuning vs execution support,
- semantic-first interpretation (EX/TS primary),
- paired statistical reporting on shared examples.

## What is not replicated

- proprietary model stacks,
- proprietary infrastructure,
- exact absolute scores from closed environments.

## Contribution focus

- reproducible open-source experiment pipeline,
- controlled comparisons under constrained hardware,
- artifact-backed interpretation with uncertainty and paired significance.

## Claim boundary

- ReAct is an infrastructure claim (validity/traceability) unless EX/TS gains are significant.
- EM is diagnostic, not a primary semantic claim.
- Directional agreement is a success criterion; exact parity is not required.

## Paragraph starter (for dissertation)

This work reproduces the experimental logic of enterprise NL-to-SQL comparison studies in an open-source environment. The contribution is not proprietary-model parity, but a transparent and rerunnable evaluation pipeline that tests whether key directional findings persist under constrained compute.
