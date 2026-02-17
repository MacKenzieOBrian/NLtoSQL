# Logbook (Condensed Research Journey)

## Project objective
Build a reproducible open-source NL->SQL evaluation pipeline and test how prompting, QLoRA, and ReAct-style infrastructure affect semantic performance under constrained compute.

## Phase 1: Framing and scope (Sep-Oct 2025)
- Defined contribution around reproducibility and open-source constraints.
- Shifted evaluation emphasis from syntax quality to execution semantics.
- Locked claim boundary: EX/TS-first evidence, not benchmark storytelling.

## Phase 2: Baseline pipeline (Nov-Dec 2025)
- Implemented schema extraction, execution harness, and deterministic evaluation path.
- Completed base prompting runs on ClassicModels 200-item set.
- Established that few-shot improves structure and semantics relative to zero-shot.

## Phase 3: QLoRA adaptation (Jan 2026)
- Built adapter training/evaluation workflow with constrained-resource settings.
- Produced Llama QLoRA runs and matched base-vs-QLoRA comparisons.
- Observed mixed adaptation gains in current snapshot, especially at matched `k=3`.

## Phase 4: ReAct infrastructure alignment (Feb 2026)
- Replaced fixed controller ordering with model-driven Thought/Action/Observation loop.
- Added strict `finish` gating and bounded repair/step budgets.
- Removed deterministic repair templates from the main reported loop path.
- Interpreted ReAct as infrastructure unless EX/TS gains are demonstrated.

## Current evidence position (from `results/analysis/`)
- Strong few-shot EX gains for base and QLoRA (`k=0 -> k=3`).
- Current Llama QLoRA snapshot does not outperform base EX at `k=3`.
- Qwen baseline shows stronger EX at higher `k` in available runs.
- ReAct diagnostic value is clear; performance claim remains secondary at current sample size.

## Next analysis actions
1. Complete remaining planned sweeps (especially missing matched QLoRA/model-family cells).
2. Run TS checks on selected stable `k=3` runs.
3. Regenerate comparison artifacts and finalize claim language from paired + taxonomy outputs.
