# Logbook (Condensed Timeline)

## Project aim

Evaluate how far open-source NL->SQL can be improved under constrained hardware using controlled comparisons (baseline prompting, QLoRA, and execution infrastructure).

## Phase 1 — Planning and Scope (Sep–Oct 2025)

| Date | Key outcome | Evidence relevance |
| --- | --- | --- |
| 2025-09-29 | Reproducibility gap defined as core research angle | frames contribution boundary |
| 2025-10-13 | EX/TS/VA/EM evaluation framing adopted | metric hierarchy rationale |
| 2025-10-27 | Infrastructure moved to execution-ready architecture (`QueryRunner`) | enabled tool-driven evaluation |

## Phase 2 — Baseline Setup and Prompting (Nov–Dec 2025)

| Date | Key outcome | Evidence relevance |
| --- | --- | --- |
| 2025-11-03 | Stable execution pipeline established | prerequisite for EX/TS claims |
| 2025-12-14 | Few-shot baseline introduced with deterministic guardrails | controlled prompting comparisons |
| 2025-12-23 | Baseline 200-item runs completed (`k=0`, `k=3`) | primary prompt-effect evidence |

## Phase 3 — QLoRA (Jan 2026)

| Date | Key outcome | Evidence relevance |
| --- | --- | --- |
| 2026-01-12 | QLoRA run showed semantic lift vs untuned baseline conditions | primary adaptation-effect evidence |

## Phase 4 — ReAct and Error Analysis (Jan–Feb 2026)

| Date | Key outcome | Evidence relevance |
| --- | --- | --- |
| 2026-01-25 | High VA but low EX on full-set check | confirmed semantic bottleneck |
| 2026-01-29 | Structured trace logging added | supports auditability claims |
| 2026-02-09 | Constraint/linking refinements iterated from error patterns | links failures to model behavior |
| 2026-02-12 | Loop hardened to return `no_prediction` on exhausted budget | aligns implementation and viva narrative |

## Current position

- Primary claims: baseline vs QLoRA with paired statistics.
- ReAct role: infrastructure for execution robustness and traceability.
- Dominant residual errors: join path, aggregation scope, value linking.

## Writing use (quick)

For each claim section:
1. State comparison setup.
2. Report EX/TS first (then VA, EM).
3. Add paired delta + significance.
4. Explain with failure taxonomy.
