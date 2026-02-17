# Examiner Q&A

## Q1. What is the primary contribution?
A reproducible open-source NL->SQL evaluation pipeline that compares prompting, QLoRA, and ReAct-style infrastructure under constrained compute using semantic-first metrics and paired statistics.

## Q2. How does this relate to Ojuri?
It replicates the comparison discipline (agentic framing + execution-centric evaluation), not proprietary stack parity.

Reference: `REFERENCES.md#ref-ojuri2025-agents`

## Q3. Why are EX and TS prioritized over EM?
SQL strings can differ while meaning is equivalent. EX/TS directly test semantic behavior by execution, while EM is mainly diagnostic.

## Q4. What statistics make the claims defensible?
- 95% Wilson intervals for per-run binary rates.
- Paired deltas on identical questions.
- Exact McNemar p-values on discordant pairs.

## Q5. Why paired comparisons?
They control for question difficulty because each method is evaluated on the same NL questions.

## Q6. What does your ReAct loop do now?
Model-driven `Thought -> Action -> Observation` tool selection. `finish` is only accepted after successful `run_sql`. If the loop exhausts step/repair budget, it returns `no_prediction`.

## Q7. Why was that loop change important?
It removed controller-forced ordering and improved fidelity to ReAct reference behavior, so outcomes are less likely to be orchestration artifacts.

## Q8. What are your current empirical takeaways?
- Few-shot improves EX significantly for base and QLoRA snapshots.
- Current Llama QLoRA snapshot does not exceed base EX at `k=3`.
- ReAct provides strong traceability, but current overlap sample does not show significant EX improvement.

## Q9. What still fails most often?
Projection, invalid SQL, join-path decisions, and value-linking errors remain dominant categories.

## Q10. What do you explicitly not claim?
- No claim of universal SOTA.
- No claim of proprietary-system parity.
- No claim of broad cross-domain generalization from a single-schema setup.

## Key artifacts to show in viva/demo
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/summary.md`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/overall_metrics_wide.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/paired_deltas.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/results/analysis/failure_taxonomy.csv`
- `/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/react_pipeline.py`
