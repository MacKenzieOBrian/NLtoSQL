# Evaluation, Findings, and Limits

Evaluation in this dissertation is organized around semantic correctness rather than formatting correctness. EX and TS are treated as primary claim metrics, VA is treated as a secondary reliability signal, and EM is treated as diagnostic context. This priority keeps the interpretation aligned with whether generated SQL actually answers the intended question.

The statistical policy is designed for paired binary outcomes. Rate uncertainty is reported with Wilson intervals, and method deltas are tested with exact McNemar on discordant pairs. This is implemented in `nl2sql/evaluation/research_stats.py` and applied in `scripts/generate_research_comparison.py`. The practical implication for writing is that claims should include direction and magnitude of change, plus confidence and significance, rather than reporting isolated percentages.

Current snapshot findings support a consistent few-shot benefit in this setup. Base-model EX improves meaningfully from `k=0` to `k=3`, and QLoRA also shows improvement from `k=0` to `k=3` under matched conditions. In the current Llama snapshot, QLoRA at `k=3` does not clearly surpass the corresponding base baseline at the same `k`, so this is treated as a mixed empirical result rather than framed as a universal adaptation win.

Agentic ReAct results are interpreted as infrastructure evidence unless semantic gains are shown with adequate overlap and significance. This keeps the dissertation from overclaiming based on trace quality alone. Traces are still valuable because they expose failure mechanisms and repair behavior, but they do not replace EX and TS evidence.

Limitations are explicit. The benchmark domain is single-schema and MySQL-oriented. Some runs are constrained by hardware and not all method-model-seed combinations are complete at equal depth. Error patterns still cluster in schema and value linking and in compositional SQL decisions, even when execution succeeds. These limitations are not peripheral notes; they define the valid boundary of the project claims.

For dissertation writing, this section should be used to enforce claim discipline. Every performance statement should tie to artifacts in `results/analysis/`, and every strong conclusion should be accompanied by the uncertainty and paired-significance context that supports it.
