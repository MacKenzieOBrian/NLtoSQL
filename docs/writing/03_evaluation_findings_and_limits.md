# Evaluation, Findings, and Limits

Evaluation is organized around executable and semantic behavior rather than formatting quality. EX is the primary claim metric, VA is a reliability signal, EM is diagnostic, and TS is optional when the test-suite databases are enabled.

The final statistical policy matches supervisor requirements and is implemented in `scripts/generate_research_comparison.py`:
- mean and median per run/metric
- Shapiro-Wilk normality checks
- paired t-tests on predefined run comparisons

Outputs are written to:
- `results/analysis/stats_mean_median_shapiro.csv`
- `results/analysis/stats_paired_ttests.csv`

Current claim discipline should be:
- report effect direction and size first (e.g., EX delta),
- then report significance outcome,
- then discuss practical interpretation and limits.

If optional reliability controls are enabled (constrained decoding, SQL guardrails, postprocess), report them as extension experiments in a separate subsection so the primary base-vs-QLoRA hypothesis remains method-pure.

Limitations are explicit. The benchmark is single-schema and MySQL-oriented. Results are sensitive to available seed coverage and model compute budget. This work does not claim proprietary parity or cross-domain generalization.

At the current repository state, historical results were intentionally cleared before final reruns. Final dissertation claims must be based on the refreshed run artifacts and regenerated stats files above.
