# Submission Traceability Note

This note keeps the final write-up narrow. Each claim below is either kept as written, simplified, or softened so it matches the current codebase.

| Area | Claim kept in submission | Status | Code anchor | Why this wording is defensible |
|---|---|---|---|---|
| Research scope | The project compares prompting, QLoRA, and a bounded ReAct extension under hardware constraints. | keep | `scripts/run_baseline_llama.py`, `scripts/run_qlora_llama.py`, `scripts/run_react_llama.py` | These are the fixed rerun entrypoints used for the dissertation conditions. |
| Baseline pipeline | Baseline flow is schema summary -> prompt -> raw generation -> guarded execution -> scoring. | simplify | `nl2sql/evaluation/eval.py:235`, `nl2sql/evaluation/eval.py:273` | The baseline scorer does not run the full schema-aware validation step before scoring. |
| Schema summary | The schema is passed as compact `table(col1, col2, ...)` text with useful columns shown early. | keep | `nl2sql/core/schema.py:50`, `nl2sql/core/schema.py:61` | This matches the implemented summarisation logic and is easy to explain. |
| Prompt rules | The prompt uses four fixed rules to reduce prose, multi-statement output, hallucinated schema use, and unnecessary ranking clauses. | keep | `nl2sql/core/prompting.py:10`, `nl2sql/core/prompting.py:28` | The rules are explicit in code and map to obvious failure modes. |
| Few-shot choice | `k=3` is presented as a manageable few-shot setting, not as an optimal value. | simplify | `nl2sql/evaluation/grid_runner.py:23`, `nl2sql/evaluation/eval.py:165` | The code fixes `k` to `[0, 3]` but does not prove `k=3` is best. |
| QLoRA rationale | QLoRA is described as a memory-saving way to adapt the model on limited hardware. | simplify | `nl2sql/infra/experiment_helpers.py:26`, `scripts/run_qlora_llama.py:45` | This matches the actual training setup without overexplaining PEFT theory. |
| ReAct scope | ReAct is one fixed repair configuration, not part of the full seeded baseline/QLoRA grid. | keep | `nl2sql/infra/experiment_helpers.py:91`, `nl2sql/agent/react_pipeline.py:231` | The code fixes the ReAct settings and uses validation inside that path only. |
| Main metrics | The main grid always reports `VA`, `EM`, and `EX`; `TS` is added only for selected `k=3` runs. | simplify | `nl2sql/evaluation/grid_runner.py:23`, `nl2sql/evaluation/grid_runner.py:25`, `nl2sql/evaluation/eval.py:265` | This avoids the older over-broad wording that made `TS` sound universal. |
| Significance scope | Formal significance is EX-only and limited to the fixed eight planned baseline/QLoRA comparisons. | keep | `nl2sql/evaluation/simple_stats.py:186`, `nl2sql/evaluation/simple_stats.py:205` | This matches the current statistical code and excludes ReAct from that family. |
| Evidence workflow | Official reporting comes only from `results/final_pack/` plus `python scripts/build_final_analysis.py`. | keep | `scripts/build_final_analysis.py:20`, `nl2sql/evaluation/final_pack.py:78` | This is the auditable evidence path used for final claims. |

## Submission surface

Keep as official:
- `nl2sql/`
- `scripts/`
- `README.txt`
- `technical_description.md`
- `results/final_pack/`
- `results/final_analysis/`

Keep as support material only:
- `notebooks/`
- `notebooks/01_demo.ipynb`
- `diagrams.md`

Removed or softened because they were harder to defend than the current code:
- wording that implied baseline pre-execution schema validation
- wording that implied `TS` was run for every condition
- wording that implied significance was searched across all possible condition pairings
- legacy notes that still described an obsolete statistics workflow
