# Chapters 1-3 Code Alignment Audit

Audit target: [Chapter 1,2,3.docx](/Users/mackenzieobrian/Downloads/Chapter%201,2,3.docx)

Code source of truth:
- [eval.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py)
- [react_pipeline.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py)
- [grid_runner.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py)
- [simple_stats.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py)
- [prompting.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/prompting.py)
- [schema.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/schema.py)
- [training_set.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/infra/training_set.py)
- [validation.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/validation.py)
- [query_runner.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/query_runner.py)
- [build_final_analysis.py](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/build_final_analysis.py)
- [final_pack.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/final_pack.py)

Status labels:
- `accurate`: directly supported by the current code
- `soften`: broadly right but stronger than the code supports
- `rewrite`: materially inaccurate against the current implementation
- `not supported by code`: may still be true, but must be supported from workbook/logbook/literature rather than code

## Findings

### P1 inaccurate / hard to defend

1. **Objective 4 misstates the implemented evaluation contract**
- Draft claim: all conditions are evaluated with `VA`, `EX`, and `TS`; benchmark thresholds are stated; EX is compared “across conditions”.
- Why this conflicts with code:
  - the fixed grid records `VA`, `EM`, and `EX` for all runs, with `TS` enabled only for `k=3`: [grid_runner.py:23](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L23), [grid_runner.py:76](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L76), [eval.py:281](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L281)
  - formal significance is EX-only and limited to the fixed eight planned baseline/QLoRA comparisons: [simple_stats.py:186](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py#L186), [simple_stats.py:205](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py#L205)
  - the threshold targets are not encoded anywhere in the codebase.
- Replacement sentence:
  - `Objective 4: Evaluate the main conditions against the same 200-item test set using Valid SQL (VA), Exact Match (EM), and Execution Accuracy (EX), with Test-Suite Accuracy (TS) applied only to the selected k=3 runs. Compare per-seed EX rates only within the fixed planned baseline and QLoRA comparison family using one-sample t-tests, Welch's t-tests, or Mann-Whitney U tests where appropriate.`

2. **Chapter 2.2.1 overstates both where validation happens and what the validator checks**
- Draft claim: PICARD motivates a schema-aware validation step that checks model output before execution.
- Why this conflicts with code:
  - baseline evaluation does not call `validate_sql()` before execution; it sends raw model output directly to the guarded runner: [eval.py:235](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L235), [eval.py:243](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L243), [eval.py:273](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L273)
  - `validate_sql()` is used in the ReAct loop: [react_pipeline.py:229](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L229)
  - the validator only performs lightweight checks: cleaning, `SELECT *` rejection, and table-name checks; it does not validate columns: [validation.py:59](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/validation.py#L59), [validation.py:77](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/validation.py#L77)
- Replacement sentence:
  - `The same class of error motivates the lightweight validation and repair layer used in the ReAct path, where SQL is cleaned, checked for basic safety, and checked for known table references before execution.`

3. **Chapter 3.4 describes training-set checks as if they are part of the fixed rerun path**
- Draft claim: before training, the dataset is checked for leakage, duplicates, non-`SELECT` queries, and basic executability.
- Why this conflicts with code:
  - those checks exist in the validation helper module: [training_set.py:18](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/infra/training_set.py#L18), [training_set.py:56](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/infra/training_set.py#L56)
  - but the fixed QLoRA rerun script trains directly from the raw train file and does not call that validation helper: [run_qlora_llama.py:45](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_qlora_llama.py#L45), [run_qlora_llama.py:57](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_qlora_llama.py#L57)
- Replacement sentence:
  - `In the development workflow, the training set was validated for leakage, duplicates, non-SELECT rows, and basic executability before adapter training. The fixed rerun script assumes that cleaned dataset as input.`

### P2 overstated / needs softening

1. **Chapter 1.6 is too strong when it says all three strategies share the same evaluation metrics and can be attributed cleanly to method**
- Why this needs softening:
  - baseline and QLoRA share the fixed `k x seed` grid: [grid_runner.py:23](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L23)
  - ReAct is one fixed configuration, not the same repeated grid: [react_pipeline.py:22](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L22), [run_react_llama.py:33](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_react_llama.py#L33)
  - formal testing excludes ReAct: [simple_stats.py:186](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py#L186)
- Better wording:
  - `The three strategies share the same schema and test set, while the main repeated seeded comparison is applied to the baseline and QLoRA conditions. ReAct is reported as a fixed extension rather than part of the same inferential family.`

2. **“Read-only query runner” is slightly stronger than the actual safety mechanism**
- Why this needs softening:
  - the runner blocks destructive SQL tokens before execution, but it does not formally enforce `SELECT`-only execution: [query_runner.py:25](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/query_runner.py#L25), [query_runner.py:31](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/query_runner.py#L31), [query_runner.py:80](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/query_runner.py#L80)
- Better wording:
  - `guarded execution runner` or `guarded SQL runner that blocks destructive tokens before execution`

3. **The hardware statements are not code-backed facts**
- Draft claim: the system runs on a Colab-style GPU with about 24 GB VRAM and on consumer hardware.
- Why this is not supported by code:
  - the code reflects design choices consistent with constrained hardware, such as 4-bit NF4 loading and QLoRA: [model_loading.py](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/infra/model_loading.py), [run_qlora_llama.py:53](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_qlora_llama.py#L53)
  - but the exact `24 GB` figure is not encoded in the repo.
- Keep only if supported by workbook, Colab logs, or supervisor-approved methodology notes.

4. **Chapter 1 Objective 1 and Chapter 3 baseline description use “schema-aware” language that is acceptable for prompting, but not for validation**
- Why this matters:
  - schema-aware prompting is accurate because the prompt explicitly includes schema text before the question: [prompting.py:28](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/prompting.py#L28)
  - schema-aware validation is limited and ReAct-only, not a general baseline property: [validation.py:59](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/validation.py#L59), [react_pipeline.py:229](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L229)
- Better wording:
  - keep `schema-aware` for prompt construction
  - avoid using it to describe the baseline scoring path

### P3 wording cleanup

1. **Metric naming is inconsistent between Objective 4 and Chapter 3**
- Objective 4 omits `EM`, while Chapter 3 includes it.
- The code computes `EM` in every evaluated item: [eval.py:274](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L274)
- Make Objective 4 and the later metric section use the same metric list.

2. **Chapter 2.2.1 implies broader validation than the actual helper performs**
- Even after moving validation to ReAct, keep the wording narrow: the helper checks table references and some formatting constraints, not full schema correctness: [validation.py:77](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/validation.py#L77)

3. **Chapter 1 contribution language should stay as research positioning, not code fact**
- Claims such as “extends Ojuri et al. in three specific ways” are reasonable, but they are not code-verifiable facts.
- Keep them as contribution framing, not as implementation proof.

## Claim-to-Code Audit Table

| Chapter / section | Claim summary | Status | Code anchor | Audit note |
|---|---|---|---|---|
| 1.2 Scope and constraints | Open-source models, one-schema scope, systematic final evidence path | accurate | [run_baseline_llama.py:16](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_baseline_llama.py#L16), [build_final_analysis.py:20](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/build_final_analysis.py#L20) | Supported by the fixed scripts and final-pack analysis workflow. |
| 1.4 Objective 1 | Baseline uses schema-grounded prompting with `k=0` and `k=3` | accurate | [prompting.py:21](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/prompting.py#L21), [grid_runner.py:23](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L23) | Matches prompt construction and fixed grid constants. |
| 1.4 Objective 2 | QLoRA on a 200-sample training set | accurate | [run_qlora_llama.py:45](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_qlora_llama.py#L45) | The fixed script loads `classicmodels_train_200.jsonl`. |
| 1.4 Objective 3 | ReAct is a bounded generate-validate-execute-repair loop | accurate | [react_pipeline.py:292](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L292), [react_pipeline.py:31](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L31) | Matches the algorithm and fixed repair budget. |
| 1.4 Objective 4 | Universal `TS`, threshold targets, significance “across conditions” | rewrite | [grid_runner.py:25](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L25), [simple_stats.py:205](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py#L205) | This is the biggest mismatch in Chapters 1-3. |
| 1.6 Approach | All three strategies share the same evaluation metrics and differences can be attributed to method | soften | [grid_runner.py:23](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L23), [run_react_llama.py:46](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_react_llama.py#L46) | ReAct is not run as the same seeded comparison family. |
| 2.2.1 PICARD bridge | Validation checks model output before execution | rewrite | [eval.py:235](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L235), [react_pipeline.py:229](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L229) | Move this claim to ReAct, not baseline. |
| 2.4 ICL rationale | Small prompt-condition set (`k=0`, `k=3`) is used rather than an exhaustive sweep | accurate | [grid_runner.py:23](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L23) | This is directly reflected in the fixed grid. |
| 2.4.1 Repair prompt rationale | Repair is treated as a different task from generation | accurate | [react_pipeline.py:155](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L155) | The repair stage is explicitly zero-shot and separate from generation exemplars. |
| 3.3 Baseline pipeline | Schema summary -> prompt -> raw generation -> guarded execution -> scoring | accurate | [eval.py:221](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L221), [eval.py:273](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L273) | This section is one of the strongest aligned parts of the draft. |
| 3.3.1 Schema representation | Compact `table(col1, col2, ...)`, prioritized useful columns, truncates wide tables | accurate | [schema.py:50](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/schema.py#L50), [schema.py:61](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/schema.py#L61) | Good match to implementation. |
| 3.3.2 Prompt construction | Four fixed prompt rules, schema shown before exemplars and question | accurate | [prompting.py:10](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/prompting.py#L10), [prompting.py:28](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/core/prompting.py#L28) | Good match to implementation. |
| 3.3.3 Few-shot setting | `k=0` and `k=3`; fixed seed; current item removed to avoid leakage | accurate | [eval.py:153](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L153), [eval.py:176](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L176) | Good match to implementation. |
| 3.3.4 Generation and post-processing | Baseline does not use full schema-aware validation before scoring | accurate | [eval.py:235](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L235), [eval.py:273](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L273) | Good correction already present in Chapter 3. |
| 3.4 Fine-tuning design | Training data is checked before training | soften | [training_set.py:74](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/infra/training_set.py#L74), [run_qlora_llama.py:45](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_qlora_llama.py#L45) | True in workflow terms, not in the fixed rerun script itself. |
| 3.5 ReAct extension | Fixed configuration with `few_shot_k=3`, `few_shot_seed=7`, `max_repairs=2`, `max_steps=8`, `max_new_tokens=256` | accurate | [react_pipeline.py:30](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/agent/react_pipeline.py#L30), [run_react_llama.py:33](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/run_react_llama.py#L33) | Good match to implementation. |
| 3.6.1 Fixed run policy | `k=[0,3]`, seeds `[7,17,27]`, `TS` at `k=3` only, `max_new_tokens=128` | accurate | [grid_runner.py:23](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L23), [grid_runner.py:124](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/grid_runner.py#L124) | Good match to implementation. |
| 3.6.2 Metrics | `VA`, `EM`, `EX`, `TS` with `EX` as main semantic metric | accurate | [eval.py:281](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/eval.py#L281) | The code computes all four at item/run level, with `TS` conditional on config. |
| 3.6.3 Final evidence workflow | Manual `final_pack` -> `build_final_analysis.py` -> official CSVs | accurate | [build_final_analysis.py:20](/Users/mackenzieobrian/MacDoc/Dissertation/scripts/build_final_analysis.py#L20), [final_pack.py:78](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/final_pack.py#L78) | Good match to implementation. |
| 3.6.4 Hypothesis testing | Per-seed EX rates; deterministic-aware one-sample / Welch / Mann-Whitney; fixed eight planned comparisons only | accurate | [simple_stats.py:80](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py#L80), [simple_stats.py:186](/Users/mackenzieobrian/MacDoc/Dissertation/nl2sql/evaluation/simple_stats.py#L186) | Strong match to implementation. |

## Chapter Summary

### Well aligned already
- Chapter 3.3 prompt/schema/baseline path
- Chapter 3.5 fixed ReAct configuration and scope
- Chapter 3.6 fixed run grid, manual final-pack workflow, and EX-only hypothesis testing

### Must be rewritten before submission
- Chapter 1 Objective 4
- Chapter 2.2.1 PICARD-to-validation bridge

### Should be softened before submission
- Chapter 1.6 common-evaluation wording
- Chapter 3.4 training-set checks if you want the draft to reflect the fixed rerun scripts rather than the broader development workflow
- any wording that calls the runner strictly `read-only`

### Literature-supported but not code facts
- the exact `24 GB` hardware figure
- `consumer hardware` framing
- comparative contribution language such as “extends Ojuri et al. in three ways”
- threshold targets like `VA > 90%`, `EX > 80%`, `TS > 70%`
