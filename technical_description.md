# Technical Description

This chapter explains what the system does, why it was built this way, and how the design fits the code that was finally used. The project is not a polished end-user product. It is a research prototype built to compare prompting, QLoRA, and a small ReAct loop under limited hardware.

---

## 3.1 Requirements Refinement and Design Constraints

The starting goal was simple: take a business question in English and turn it into SQL that can run on a real database. To make that manageable, the project was narrowed by five main constraints.

First, the system had to use open-source models rather than proprietary APIs. This made the workflow easier to inspect and easier to reproduce.

Second, the project had to stay within one schema, ClassicModels. That kept the work focused on the comparison between prompting, fine-tuning, and repair, instead of turning it into a much larger cross-domain study.

Third, the system had to fit limited hardware. The work was designed for a Colab-style GPU with about 24 GB of VRAM. This pushed the design toward short prompts, a small few-shot setting, QLoRA instead of full fine-tuning, and a bounded ReAct loop.

Fourth, model-generated SQL had to be treated as untrusted input. The system therefore needed a safe execution path and a clear way to record failures.

Fifth, the evaluation had to be systematic. The project uses a fixed experiment grid, fixed metrics, and a final evidence workflow so that reported results come from a clear set of official runs.

These constraints also explain what the project does not try to do. It does not attempt a rich user interface, chart generation, natural-language answers, or cross-domain database support. Those ideas are useful, but they are outside the main research question.

---

## 3.2 Architecture and High-Level Design

The project started mainly in notebooks. That was useful at first, but repeated logic became hard to manage. The final version moves reusable code into the `nl2sql/` package and keeps notebooks as thin wrappers.

The package is split into four layers:

```text
nl2sql/
├── core/        prompt building, schema handling, validation, and query execution
├── agent/       the ReAct repair loop
├── evaluation/  scoring, fixed runs, and final analysis
└── infra/       model loading, database access, and notebook helpers
```

This structure was chosen for a practical reason: the same logic had to be reused across several conditions. Keeping that logic in one place reduces duplication and makes the official run path easier to explain.

The main alternative was to keep the project mostly notebook-based. That would have been quicker in the short term, but it would have made the final execution path harder to defend. Using modules gives one clearer source of truth.

---

## 3.3 Core NL-to-SQL Pipeline

The baseline pipeline is simple by design. For each question it reads the schema, turns it into short prompt text, builds the prompt, asks the model for SQL, sends the output through a guarded read-only runner, and then scores the result.

### Schema representation

The model does not receive full database DDL. Instead, the system builds a short schema summary in the form `table(col1, col2, ...)`. This is easier to fit into the prompt and easier for the model to use.

The summary is ordered on purpose. Primary keys and name-like columns are placed first because they are often useful in joins and filters. Very wide tables are cut down so the prompt stays manageable.

### Prompting technique

The prompt uses one fixed system message with four rules:

- output only SQL
- output exactly one statement starting with `SELECT`
- use only tables and columns from the schema
- use `ORDER BY` and `LIMIT` only when the question asks for ranking

Each rule was added for a practical reason. The first rule reduces explanation text around the query. The second reduces multi-statement outputs. The third reduces made-up table names. The fourth reduces unnecessary ranking clauses.

The schema is shown before any exemplars and before the final question. This keeps the model grounded in the database it is supposed to query.

### Why `k=3` was used

The comparison uses only two shot settings: `k=0` and `k=3`. `k=0` gives the zero-shot baseline. `k=3` gives a small few-shot condition that is still easy to run and explain.

More values were possible, but they would have created more runs without helping the main comparison enough to justify the extra cost. Exemplars are sampled with a fixed seed, and the current test item is removed from the pool to avoid leakage.

`k=0` is deterministic in this setup because no exemplars are sampled. For the final method, that means one canonical `k=0` run is enough. `k=3` is stochastic because the exemplar sample changes with the seed, so the final method keeps multiple `k=3` runs for the main few-shot comparison.

### Generation and scoring path

Generation uses `do_sample=False` so the decoding is repeatable. The main baseline scorer also uses `max_new_tokens=128`, which is enough for typical ClassicModels queries without letting outputs become unnecessarily long.

The baseline path is kept close to the raw model output. It does not run the full schema-aware validation step before scoring. Instead, the generated text is sent straight to the guarded read-only runner. That runner blocks dangerous SQL and records failures. The stricter validate-and-repair path belongs to ReAct, not the baseline.

This separation was chosen because it keeps the baseline easier to interpret. If too much extra cleaning is added before scoring, it becomes harder to tell whether success came from the model or from extra reliability logic.

---

## 3.4 Fine-Tuning Design

QLoRA is the fine-tuning condition. In simple terms, it is a memory-saving way to adapt the model. That mattered because full fine-tuning of 7B-8B models was not realistic on the available hardware.

The training data uses the same chat style as the evaluation pipeline. Before training, the dataset is checked for leakage, duplicates, non-`SELECT` queries, and basic executability. These checks reduce the risk of blaming the model for problems that actually come from the training data.

The QLoRA settings are kept fixed across both models. This keeps the comparison focused on the effect of adaptation rather than on heavy hyperparameter tuning.

---

## 3.5 ReAct Extension

The ReAct extension adds a small repair loop:

1. generate a candidate query
2. validate it
3. execute it
4. repair it if validation or execution fails

The aim is not to build a large agent platform. It is to test whether a limited amount of feedback helps.

The implementation is kept local rather than using an external agent framework. This makes the behaviour easier to inspect and easier to keep consistent with the rest of the project.

Several ReAct settings are fixed in code:

| Setting | Value | Reason |
|---|---:|---|
| `few_shot_k` | 3 | matches the main few-shot setting |
| `few_shot_seed` | 7 | keeps the prompt context fixed |
| `max_repairs` | 2 | allows a small repair budget without long loops |
| `max_steps` | 8 | prevents the loop from running indefinitely |
| `max_new_tokens` | 256 | allows slightly longer repair outputs than baseline generation |
| `do_sample` | `False` | keeps behaviour deterministic |

The repair prompt is zero-shot. This is because repair is a different task from generation. Generation maps a question to SQL. Repair maps broken SQL plus an error message to corrected SQL. Reusing the generation exemplars here could confuse the model.

In the official dissertation runs, ReAct is treated as one fixed configuration with `few_shot_k=3` and `few_shot_seed=7`. It is not part of the full `k x seed` baseline/QLoRA grid.

ReAct was kept as an extension rather than the default system because it adds latency and complexity.

---

## 3.6 Evaluation, Testing, and Analysis Workflow

The evaluation is kept fixed rather than notebook-driven. This makes the run policy easier to rerun and easier to defend.

### Fixed run policy

The main baseline and QLoRA grid is:

| Setting | Value |
|---|---|
| `k` values | `[0, 3]` |
| `k=0` seeds | `[7]` |
| `k=3` seeds | target `[7, 17, 27, 37, 47]` |
| `max_new_tokens` | `128` |
| TS enabled for | `k=3` only |
| TS databases | `10` perturbed databases |
| TS row limit | `500` |

This grid keeps the main comparison small but consistent. `k=0` gives the zero-shot baseline, but that condition is deterministic because no exemplars are sampled, so repeated seeds are not informative. `k=3` gives the few-shot comparison and uses multiple seeds because exemplar sampling varies by seed.

The project originally used three `k=3` seeds to control Colab and database-hosting cost. The later target for the stochastic few-shot setting was five seeds because that gives a stronger run-level comparison while still staying manageable.

Test Suite Accuracy is turned on only for `k=3`. It is more expensive to run, and it matters most for the main few-shot condition.

### Metric design

The main grid always records three core metrics: `VA`, `EM`, and `EX`. A fourth metric, `TS`, is added only for the selected `k=3` runs.

The metric meanings are:

- `VA`: whether the SQL runs without error
- `EM`: whether the SQL text matches the gold query after normalisation
- `EX`: whether the returned rows match the gold query result
- `TS`: whether the prediction still matches under perturbed databases

`EX` is the main semantic metric because it best reflects whether the answer is correct.

### Testing and validation practice

Testing in this project is mainly workflow-based rather than unit-test heavy. The main risks come from data quality, prompt behaviour, SQL execution, and experiment configuration.

The main checks are:

- training-set validation before QLoRA training
- schema validation in the ReAct repair path
- SQL safety checks before queries reach the database
- fixed experiment scripts that always write JSON results
- a final analysis script that rebuilds the CSV evidence tables from a manual final pack

### Final evidence workflow

The official analysis path is:

1. run the fixed experiment scripts
2. copy the chosen official JSON files into `results/final_pack/`
3. run `python scripts/build_final_analysis.py`
4. use the generated CSV files for reporting

This manual `final_pack` step was kept on purpose. Automatic discovery would be more convenient, but it would also be harder to audit. The manual pack makes it clear which runs support the final claims.

The notebooks are still useful for inspection and rerun support, but they are not the official evidence path.

### Significance testing

The significance testing is based on per-seed EX rates rather than raw per-item binary outcomes. That means each `k=3` run contributes one EX value such as `0.415` or `0.557`.

Formal testing is limited to the two main `k=3` baseline-versus-QLoRA comparisons in the final analysis, one for Llama and one for Qwen. ReAct is not included in those tests because it is not run as the same repeated seed grid.

`k=0` results are reported descriptively only because that setting is deterministic in this project.

To keep the method easy to explain, the same two-sided Mann-Whitney U test is used for both formal `k=3` comparisons.

This is a better fit for the current experiment design than treating per-item `0/1` outcomes as the main unit of comparison.

---

## 3.7 Non-Functional Considerations

### Security

Model output is treated as untrusted input, so destructive SQL is blocked before execution.

### Reliability and reproducibility

Results are written to JSON with configuration metadata such as `k`, seed, and timestamp. This makes reruns and later checks easier.

### Performance

Compact schema text, deterministic decoding, row limits, and selective TS scoring help keep the workload within the hardware budget.

### Maintainability and usability

Moving shared logic into modules reduces duplication and makes the run path easier to follow. The notebooks are still useful as readable wrappers, but they are no longer the source of truth.

---

## 3.8 Chapter Summary

This chapter described a system that is designed more for fair evaluation than for product polish. The key decisions are small and practical: short schema text, fixed prompt rules, `k=3` few-shot prompting, QLoRA for memory-saving adaptation, a bounded ReAct loop, and a manual final evidence pack.

These choices are easier to explain, easier to rerun, and easier to defend against the code.
