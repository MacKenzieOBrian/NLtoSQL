# Technical Description

This chapter explains the current delivered system and the main engineering decisions behind it. The project is best understood as an evaluation-first NL-to-SQL prototype rather than a polished end-user product. The main technical challenge was not building a user interface. It was building a reproducible and safe pipeline that could compare prompting, QLoRA, and a small ReAct-style extension on limited hardware.

---

## 3.1 Requirements Refinement and Design Constraints

The project started from a simple user need: translate a natural language business question into SQL that can run on a real database. In practice, that broad aim had to be narrowed into a workable computing project. The final system was therefore shaped by five design constraints.

First, the system had to work with open-source models rather than proprietary APIs. This made the project more reproducible and better aligned with the dissertation goal of evaluating what could be achieved in a transparent setting.

Second, the scope had to stay within one database schema. Supporting many schemas would have turned the project into a much larger schema-linking problem and would have made the evaluation harder to defend. The ClassicModels database gave a fixed target that was large enough to be non-trivial but still manageable.

Third, the system had to run under constrained hardware. This affected almost every later decision, including the use of compact schema text, a small few-shot setting, QLoRA instead of full fine-tuning, and a limited ReAct loop.

Fourth, safety and reproducibility were treated as core requirements rather than optional extras. Model-generated SQL cannot be trusted by default, so destructive statements had to be blocked. In the same way, experimental outputs had to be written in a form that could be rerun and audited later.

Fifth, the project had to be evaluated systematically. That meant using a fixed experiment grid, fixed metrics, and a final evidence workflow that did not depend on manually remembering which notebook cells had been run.

These constraints explain why some ideas were left out. A richer user interface, charts, natural-language answers, and cross-domain database support would all be interesting additions, but they were not central to the core technical question being tested here.

---

## 3.2 Architecture and High-Level Design

Early development was notebook-heavy. That was useful for experimentation, but it quickly became difficult to maintain because the same logic started appearing in several places. The final version of the project moves most reusable logic into a local Python package, `nl2sql/`, and leaves the notebooks as thin workflow wrappers and reporting aids.

The system is organised into four layers:

```text
nl2sql/
├── core/        reusable NL-to-SQL building blocks
├── agent/       ReAct-style repair loop
├── evaluation/  scoring, fixed runs, and final analysis
└── infra/       database, model-loading, and notebook helpers
```

This architecture was chosen for maintainability. The core layer contains functions that should behave the same no matter which experiment is being run, such as prompt construction, SQL extraction, validation, and query execution. The agent layer contains the optional ReAct logic. The evaluation layer owns metrics and experiment control. The infra layer contains practical support code for loading models, connecting to the database, and running notebooks.

The main alternative was to keep the project mainly inside notebooks. That was rejected because it made changes harder to track and increased the risk that different notebooks would quietly diverge. By moving the logic into modules, the same prompt builder, scorer, and safety checks are reused across all conditions.

This is also a more professional software engineering structure. It separates concerns, reduces duplication, and makes the authoritative execution path much clearer.

---

## 3.3 Core NL-to-SQL Pipeline

At a high level, the baseline system follows the same sequence for every question:

1. read the database schema
2. compress it into prompt text
3. build a chat prompt
4. generate SQL
5. execute it through a guarded read-only runner
6. score the result

This sounds simple, but each stage required design choices.

### Schema representation

The system does not give the model raw database DDL. Instead, it builds a compact schema summary in the form `table(col1, col2, ...)`. This choice was made for two reasons. First, raw DDL is long and noisy. Second, the task only needs the model to know which tables and columns exist, not every storage detail.

The summary is also ordered deliberately. Primary keys and identifier-like columns such as `name`, `id`, `code`, and `number` are placed first. This reflects the kinds of fields that are usually most useful in business questions and joins. Very wide tables are truncated so the schema summary stays usable inside the prompt budget.

The alternative was to include the entire schema in a more complete format. That was rejected because it would have consumed context length without clearly helping the model make better SQL decisions.

### Prompting technique

The main prompt builder uses a fixed system message with four simple rules:

- output only SQL
- output exactly one statement starting with `SELECT`
- use only tables and columns from the schema
- use `ORDER BY` and `LIMIT` only when the question asks for ranking

These rules were not chosen to sound sophisticated. They were chosen because they directly addressed common failure modes seen during development. Without the first rule, models often produced explanations around the SQL. Without the second, they sometimes produced more than one statement. Without the third, they hallucinated believable but wrong table names. The fourth rule helped reduce unnecessary ranking clauses.

The prompt order is also important. The schema is shown before the exemplars and before the final question. This keeps the whole interaction grounded in the same database context. If the examples came first, the model would see SQL patterns before it had seen the schema it was supposed to use.

### Why `k=3` was used

The dissertation compares `k=0` and `k=3`. Here, `k` is the number of few-shot exemplars added to the prompt.

`k=0` is the zero-shot baseline. It answers the question without worked examples and shows what the model can do from instructions and schema alone.

`k=3` was chosen as the few-shot condition because it is large enough to demonstrate the intended question-to-SQL format, but still small enough to keep the prompt compact and the comparison easy to explain. A larger sweep such as `k=1, 3, 5, 8` was possible, but it would have added many more runs, made the evaluation heavier, and shifted the project away from its main comparison questions. Using one clear few-shot setting made the experiment easier to defend.

The exemplars are sampled from the benchmark pool with a fixed random seed, and the current test item is removed from the exemplar pool to avoid leakage. This is important. If the exact question being tested appeared in the prompt examples, the system would no longer be measuring genuine generalisation.

### Generation settings

Generation is kept deterministic by setting `do_sample=False`. This was done for reproducibility. If sampling were enabled, run-to-run variation would come from decoding randomness as well as from exemplar choice. The baseline and QLoRA pipeline also uses `max_new_tokens=128`, which is enough for typical SQL queries while limiting long or rambling outputs.

This is a pragmatic choice rather than a theoretical one. The aim here is controlled evaluation, not creative text generation.

### Post-processing and validation

The main baseline scorer stays close to the raw model output. It does not run the full schema-aware `validate_sql()` step before scoring. Instead, it sends the generated text directly to a guarded read-only query runner, which blocks destructive statements and records execution failures as invalid predictions.

This was a deliberate trade-off. Adding more aggressive pre-execution cleaning to the baseline would have made the system look stronger, but it would also have made it harder to separate model quality from extra reliability logic. The stricter extract-and-validate path is used in the ReAct extension, where validation feedback is useful because the agent can repair a failed query and try again.

Overall, the baseline pipeline was designed to be simple, inspectable, and reproducible. That was more important for this project than building a heavily engineered prompt stack with many hidden heuristics.

---

## 3.4 Fine-Tuning Design

The second main condition in the dissertation is QLoRA fine-tuning. Full fine-tuning was considered unrealistic for the available hardware, so a parameter-efficient approach was chosen instead.

QLoRA was suitable because it allows the project to adapt the model using low-rank adapters while keeping the main model weights quantised. In practice, this made the experiments feasible in a Colab-style GPU environment without redesigning the whole project around distributed training.

The training data is formatted using the same chat-template style as the prompting pipeline. This keeps the fine-tuned model aligned with the way it will later be evaluated. Before training, the training set is checked for leakage, deduplication issues, non-`SELECT` queries, and basic executability. That validation step is important because poor training data would make it difficult to tell whether later failures came from the model or from the dataset.

The project therefore uses fine-tuning in a controlled way: not as an attempt to build the largest possible training pipeline, but as a fair comparison point against prompt-based generation.

---

## 3.5 ReAct Extension and Alternatives Considered

The project also includes an optional ReAct-style extension. This layer adds a simple feedback loop on top of the baseline pipeline:

1. generate a candidate query
2. validate it
3. execute it
4. repair it if validation or execution fails

The ReAct implementation is intentionally local and lightweight. An external agent framework could have been used, but that would have added more moving parts and reduced control over the exact experiment behaviour. A local loop was easier to inspect, easier to log, and easier to keep consistent with the rest of the codebase.

Several ReAct settings are fixed in code:

| Setting | Value | Reason |
|---|---:|---|
| `few_shot_k` | 3 | keep ReAct aligned with the main few-shot setting |
| `few_shot_seed` | 7 | fixed prompt context for reproducibility |
| `max_repairs` | 2 | enough for a small repair budget without long loops |
| `max_steps` | 8 | prevents the agent from running indefinitely |
| `max_new_tokens` | 256 | allows slightly longer repair outputs than baseline generation |
| `do_sample` | `False` | keeps behaviour deterministic |

The repair prompt is zero-shot. This is an important design choice. The normal generation exemplars are pairs of natural language questions and correct SQL. Repair is a different task: it takes broken SQL plus an error message and tries to fix it. Reusing the generation exemplars inside the repair stage would mix two different formats and likely confuse the model. For that reason, the repair stage gets schema information and the current error context, but not the original few-shot examples.

ReAct was kept as an extension rather than the default system because it adds latency and complexity. That trade-off was acceptable for research comparison, but not necessary for every run.

---

## 3.6 Evaluation, Testing, and Analysis Workflow

The project follows a fixed evaluation recipe rather than ad hoc notebook runs. This was a deliberate professional decision. A dissertation is easier to defend when the reader can see exactly which configurations were run and how the outputs were selected.

### Fixed run policy

The main baseline and QLoRA grid is:

| Setting | Value |
|---|---|
| `k` values | `[0, 3]` |
| seeds | `[7, 17, 27]` |
| `max_new_tokens` | `128` |
| TS enabled for | `k=3` only |
| TS databases | `10` perturbed databases |
| TS row limit | `500` |

This grid balances coverage with practicality. `k=0` provides the zero-shot baseline. `k=3` provides the few-shot comparison. Three seeds are enough to observe variation in the few-shot runs without creating an unnecessarily large experiment matrix.

Test Suite Accuracy is enabled only for `k=3`. This was done to focus the most expensive semantic test on the main few-shot condition, where it adds the most value. Running TS on every condition would increase cost and runtime without improving the main story of the chapter.

### Metric design

The system records four metrics:

- `VA`: whether the SQL runs without error
- `EM`: whether the SQL text matches the gold query after normalisation
- `EX`: whether the returned rows match the gold query result
- `TS`: whether the prediction still matches under perturbed databases

`EX` is the main semantic metric because it measures what matters most: whether the query returns the correct answer.

### Testing and validation practice

Testing in this project is mainly validation-driven rather than unit-test heavy. This reflects the nature of the system. Much of the risk comes from data quality, prompt behaviour, SQL execution, and experiment configuration rather than from small isolated algorithms.

The main testing procedures are:

- training-set validation before QLoRA training
- schema validation in the ReAct repair path
- SQL safety checks before queries reach the database
- fixed experiment scripts that always write JSON results
- a final analysis script that rebuilds the CSV evidence tables from a manual final pack

This is a systematic workflow. It makes failure points visible and reduces the chance of silently mixing unofficial runs into the dissertation evidence.

### Final evidence workflow

The official analysis path is:

1. run the fixed experiment scripts
2. copy the chosen official JSON files into `results/final_pack/`
3. run `python scripts/build_final_analysis.py`
4. use the generated CSV files for reporting

The manual `final_pack` stage is important. The alternative would have been automatic discovery of files from the whole `results/` tree. That was rejected because it is harder to audit. A small manual evidence pack makes it clear exactly which runs support the final claims.

### Significance testing

The final pairwise comparison step works on per-seed EX rates, not on the raw per-item binary values. For each condition, one EX rate is computed per seed. This produces values such as `0.415` or `0.557`, which are more suitable for comparing overall condition performance than large lists of `0/1` item outcomes.

The test choice then depends on the type of comparison:

- if one side is deterministic, as can happen for `k=0`, a one-sample t-test is used against that fixed rate
- if both sides vary by seed, normality is checked first
- if both look normal, Welch's t-test is used
- otherwise, Mann-Whitney U is used
- if both sides are deterministic constants, the comparison is reported descriptively only

This is a better fit for the current experiment design because it compares actual condition-level values rather than mostly binary differences.

---

## 3.7 Non-Functional Considerations

### Security

Security mattered because the system executes model-generated SQL on a live database. To reduce risk, destructive statements are blocked before execution using a token-based safety check, and database access is routed through a controlled query runner. In a dissertation context, this is an example of professional caution: the model is treated as untrusted input.

### Reliability and reproducibility

Reliability is supported in several ways. Results are stored in frozen data structures after scoring, saved to JSON immediately after runs, and labelled with configuration metadata such as `k`, seed, and timestamp. Deterministic decoding and fixed seeds further reduce ambiguity. This makes reruns and later checking much easier.

### Performance

Performance was constrained mainly by model inference, not by database time. Several decisions respond directly to that fact:

- compact schema summaries instead of raw DDL
- deterministic decoding instead of repeated sampling
- row limits during query execution and comparison
- QLoRA rather than full fine-tuning
- TS enabled only where it adds the most value

These are not arbitrary simplifications. They are design responses to the real hardware limits of the project.

### Maintainability and usability

Moving logic out of notebooks and into modules improved maintainability. Keeping the notebooks as readable wrappers still helps usability, because they show the workflow step by step, but the authoritative logic now lives in reusable files. This is a better balance than either extreme of notebook-only development or a fully abstracted codebase with no readable experiment entry points.

---

## 3.8 Chapter Summary

The final system was designed as a controlled and defensible NL-to-SQL research prototype. The key decisions were shaped by safety, reproducibility, hardware limits, and the need for a fair comparison between prompting, QLoRA, and a small ReAct extension. Choices such as compact schema text, `k=3` few-shot prompting, deterministic decoding, QLoRA training, and a manual final evidence pack were not isolated implementation details. They were deliberate responses to the project requirements and constraints.

For that reason, the technical contribution of the project is not only the code that generates SQL. It is also the engineering of a reliable evaluation workflow around that code.
