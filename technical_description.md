# Technical Description

---

## 3.1 System Architecture

The system is structured as a four-layer Python library, housed under the `nl2sql/` package. Rather than writing evaluation logic and model calls in a single notebook, the codebase separates concerns across four distinct layers: `core/`, `agent/`, `evaluation/`, and `infra/`. This layering was a deliberate design decision made early in development, motivated by the need to test multiple conditions — different models, different shot counts, different scoring policies — without duplicating logic or risking inconsistency across runs.

```
nl2sql/
├── core/          # reusable pipeline building blocks
│   ├── schema.py  # schema introspection and serialisation
│   ├── prompting.py      # few-shot message construction
│   ├── llm.py            # model generation and SQL extraction
│   ├── postprocess.py    # SQL normalisation and cleanup
│   ├── sql_guardrails.py # raw output sanitisation
│   ├── validation.py     # pre-execution schema validation
│   └── query_runner.py   # safe SQL execution and audit logging
├── agent/         # ReAct orchestration layer
│   ├── agent_tools.py    # shared runtime context
│   ├── prompts.py        # system prompts for generation and repair
│   └── react_pipeline.py # generate → validate → execute → repair loop
├── evaluation/
│   ├── eval.py                 # evaluation runner, metrics, and result serialisation
│   ├── grid_runner.py          # fixed baseline/QLoRA grid execution
│   ├── final_pack.py           # manual final-pack loader and table builder
│   └── simple_stats.py         # VA/EX/TS summaries and EX-only Wilcoxon/BH-FDR tests
└── infra/         # notebook-facing setup and orchestration helpers
    ├── db.py                # Cloud SQL connection and TS engine helpers
    ├── notebook_utils.py    # auth, paths, and dataset loading
    ├── model_loading.py     # quantised model and adapter loading
    ├── training_set.py      # training-set validation helpers
    └── experiment_helpers.py# ReAct setup and QLoRA training helpers
```

The core layer holds the reusable NL-to-SQL building blocks: prompting, generation, validation, output cleaning, and safe query execution. The agent layer builds on core to implement the ReAct extension. The evaluation layer handles scoring, the fixed grid runner, manual evidence loading, and the final statistical comparison. The infra layer contains notebook-facing code such as DB/auth setup, model-loading wrappers, and the remaining reusable training/ReAct helpers. This separation keeps the experimental notebooks thin while still exposing a small number of named entry points for each workflow.

Notebooks still exist as readable walkthroughs, but the final dissertation rerun path now uses fixed scripts rather than notebook cells as the official execution surface. The baseline and QLoRA scripts call `run_eval_grid()` directly, while the ReAct scripts still use the small `run_react_notebook_eval()` helper because that path writes a distinct report format. The official story is therefore short: run the fixed scripts, manually select the cited JSON files, and build the final CSV tables from that manual pack.

---

## 3.2 Database Access and the Safety-First Execution Path

All database access in the system flows through a single path defined in `query_runner.py`. This was a non-negotiable design requirement: model-generated SQL, by definition, cannot be trusted. The system had to be capable of rejecting destructive statements before they reached the database.

The safety check is implemented as a simple token-scan blocklist:

```python
DEFAULT_FORBIDDEN_TOKENS = [
    "drop ", "delete ", "truncate ", "alter ", "create ",
    "update ", "insert ", "grant ", "revoke ",
]

def check_sql_safety(sql: str, ...) -> None:
    lowered = (sql or "").strip().lower()
    for token in tokens:
        if token in lowered:
            raise ValueError(f"Destructive SQL token detected: {token.strip()}")
```

The trailing space in each token (e.g. `"drop "`) is intentional: it prevents false positives on column names such as `drop_date` or `update_count` while still catching all relevant DML statements. A more sophisticated approach — parsing the SQL AST and checking statement types — was considered but rejected on the grounds of complexity. SQL parsing libraries like `sqlglot` introduce dependency overhead, and AST-based checks can fail on malformed SQL that a token scan would still correctly reject. For a read-only research environment where the only goal is preventing accidental writes, the token approach is sufficient.

The blocklist is defined as a module-level constant in `query_runner.py` and imported by `sql_guardrails.py` and `eval.py`. Having a single authoritative list means that adding a new forbidden token propagates to all callers automatically — there is no risk of a caller maintaining its own out-of-sync copy.

Connection management uses Python's `contextlib.contextmanager` (Python Software Foundation, 2024) to ensure connections are always closed, even when execution fails:

```python
@contextmanager
def safe_connection(engine: Engine) -> Iterator[sqlalchemy.engine.Connection]:
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
```

The `finally` block runs unconditionally — whether the query succeeds, raises a SQLAlchemy exception, or is interrupted by a keyboard interrupt in the notebook. This prevents connection leaks during long evaluation runs where dozens of queries are executed in sequence.

Connecting to Google Cloud SQL from a Colab environment required using the Cloud SQL Python Connector library rather than a standard SQLAlchemy connection URL. Standard URLs encode credentials in plaintext and do not support IAM-based authentication. The connector wraps the connection inside a `creator` callable that SQLAlchemy calls on demand:

```python
def create_engine_with_connector(*, instance_connection_name, user, password, db_name):
    connector = Connector()
    def getconn():
        return connector.connect(instance_connection_name, "pymysql", ...)
    engine = sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)
    return engine, connector
```

Both the engine and the connector are returned, since the connector holds its own background thread and must be explicitly closed at the end of the session. Returning both from the factory ensures callers cannot create a connector and then lose the reference needed to close it.

The `QueryRunner` class wraps this infrastructure into a stateful executor that maintains an audit history of every query attempted:

```python
class QueryRunner:
    """..."""
    def run(self, sql: str, ...) -> QueryResult:
        timestamp = now_utc_iso()
        try:
            self._safety_check(sql)
            with safe_connection(self.engine) as conn:
                result = conn.execute(sqlalchemy.text(sql), ...)
                rows = result.fetchmany(self.max_rows + 1)
                truncated = len(rows) > self.max_rows
                ...
        except Exception as e:
            out = QueryResult(..., success=False, error=str(e))
        self.history.append(out)
        return out
```

Every call to `run()` produces an immutable `QueryResult` record regardless of success or failure. The truncation check (`fetchmany(max_rows + 1)`) is a deliberate trick: fetching one extra row is enough to detect that the result set exceeds the limit, without materialising the entire result in memory. This bounds memory usage during evaluation without needing a separate `COUNT(*)` query.

---

## 3.3 Immutability for Evaluation Integrity

A recurring pattern across the codebase is the use of Python's frozen dataclasses (Python Software Foundation, 2024) for objects whose values must not change after creation. Three key types use this pattern.

`QueryResult` is the record of a single query execution. Once a query has run and been scored, its `success`, `rowcount`, and `error` fields must remain fixed. Making the dataclass frozen means that any attempt to mutate a result after the fact — for example, retroactively marking a failed query as successful — raises a `FrozenInstanceError` at runtime rather than silently corrupting the evaluation.

`EvalItem` is the scored record for one test item. It stores the raw model output, the processed prediction, and all four metric scores. Freezing it ensures that the object written to the results JSON is exactly the object that was scored — there is no code path that could update `ex=True` after writing to disk.

`EvalRunConfig` bundles the small set of settings that still belong inside the evaluation layer itself. Before this dataclass was introduced, `eval_run()` accepted many separate keyword arguments, and the call sites were difficult to read and prone to argument-order errors. Grouping the remaining internal settings into a frozen config object keeps the lower-level evaluation code tidy while the fixed scripts and runnable notebook mirrors stay simple.

```python
@dataclass(frozen=True)
class EvalRunConfig:
    max_new_tokens: int = 128
    max_rows: int = 50
    max_compare_rows: int = 10000
    avoid_exemplar_leakage: bool = True
    ts_suite_db_names: Optional[list[str]] = None
    ts_make_engine_fn: Optional[Callable[[str], Engine]] = None
    ts_max_rows: int = 500
    ...
```

The use of frozen dataclasses over plain dictionaries was a deliberate choice. Dictionaries allow arbitrary key mutation and provide no type information at the call site. A frozen dataclass provides IDE autocompletion, static type checking, and runtime immutability guarantees simultaneously, with no additional dependencies.

---

## 3.4 Reproducibility Engineering

Scientific validity in machine learning evaluation requires that results be exactly reproducible: running the same condition twice must produce the same scores. Two mechanisms ensure this in the system.

The first is a seeded per-run random number generator for exemplar sampling:

```python
rng = random.Random(seed)
...
exemplars = rng.sample(pool, k)
```

Python's `random.Random` class (Python Software Foundation, 2024) creates an independent RNG instance seeded from the integer `seed`. This isolates the exemplar sampling from any other random operations happening in the notebook environment — importing a library, shuffling a different list, or any other call to the global `random` module cannot affect which examples are drawn for a given `(k, seed)` condition. Every evaluation run records its seed in the JSON output, so results files are self-documenting.

The second mechanism is leakage prevention. When sampling k exemplars for test item i, item i must be excluded from the candidate pool:

```python
return [
    ex for ex in pool
    if not (ex.get("nlq") == nlq and ex.get("sql") == gold_sql)
]
```

Without this filter, the model could receive the exact question it is being tested on as one of its worked examples, trivially recalling the gold SQL rather than generating it from reasoning. Both the natural language question and the gold SQL must match for an item to be excluded: matching on the question text alone would incorrectly remove paraphrases that happen to share wording but map to different queries.

These two mechanisms together ensure that reported scores reflect genuine model capability rather than sampling artefacts or data contamination.

---

## 3.5 Evaluation Metric Design

The evaluation pipeline scores each prediction on four independent metrics, chosen to measure different aspects of correctness at increasing levels of rigour.

**Validity Accuracy (VA)** is the most lenient metric: does the predicted SQL execute without error? This captures both syntactic validity (the SQL can be parsed) and schema validity (the referenced tables and columns exist). It does not say anything about whether the result is correct.

**Exact Match (EM)** compares the normalised string representation of the predicted SQL against the gold SQL. Normalisation strips trailing semicolons, collapses whitespace, and lowercases the entire string. EM is strict — a semantically equivalent query with a different column order or alias name will fail. This metric is most useful as a lower bound and for identifying cases where the model has reproduced the gold phrasing exactly.

**Execution Accuracy (EX)** is the primary metric for semantic correctness. Both the predicted and gold SQL are executed against the live database; the result row sets are compared using Python's `collections.Counter`:

```python
return Counter(pred_rows) == Counter(gold_rows), None, None
```

`Counter` provides bag equality — it counts occurrences of each row but ignores order. This matches the definition used by the Spider benchmark (Yu et al., 2018), where SQL execution accuracy is defined as equality of unordered result multisets. A prediction can pass EX with completely different SQL syntax from the gold, as long as the rows it returns are identical. This is the most reliable indicator of real-world correctness.

**Test Suite Accuracy (TS)** extends EX by running both queries against N perturbed variants of the database, each with different data values. A query that hard-codes a specific value (`WHERE country = 'France'`) would pass EX on the original database but fail on a perturbed variant where the French customers have been replaced with German ones. TS therefore measures generalisation rather than point-in-time correctness, following the methodology of Zhong et al. (2020). In the implementation used here, a prediction scores TS=1 only if it matches the gold on every checked perturbed database; a single mismatch is sufficient to fail, and if the gold query itself fails on any replica the item is scored TS=0 rather than skipped.

One subtle implementation challenge arose in the TS scorer: floating-point equality. Two identical `NULL` values in different query results compare as equal in Python. But `float('nan') != float('nan')` by IEEE 754 definition, meaning two rows that should match can appear different if either contains a `NaN`. This was fixed by normalising all `NaN` values to the sentinel string `"NaN"` before comparison, and rounding floats to 10 decimal places to suppress floating-point drift:

```python
def _coerce_cell(x):
    if isinstance(x, float):
        if math.isnan(x): return "NaN"
        return round(x, 10)
    return x
```

---

## 3.6 The Prompt Engineering Layer

The system prompt for the baseline evaluation was developed iteratively by identifying failure modes in early model outputs and adding rules to address each one. The comments in `prompting.py` document this reasoning explicitly:

- *"Output only SQL"* — without this instruction, both Llama and Qwen frequently prefaced their output with explanatory prose ("Here is the SQL query that answers your question:"), which broke downstream SQL extraction.
- *"Exactly one statement starting with SELECT"* — some model configurations emitted two queries separated by a comment, or emitted a `WITH` clause followed by a `SELECT`.
- *"Use only tables and columns in the provided schema"* — models consistently hallucinated plausible-sounding table names (`customer_orders`, `product_details`) that did not exist in the ClassicModels schema.
- *"ORDER BY and LIMIT only when the question asks for ranking"* — this is an output-cleanliness rule rather than the main source of semantic correctness. Because EX uses bag equality, a spurious `ORDER BY` often does not change the EX score by itself. In the final simplified rerun path, the primary evaluation keeps the raw model output rather than relying on a second cleanup profile, so this rule is best understood as prompt guidance rather than a downstream rescue step.

Schema information is provided to the model before the few-shot examples, rather than after:

```python
msgs = [
    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
    {"role": "user",   "content": "Schema Details:\n" + schema},
    # ... exemplars ...
    {"role": "user",   "content": f"Natural Language Question: {nlq}"},
]
```

This ordering is intentional. When table names appear in context before the model reads the worked examples, the examples reinforce schema-grounded generation. Placing the schema after the examples would mean the model sees the correct SQL syntax before it has read the schema it should use — an inconsistency that, in early experiments, increased the frequency of hallucinated table names.

---

## 3.7 The ReAct Pipeline: Design Decisions

The ReAct pipeline (Yao et al., 2023) adds a feedback loop on top of the baseline generation path: generate a candidate, validate it, execute it, and repair it if either step fails. The loop is controlled by a shared state object rather than local variables or nested closures:

```python
@dataclass
class _ReactState:
    step: int = 0
    trace: list[dict] = field(default_factory=list)
    current_sql: str | None = None
    repairs_used: int = 0
```

Explicit state threading was chosen over using closures or `nonlocal` variable rebinding for clarity. In Python, `nonlocal` can be used to modify a variable in an enclosing function's scope, but it makes the data flow implicit and difficult to inspect during debugging. An explicit `_ReactState` object can be printed, logged, or inspected at any point in the loop without additional tooling.

The repair budget (`repairs_used`) is shared across both validation failures and execution failures. This was a deliberate trade-off: with `max_repairs=2`, the loop can afford one validation repair and one execution-guided repair before stopping. Setting separate budgets per failure type would require two counters and more complex stopping logic for minimal benefit at the scale of a dissertation experiment.

Repair prompts are intentionally zero-shot — they do not include the few-shot NLQ→SQL examples used during generation:

```python
messages = [
    {"role": "system", "content": SQL_REPAIR_SYSTEM_PROMPT},
    {"role": "user",   "content": "Schema Details:\n" + schema_text},
    {"role": "user",   "content": repair_prompt},  # bad SQL + error + hint
]
```

The reason is format mismatch: the generation exemplars are (question, SQL) pairs, which teach the model to translate natural language to SQL. Repair is a different task — (bad SQL, error message) to (fixed SQL) — and providing generation exemplars in a repair context would present the model with examples in the wrong format, likely increasing confusion. In the absence of a dedicated repair corpus, zero-shot is the correct choice, consistent with the DIN-SQL self-correction approach (Pourreza and Rafiei, 2023).

Every step in the loop appends to the `trace` list, recording the action taken, the SQL at that point, and any error or validation result. This trace is serialised into the results JSON alongside the final prediction, providing a full audit trail of the agent's decision path for every test item — a requirement for the ablation analysis in Chapter 4.

---

## 3.8 Non-Functional Considerations

**Security.** The primary security concern is SQL injection: a model-generated query that contains a destructive statement could corrupt the live ClassicModels database. This is addressed at two independent layers. First, `check_sql_safety()` rejects any query containing a DML token before it reaches the database connection. Second, the Cloud SQL Connector uses IAM-based authentication; the database user credentials are notebook-local variables set per session and are never written to files or committed to version control. These two layers together mean that even if the DML blocklist missed a token, the database user's permissions would need to include write access for damage to occur.

**Performance.** Model inference is the dominant cost — each of the 200 test items requires one forward pass through a 7–8 billion parameter model on a single GPU. Database query time is negligible by comparison, but two measures bound the worst-case overhead. Row limits cap result sets at 50 rows for evaluation queries and 10,000 rows for execution-accuracy comparison. Schema text is cached in `AgentContext` after the first retrieval: the system checks a text cache, then a structured dict cache, before making a live database call. In practice this means the schema is queried from the database at most once per evaluation session.

**Reliability.** Frozen dataclasses prevent result mutation after scoring. All results are written to JSON immediately after each run with a UTC timestamp, so a notebook crash partway through a session does not lose completed results — each JSON file is a complete, self-contained record for that condition. The `QueryResult.history` list on `QueryRunner` provides a per-session audit of every attempted query, including failures, which was used during development to diagnose systematic extraction failures without re-running the full evaluation.

**Reproducibility.** Beyond the seeded RNG described in Section 3.4, each results JSON records the configuration under which it was produced: model ID, k, seed, limit, and a UTC timestamp. In the final dissertation workflow, reproducibility is enforced by a manual evidence pack rather than by recursively scanning the entire `results/` tree. After the fixed rerun scripts have produced their raw JSON files, only the selected official files are copied into `results/final_pack/` using canonical names such as `llama_base_k3_seed17.json`. The script `scripts/build_final_analysis.py` then reads only that folder and writes four CSV outputs under `results/final_analysis/`: a manifest, a flat per-item table, a per-condition summary, and an EX-only pairwise Wilcoxon test table with BH-FDR correction. This makes the cited evidence explicit and auditable: the dissertation claims are tied to a small folder of hand-selected JSON files rather than to a discovery layer that decides which archived runs to include.
