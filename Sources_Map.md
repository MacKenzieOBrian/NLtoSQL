# Sources Map
> Online documentation, GitHub issues, and forums mapped to key code patterns in the project.
> Use this to explain where each implementation decision comes from.

---

## HuggingFace — Model Generation

### `tokenizer.apply_chat_template()`
**File**: `nl2sql/core/llm.py`
**Source**: [HuggingFace — Chat Templating](https://huggingface.co/docs/transformers/en/chat_templating)
**What it covers**: How to format a list of `{"role", "content"}` message dicts into token IDs the model expects. The docs explain `add_generation_prompt=True` (adds the assistant turn-start token so the model knows to continue as assistant), `tokenize=True`, and `return_tensors="pt"`. This is the canonical HuggingFace pattern for chat models.

---

### `pad_token_id = tok.pad_token_id or tok.eos_token_id`
**File**: `nl2sql/core/llm.py`
**Source 1**: [Llama 3 model card discussion — pad_token_id warning](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/40)
**Source 2**: [HuggingFace Forums — Setting pad_token_id to eos_token_id for open-end generation](https://discuss.huggingface.co/t/setting-pad-token-id-to-eos-token-id-50256-for-open-end-generation/22247)
**What it covers**: Llama-3 and Qwen-2.5 both lack a dedicated pad token. Without setting `pad_token_id` explicitly in `model.generate()`, HuggingFace emits a warning for every call. The standard fix, documented in both the model card discussion thread and the forums, is to fall back to `eos_token_id`. The pattern used in the codebase (`getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)`) is the safe form that handles tokenizers where the attribute may not exist.

---

### `effective_temperature = float(temperature) if do_sample else 1.0`
**File**: `nl2sql/core/llm.py`
**Source**: [GitHub Issue #31839 — Greedy sampling gives a warning message](https://github.com/huggingface/transformers/issues/31839)
**What it covers**: From transformers v4.39, passing `temperature` with `do_sample=False` raises a `UserWarning` because temperature has no mathematical effect in greedy decoding. The issue thread confirms the workaround: pass `temperature=1.0` (the neutral value) when greedy decoding is selected, and only pass the configured temperature when `do_sample=True`. This suppresses hundreds of warnings across a 200-item evaluation run.

---

### `attention_mask = torch.ones_like(input_ids)`
**File**: `nl2sql/core/llm.py`
**Source**: [HuggingFace Forums — Clarification on the attention_mask](https://discuss.huggingface.co/t/clarification-on-the-attention-mask/1538)
**What it covers**: When inputs are not padded (single sequence, no batch padding), the correct attention mask is all-ones — every token should be attended to. `torch.ones_like(input_ids)` creates a tensor of the same shape and device as the input, filled with 1s. Without an explicit mask, HuggingFace warns that reliable results cannot be guaranteed.

---

### `class _StopOnSemicolon(StoppingCriteria)`
**File**: `nl2sql/core/llm.py`
**Source 1**: [HuggingFace Docs — Generation utilities / StoppingCriteria](https://huggingface.co/docs/transformers/internal/generation_utils)
**Source 2**: [HuggingFace Forums — Implementing StoppingCriteria for code-generating Transformers](https://discuss.huggingface.co/t/implementing-stoppingcriteria-for-code-generating-transformers/52922)
**Source 3**: [GitHub — stopping_criteria.py source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/stopping_criteria.py)
**What it covers**: `StoppingCriteria` is an abstract base class. Subclassing it and implementing `__call__(self, input_ids, scores)` → `bool` is the documented way to add custom stopping logic. The forum post specifically discusses the semicolon stopping pattern for code/SQL generation. The `StoppingCriteriaList` wrapper is required by `model.generate()`.

---

### `gen_ids = out[0][input_ids.shape[-1]:]`
**File**: `nl2sql/core/llm.py`
**Source**: [HuggingFace Docs — Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
**What it covers**: `model.generate()` returns the full sequence — input tokens followed by generated tokens. To decode only the model's new output, you slice off the input length. This is the standard pattern shown in HuggingFace generation examples.

---

## PyTorch

### `with torch.no_grad()`
**File**: `nl2sql/core/llm.py`
**Source**: [PyTorch Docs — torch.no_grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html)
**What it covers**: Disables gradient computation for the wrapped block. During inference, gradients are not needed and computing them wastes GPU memory. For a 7–8B parameter model in a 24GB Colab GPU environment, this is not optional — it prevents OOM errors during evaluation runs.

### `torch.ones_like(input_ids)`
**File**: `nl2sql/core/llm.py`
**Source**: [PyTorch Docs — torch.ones_like](https://pytorch.org/docs/stable/generated/torch.ones_like.html)
**What it covers**: Creates a tensor of ones with the same shape, dtype, and device as the input. Used to construct the attention mask without manually specifying shape or device.

---

## Google Cloud SQL

### `create_engine_with_connector()`
**File**: `nl2sql/infra/db.py`
**Source 1**: [GitHub — GoogleCloudPlatform/cloud-sql-python-connector](https://github.com/GoogleCloudPlatform/cloud-sql-python-connector)
**Source 2**: [Google Cloud Docs — Connect using Cloud SQL Python Connector with SQLAlchemy (MySQL)](https://cloud.google.com/sql/docs/mysql/samples/cloud-sql-mysql-sqlalchemy-connect-connector)
**What it covers**: The Cloud SQL Python Connector handles IAM authentication and TLS tunnelling to a Cloud SQL instance. The standard SQLAlchemy connection URL (`mysql+pymysql://user:password@host/db`) cannot be used because it does not go through the connector's auth layer. The Google Cloud docs show the exact `creator=` pattern: pass a callable that returns a raw `pymysql` connection, and SQLAlchemy calls it on demand for each connection pool slot.

---

## SQLAlchemy

### `@contextmanager` / `safe_connection()`
**File**: `nl2sql/infra/db.py`
**Source 1**: [Python Docs — contextlib.contextmanager](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager)
**Source 2**: [SQLAlchemy Docs — Working with Engines and Connections](https://docs.sqlalchemy.org/en/20/core/connections.html)
**What it covers**: The `@contextmanager` decorator turns a generator function into a context manager. The `try/finally` pattern guarantees that `conn.close()` runs even if the query raises an exception. The SQLAlchemy connections docs note that explicit connection management (rather than implicit pool checkout) is required when you need deterministic close-on-error behaviour.

### `result.fetchmany(self.max_rows + 1)`
**File**: `nl2sql/core/query_runner.py`
**Source**: [SQLAlchemy Docs — ORM API / fetchmany](https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html)
**What it covers**: `fetchmany(n)` returns at most n rows. Fetching `max_rows + 1` is a known truncation-detection idiom: if the result has exactly `max_rows + 1` rows, you know there are more, without loading the entire table. This avoids a second `COUNT(*)` query and prevents materialising arbitrarily large result sets into Colab RAM.

### `sqlalchemy.text(sql)`
**File**: `nl2sql/core/query_runner.py`, `nl2sql/evaluation/eval.py`
**Source**: [SQLAlchemy Docs — Using textual SQL](https://docs.sqlalchemy.org/en/20/core/sqlelement.html#sqlalchemy.sql.expression.text)
**What it covers**: Raw string SQL must be wrapped in `text()` to be accepted by `conn.execute()` in SQLAlchemy 2.0+. Passing a bare string was deprecated and removed. `text()` marks the string as trusted literal SQL, distinct from parameterised expressions.

---

## Python Standard Library

### `@dataclass(frozen=True)` — `EvalItem`, `QueryResult`, `EvalRunConfig`
**File**: `nl2sql/evaluation/eval.py`, `nl2sql/core/query_runner.py`
**Source 1**: [Python Docs — dataclasses](https://docs.python.org/3/library/dataclasses.html)
**Source 2**: [PEP 557 — Data Classes](https://peps.python.org/pep-0557/)
**What it covers**: `frozen=True` generates `__setattr__` and `__delattr__` methods that raise `FrozenInstanceError`, making the instance immutable after `__init__`. PEP 557 describes the design rationale: frozen instances behave like tuples in that they cannot be mutated, but they retain named field access and type annotations.

### `rng = random.Random(seed)`
**File**: `nl2sql/evaluation/eval.py`
**Source**: [Python Docs — random.Random](https://docs.python.org/3/library/random.html#random.Random)
**What it covers**: `random.Random` creates an independent RNG instance with its own internal state, isolated from the module-level `random` functions. The docs note that multiple independent RNG instances can be created, each seeded separately — essential when you need reproducible sampling in one part of a programme without affecting randomness elsewhere.

### `Counter(pred_rows) == Counter(gold_rows)`
**File**: `nl2sql/evaluation/eval.py`
**Source**: [Python Docs — collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter)
**What it covers**: `Counter` is a dict subclass that counts hashable elements, equivalent to a mathematical multiset. Two `Counter` objects are equal if every element has the same count in both — order-insensitive, which is what the Spider EX metric requires. The docs describe Counter as "similar to bags or multisets in other languages".

### `with warnings.catch_warnings():`
**File**: `nl2sql/evaluation/research_stats.py`
**Source**: [Python Docs — warnings.catch_warnings](https://docs.python.org/3/library/warnings.html#warnings.catch_warnings)
**What it covers**: `catch_warnings()` creates a temporary warnings filter scope. Here it is used with `warnings.simplefilter("ignore")` so repeated Shapiro-Wilk warnings do not flood notebook output during the comparison workflow.

### `math.isnan(x)` → sentinel `"NaN"`
**File**: `nl2sql/evaluation/eval.py`
**Source**: [Python Docs — math.isnan](https://docs.python.org/3/library/math.html#math.isnan)
**What it covers**: IEEE 754 defines `NaN != NaN`, so two result rows that both contain `NULL` (read as `float('nan')` by some MySQL drivers) will compare as unequal in a `Counter`, causing a false EX mismatch. `math.isnan()` detects this case; replacing `NaN` with a sentinel string makes equality work correctly.

### `from __future__ import annotations`
**File**: All modules
**Source**: [PEP 563 — Postponed Evaluation of Annotations](https://peps.python.org/pep-0563/)
**What it covers**: Makes all type annotations in the module lazy (evaluated as strings rather than at import time). This allows forward references and the `X | None` union syntax in Python versions below 3.10, which is important for Colab compatibility where the Python version may lag.

### `Path.write_text()` / `Path.read_text()` / `.rglob()`
**File**: `nl2sql/evaluation/eval.py`, `nl2sql/evaluation/research_runs.py`, `nl2sql/evaluation/research_comparison.py`
**Source**: [Python Docs — pathlib.Path](https://docs.python.org/3/library/pathlib.html)
**What it covers**: `pathlib` provides object-oriented filesystem paths. `write_text()` writes a string to a file in one call (no open/close). `rglob("*.json")` recursively finds files matching a pattern — used in the run-discovery helper to discover result JSON files across nested run directories.

### `re.compile(r"...", re.IGNORECASE)`
**File**: `nl2sql/core/sql_guardrails.py`, `nl2sql/core/postprocess.py`, `nl2sql/core/validation.py`
**Source**: [Python Docs — re — Regular expression operations](https://docs.python.org/3/library/re.html)
**What it covers**: Pre-compiling a regex with `re.compile()` caches the pattern object. In evaluation runs where the same pattern is applied to 200 queries, this avoids re-parsing the pattern on every call. `re.IGNORECASE` (alias `re.I`) makes the match case-insensitive without manually lowercasing the input.

---

## SciPy / Statistical Testing

### `scipy.stats.wilcoxon(diffs, zero_method='wilcox')`
**File**: `nl2sql/evaluation/research_stats.py`
**Source**: [SciPy Docs — scipy.stats.wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
**What it covers**: The Wilcoxon signed-rank test tests whether the distribution of paired differences is symmetric about zero — a non-parametric alternative to the paired t-test. The docs explain `zero_method='wilcox'` (the Wilcoxon 1945 original method, which discards zero differences). This is the primary test in the dissertation because binary 0/1 metrics violate the normality assumption required by the t-test.

### `scipy.stats.ttest_rel(a, b)`
**File**: `nl2sql/evaluation/research_stats.py`
**Source**: [SciPy Docs — scipy.stats.ttest_rel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)
**What it covers**: The paired t-test. Used as a corroborating test alongside Wilcoxon. At n≥600 paired observations, the Central Limit Theorem justifies its use even with binary metric distributions. The docs describe the test statistic and degrees of freedom (n-1) used for the confidence interval calculation.

### `scipy.stats.shapiro(diffs)`
**File**: `nl2sql/evaluation/research_stats.py`
**Source**: [SciPy Docs — scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
**What it covers**: Shapiro-Wilk normality test on the paired differences. Run on every comparison pair to document whether normality looks plausible before reading the paired t-test. The warnings context is used to keep repeated range-zero warnings out of the notebook output.

### `_bh_fdr_adjust(pvalues)`
**File**: `nl2sql/evaluation/research_stats.py`
**Source**: [Statsmodels Docs — multipletests](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
**What it covers**: Applies multiple comparison correction to a family of p-values. `method='fdr_bh'` is the Benjamini-Hochberg procedure, which controls the False Discovery Rate rather than the Family-Wise Error Rate. The project now uses a small local helper that follows the same adjustment rule rather than importing `statsmodels`.
