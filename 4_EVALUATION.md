# Evaluation Engineering Log (Reformatted)

Each metric is presented in a four-part explanation format.

---

### Valid SQL (VA)

**Plain-language**  
VA checks whether the predicted SQL runs at all. It separates syntax/schema errors from semantic errors.

**Technical description**  
`QueryRunner.run` executes the SQL under a read-only guard and returns a success flag. VA is recorded directly from this flag.

**Code locations**  
`nl2sql/query_runner.py` (`QueryRunner.run`)  
`notebooks/03_agentic_eval.ipynb` (evaluation loop in cell `# 9) Full ReAct-style evaluation (VA/EX/EM/TS)`)

**Justification**  
Execution-based evaluation uses VA as a baseline validity check (Zhong et al., 2020). The trade-off is that VA can be high even when semantics are wrong.

---

### Execution Accuracy (EX)

**Plain-language**  
EX checks whether predicted SQL returns the same rows as the gold SQL on the base DB.

**Technical description**  
`execution_accuracy` runs both SQL queries and compares row multisets using a Counter. Column names are not required to match.

**Code locations**  
`nl2sql/eval.py` (`execution_accuracy`, `execute_fetch`)  
`notebooks/03_agentic_eval.ipynb` (evaluation loop)

**Justification**  
Zhong et al. (2020) show that EM is insufficient and motivate execution-based equivalence. The trade-off is that EX depends on DB state and can be fooled by accidental matches.

---

### Exact Match (EM)

**Plain-language**  
EM is a strict string match. It is retained only to diagnose formatting or postprocessing regressions.

**Technical description**  
Predicted and gold SQL are normalized (strip semicolon, lowercased) and compared directly.

**Code locations**  
`nl2sql/postprocess.py` (`normalize_sql`)  
`notebooks/03_agentic_eval.ipynb` (evaluation loop)

**Justification**  
EM was common in early benchmarks (Yu et al., 2018) but is not a semantic metric. The trade-off is low semantic validity, so EM is treated as diagnostic only.

---

### Test-Suite Accuracy (TS)

**Plain-language**  
TS checks whether predicted SQL behaves like gold across multiple perturbed DB replicas, reducing lucky execution.

**Technical description**  
`test_suite_accuracy_for_item` executes both queries on each TS DB and compares results, using ordered comparison only when ORDER BY is present in gold SQL.

**Code locations**  
`nl2sql/eval.py` (`test_suite_accuracy_for_item`, `_run_select_ts`, `_results_match_ts`)  
`notebooks/03_agentic_eval.ipynb` (TS engine factory and evaluation loop)

**Justification**  
Zhong et al. (2020) define distilled test suites; this project implements a lightweight suite-based approximation. The trade-off is reliance on perturbation quality rather than full distillation.

---

### Result Logging (JSON serialization)

**Plain-language**  
Evaluation results are saved to JSON so they can be inspected and plotted later.

**Technical description**  
The notebook save block uses `json.dumps(..., default=str)` to handle Decimal values returned by SQLAlchemy in TS debug samples.

**Code locations**  
`notebooks/03_agentic_eval.ipynb` (evaluation save block)

**Justification**  
This avoids failing the run at the final save step while preserving debug information. The trade-off is that some numeric debug fields are stored as strings.
