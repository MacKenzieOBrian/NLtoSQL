# DATA (what I need to track)

Quick recap for the reader: the ClassicModels test set anchors both evaluation and few-shot exemplars. Exemplars are only for **prompt conditioning** at inference; nothing here is used to train or update weights. Validation against the live DB keeps VA/EX honest.

## Datasets
- Train: NLQ-SQL pairs for classicmodels with joins/aggregations/filters/single-table coverage. Include schema bits in the text field for SFT.
- Test: 200 NLQ-SQL pairs held out for VA/EX/TS.
- Distilled DBs: schema-identical classicmodels variants with different data for TS.
- Schema cache: JSON dump from `list_tables` + `get_table_columns` for prompts.
- Current artifact: `data/classicmodels_test_200.json` (static, hand-curated coverage; no generator needed).

## Format
- JSON or Parquet (HF-friendly).
- Fields: `nlq`, `sql`, optional `schema_context`, optional `text` (schema + NLQ + SQL).

## How to build it
- Source: classicmodels MySQL schema.
- Coverage: one-to-many/many-to-many joins, SUM/COUNT/AVG, filters, ORDER BY/LIMIT, GROUP/HAVING.
- Synthetic boost (optional): use a stronger model; document prompts/filters if I do. For now, the 200 test pairs are fixed in JSON.
- QC: lint SQL, validate on live DB, dedup similar questions.

## Tracking
- Record dataset versions/hashes/generation scripts in LOGBOOK + commits.
- Note any manual edits or filters I apply.

## Validation
- Use the notebook helper `validate_test_set("data/classicmodels_test_200.json")` to run the 200 queries against the live DB and spot any failures. Use `limit=` for a quick smoke test. Latest runs: 200/200 success (VS Code + ADC, and Colab after env/ADC setup).

## Benchmark Usage (how data is used)
- `data/classicmodels_test_200.json` serves as the evaluation benchmark and the source of few-shot exemplars (sampled).  
- Exemplars are used **only for prompt conditioning** during inference; they are not training data.  
- This keeps model parameters frozen; differences between zero-shot and few-shot runs reflect prompt conditioning, not learning.

## Validation Process
- Gold SQL in the test set is validated against the live ClassicModels DB to ensure reference correctness and avoid false negatives in EX scoring.  
- VA/EX are computed by executing generated SQL via QueryRunner and comparing to validated gold SQL.  
- Schema caches and column ordering (PKs first, then identifier/name-like columns) are used at prompt time to reduce column-selection ambiguity.
