# DATA (what I need to track)

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
- Use the notebook helper `validate_test_set("data/classicmodels_test_200.json")` to run the 200 queries against the live DB and spot any failures. Use `limit=` for a quick smoke test.
